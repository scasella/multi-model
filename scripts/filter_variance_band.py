"""Per-model variance-band filter for the olympiad pool.

For one model (panel checkpoint OR Qwen3-30B-A3B native thinking), sample
G completions per problem at T=1.0, grade each with math_verify, and
classify the problem into one of:

    all_zero      (n_correct = 0)     — below the variance band
    variance_band (0 < n_correct < G) — trainable
    all_one       (n_correct = G)     — above the variance band

After running this for both arms, `intersect_variance_bands.py` takes
the intersection of the variance bands.

Inputs:
    --pool data/olympiad_pool/all.jsonl
    --arm {panel,thinking}
        panel:    Qwen/Qwen3-30B-A3B-Base + panel-RL checkpoint + <mutipersonaDebate> scaffold
        thinking: Qwen/Qwen3-30B-A3B + apply_chat_template(enable_thinking=True)
    --tag <name>   label for the output folder
    --n-samples G  default 8
    --temperature  default 1.0
    [--checkpoint-path PATH]  overrides the default panel checkpoint
    [--max-problems N]        cap for smoke testing

Outputs (in reports/variance_band/<tag>/):
    per_problem.jsonl    { uid, source, year, n_samples, n_correct, band }
    rollouts.jsonl       raw completions (one line per sample)
    summary.json         counts by band, by source
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tinker

from math_verify import parse as mv_parse, verify as mv_verify
from tinker_cookbook.completers import StopCondition, TinkerTokenCompleter
from tinker_cookbook.rl.rollouts import do_group_rollout
from tinker_cookbook.rl.types import (
    Action,
    ActionExtra,
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
    StepResult,
    Trajectory,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

# Panel scaffold regex + prompt (reuse).
from envs.multipersona_gsm8k import (
    PROMPT_TEMPLATE as PANEL_PROMPT,
    STOP_SEQUENCES as PANEL_STOP,
    extract_answer_text as panel_extract_answer,
)
from envs.multipersona_math import extract_last_boxed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s:%(lineno)d [%(levelname)s] %(message)s",
)
logger = logging.getLogger("filter_variance_band")


DEFAULT_PANEL_CHECKPOINT = os.environ.get(
    "PANEL_MATH_CHECKPOINT_SAMPLER",
    # Sampler weights from your post-MATH panel RL run.
    # Override via PANEL_MATH_CHECKPOINT_SAMPLER env var or --checkpoint-path CLI flag.
    "tinker://<your-panel-math-session>:train:0/sampler_weights/final",
)

THINKING_STOP = ["<|im_end|>"]


# ----------------------------------------------------------------------
# Grading (math_verify, lenient): accept integer-exact AND LaTeX-equiv.
# ----------------------------------------------------------------------


def grade(completion: str, gold: str, arm: str) -> bool:
    """Return True iff completion's final answer matches gold.

    For 'panel', prefer the <answer>...</answer> body; fall back to the
    last \\boxed{} anywhere.
    For 'thinking', use the last \\boxed{} (native Qwen3 thinking format).
    """
    candidates: list[str] = []
    if arm == "panel":
        body = panel_extract_answer(completion)
        if body:
            candidates.append(body)
            inner = extract_last_boxed(body)
            if inner:
                candidates.append(inner)
    # Always also try the last \boxed{} from the whole completion.
    inner_full = extract_last_boxed(completion)
    if inner_full:
        candidates.append(inner_full)
    candidates.append(completion)  # last resort

    try:
        gold_parsed = mv_parse("$" + gold + "$")
    except Exception:
        return False

    for cand in candidates:
        for wrapped in (f"${cand}$", cand):
            try:
                pred = mv_parse(wrapped)
                if mv_verify(gold_parsed, pred):
                    return True
            except Exception:
                continue
    return False


# ----------------------------------------------------------------------
# Env: a single-turn sampling env. No reward shaping here — we only need
# rollouts for grading.
# ----------------------------------------------------------------------


class PoolProblemEnv(Env):
    def __init__(self, problem: str, gold: str, tokenizer, arm: str):
        self.problem = problem
        self.gold = gold
        self.tokenizer = tokenizer
        self.arm = arm

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        if self.arm == "panel":
            prompt_text = PANEL_PROMPT.format(problem=self.problem)
            tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            return tinker.ModelInput.from_ints(tokens), PANEL_STOP
        else:  # thinking
            messages = [{"role": "user", "content": self.problem}]
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=True,
                tokenize=False,
            )
            tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            return tinker.ModelInput.from_ints(list(tokens)), THINKING_STOP

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        text = self.tokenizer.decode(action)
        correct = grade(text, self.gold, self.arm)
        return StepResult(
            reward=float(correct),
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=PANEL_STOP if self.arm == "panel" else THINKING_STOP,
            metrics={"correct": float(correct)},
        )


@dataclass(frozen=True)
class PoolGroupBuilder(EnvGroupBuilder):
    env_thunk: Callable[[], PoolProblemEnv]
    num_envs: int
    uid: str

    async def make_envs(self) -> Sequence[Env]:
        return [self.env_thunk() for _ in range(self.num_envs)]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        return [(0.0, {}) for _ in trajectory_group]

    def logging_tags(self) -> list[str]:
        return ["variance_band", self.uid]


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


async def run(args: argparse.Namespace) -> None:
    out_dir = ROOT / "reports" / "variance_band" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    per_problem_path = out_dir / "per_problem.jsonl"
    rollouts_path = out_dir / "rollouts.jsonl"
    summary_path = out_dir / "summary.json"

    pool_path = Path(args.pool)
    rows: list[dict] = []
    with open(pool_path) as fh:
        for line in fh:
            rows.append(json.loads(line))
    if args.max_problems:
        rows = rows[: args.max_problems]
    logger.info("pool: %d problems from %s", len(rows), pool_path)
    logger.info("arm=%s  G=%d  T=%.2f  max_tokens=%d",
                args.arm, args.n_samples, args.temperature, args.max_tokens)

    # Model selection.
    if args.arm == "panel":
        base_model = "Qwen/Qwen3-30B-A3B-Base"
        checkpoint = args.checkpoint_path or DEFAULT_PANEL_CHECKPOINT
    else:
        base_model = "Qwen/Qwen3-30B-A3B"
        checkpoint = None

    tokenizer = get_tokenizer(base_model)
    service_client = tinker.ServiceClient()
    sampling_client = (
        service_client.create_sampling_client(model_path=checkpoint)
        if checkpoint
        else service_client.create_sampling_client(base_model=base_model)
    )
    policy = TinkerTokenCompleter(
        sampling_client=sampling_client,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    builders = [
        PoolGroupBuilder(
            env_thunk=partial(PoolProblemEnv, r["problem"], r["gold"], tokenizer, args.arm),
            num_envs=args.n_samples,
            uid=r["uid"],
        )
        for r in rows
    ]

    per_problem: list[dict] = []
    t0 = time.time()
    with open(rollouts_path, "w") as fh_roll:
        for wstart in range(0, len(builders), args.concurrent_problems):
            wend = min(wstart + args.concurrent_problems, len(builders))
            logger.info("window %d..%d / %d", wstart, wend - 1, len(builders))
            groups = await asyncio.gather(
                *[do_group_rollout(gb, policy) for gb in builders[wstart:wend]]
            )
            for r, group in zip(rows[wstart:wend], groups, strict=True):
                n_correct = 0
                for si, traj in enumerate(group.trajectories_G):
                    tokens: list[int] = []
                    for tr in traj.transitions:
                        tokens.extend(tr.ac.tokens)
                    text = tokenizer.decode(tokens)
                    correct = grade(text, r["gold"], args.arm)
                    n_correct += int(correct)
                    fh_roll.write(json.dumps({
                        "uid": r["uid"],
                        "sample_idx": si,
                        "gold": r["gold"],
                        "correct": bool(correct),
                        "completion_tokens": len(tokens),
                        "completion": text,
                    }) + "\n")
                if n_correct == 0:
                    band = "all_zero"
                elif n_correct == args.n_samples:
                    band = "all_one"
                else:
                    band = "variance_band"
                per_problem.append({
                    "uid": r["uid"],
                    "source": r["source"],
                    "year": r.get("year"),
                    "n_samples": args.n_samples,
                    "n_correct": n_correct,
                    "band": band,
                })
                logger.info("  %s: %d/%d  (%s)", r["uid"], n_correct, args.n_samples, band)
            logger.info("  window done; elapsed=%.0fs", time.time() - t0)

    with open(per_problem_path, "w") as f:
        for p in per_problem:
            f.write(json.dumps(p) + "\n")

    band_counts: dict[str, int] = {}
    by_source: dict[str, dict[str, int]] = {}
    for p in per_problem:
        band_counts[p["band"]] = band_counts.get(p["band"], 0) + 1
        by_source.setdefault(p["source"], {})
        by_source[p["source"]][p["band"]] = by_source[p["source"]].get(p["band"], 0) + 1

    summary = {
        "tag": args.tag,
        "arm": args.arm,
        "base_model": base_model,
        "checkpoint": checkpoint,
        "n_problems": len(per_problem),
        "n_samples_per_problem": args.n_samples,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "bands": band_counts,
        "bands_by_source": by_source,
        "elapsed_sec": time.time() - t0,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("=== summary ===")
    logger.info("%s", json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT / "data/olympiad_pool/all.jsonl"))
    ap.add_argument("--arm", choices=["panel", "thinking"], required=True)
    ap.add_argument("--tag", required=True, help="output folder name under reports/variance_band/")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--concurrent-problems", type=int, default=8)
    ap.add_argument("--checkpoint-path", default=None,
                    help="Override the default panel checkpoint (ignored for 'thinking').")
    ap.add_argument("--max-problems", type=int, default=None)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run(args))
