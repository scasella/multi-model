"""Vibe-check transfer eval on AIME 2024 + AIME 2025.

Samples n=8 completions per problem across 20 AIME problems
(10 AIME24 + 10 AIME25, first-by-index from each set) at T=1.0 against
a single trained adapter, under two prompt variants:

  panel     — full multi-persona panel template (the training prompt)
  no_panel  — stripped User/Assistant framing requesting <answer> tags;
              same surrounding User/Assistant framing the RL'd adapter
              saw during training, but without any panel scaffolding.

Reports:
  - overall hit rate (correct-sample rate) with Wilson 95% binomial CI
  - pass@8: #problems where at least one of the 8 samples was correct
  - per-problem avg and correct count

This is a *handful-sized* vibe check. 20 AIME problems × 8 samples gives
a 160-sample pool per variant; Wilson CI widths at low rates will still
be wide (±5–10pp typical). Do not over-interpret.

Usage:
    python scripts/eval_aime_vibecheck.py \\
        sampler_path=tinker://<sess>:train:0/sampler_weights/final \\
        variant=panel  tag=final_panel
    python scripts/eval_aime_vibecheck.py \\
        sampler_path=tinker://<sess>:train:0/sampler_weights/final \\
        variant=no_panel  tag=final_nopanel

Outputs:
    reports/eval_aime_vibecheck/<tag>/summary.json
    reports/eval_aime_vibecheck/<tag>/rollouts.jsonl
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import chz
import tinker
from datasets import load_dataset

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s:%(lineno)d [%(levelname)s] %(message)s",
)
logger = logging.getLogger("eval_aime_vibecheck")


# -------------------------------------------------------------------------
# Prompt templates
# -------------------------------------------------------------------------

PANEL_TEMPLATE = (
    "A conversation between User and Multi-Persona Panel of Experts. "
    "The user asks a question, and the Multi-Persona Panel of Experts solves it. "
    "The Multi-Persona Panel of Experts first deliberates and debates the reasoning process "
    "with each other and then provides the user with the answer. "
    "The deliberation process and answer are enclosed within "
    "<mutipersonaDebate>...</mutipersonaDebate> and <answer>...</answer> tags, respectively, i.e., "
    "<mutipersonaDebate> deliberation process here </mutipersonaDebate> "
    "<answer>answer here </answer>. "
    "User: {problem}. Assistant: "
)

# Same User/Assistant framing as training, but all panel scaffolding removed.
# Still requests an <answer> tag so the grader has a clean signal — otherwise
# we'd be benchmarking answer extraction, not reasoning.
NO_PANEL_TEMPLATE = (
    "User: {problem}\n\n"
    "Please solve this problem. State your final integer answer inside "
    "<answer>...</answer> tags.\n\n"
    "Assistant: "
)

STOP_SEQUENCES: list[str] = ["</answer>"]


# -------------------------------------------------------------------------
# Answer extraction and grading for AIME (integer answers 0-999)
# -------------------------------------------------------------------------

_ANSWER_RE = re.compile(r"<answer>(?P<body>.*?)(?:</answer>|$)", re.DOTALL)
_BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]*)\}")
_INT_RE = re.compile(r"-?\d+")


def extract_int(completion: str) -> int | None:
    """Extract an integer answer.

    Priority:
      1. Last integer inside the first <answer>...</answer> block
      2. Last integer inside any \\boxed{...}
      3. Last integer anywhere in the completion (fallback)
    """
    candidates: list[str] = []
    m = _ANSWER_RE.search(completion)
    if m:
        candidates.append(m.group("body"))
    candidates.extend(_BOXED_RE.findall(completion))
    candidates.append(completion)
    for s in candidates:
        nums = _INT_RE.findall(s)
        if not nums:
            continue
        try:
            return int(nums[-1])
        except ValueError:
            continue
    return None


def check_correct_aime(completion: str, gold: str) -> bool:
    """Integer-exact match against gold AIME answer (gold is a string like '204')."""
    try:
        gold_int = int(str(gold).strip())
    except ValueError:
        return False
    pred = extract_int(completion)
    return pred is not None and pred == gold_int


# -------------------------------------------------------------------------
# Env + GroupBuilder
# -------------------------------------------------------------------------


class AIMEEnv(Env):
    """Single-turn env: prompt → rollout → integer-match reward.

    The reward is just correctness (no tag_coef penalty here — this is eval,
    not training).
    """

    def __init__(
        self,
        problem: str,
        gold: str,
        tokenizer,
        template: str,
    ):
        self.problem = problem
        self.gold = gold
        self.tokenizer = tokenizer
        self.template = template

    def _prompt_text(self) -> str:
        return self.template.format(problem=self.problem)

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        prompt = self._prompt_text()
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        return tinker.ModelInput.from_ints(tokens), STOP_SEQUENCES

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        text = self.tokenizer.decode(action)
        stop_reason = (extra or {}).get("stop_reason") if extra else None
        text_for_grading = text + "</answer>" if stop_reason == "stop" else text
        correct = check_correct_aime(text_for_grading, self.gold)
        return StepResult(
            reward=float(correct),
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=STOP_SEQUENCES,
            metrics={"correct": float(correct)},
        )


@dataclass(frozen=True)
class AIMEGroupBuilder(EnvGroupBuilder):
    env_thunk: Callable[[], AIMEEnv]
    num_envs: int
    year: int
    idx: int

    async def make_envs(self) -> Sequence[Env]:
        return [self.env_thunk() for _ in range(self.num_envs)]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        return [(0.0, {}) for _ in trajectory_group]

    def logging_tags(self) -> list[str]:
        return ["aime_vibecheck", f"aime{self.year}", f"p{self.idx}"]


# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------


@chz.chz
class EvalConfig:
    sampler_path: str
    variant: Literal["panel", "no_panel"]
    tag: str
    model_name: str = "Qwen/Qwen3-30B-A3B-Base"
    n_per_year: int = 10
    n_samples: int = 8
    max_tokens: int = 8192
    temperature: float = 1.0
    # Number of problems rolled out concurrently. With n_samples=8 each,
    # 5 problems → 40 concurrent sampling requests.
    concurrent_problems: int = 5
    out_root: str = str(ROOT / "reports" / "eval_aime_vibecheck")


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------


async def _eval(cfg: EvalConfig) -> None:
    out_dir = Path(cfg.out_root) / cfg.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    rollouts_path = out_dir / "rollouts.jsonl"
    summary_path = out_dir / "summary.json"

    logger.info("variant=%s  tag=%s  n_per_year=%d  n_samples=%d",
                cfg.variant, cfg.tag, cfg.n_per_year, cfg.n_samples)
    logger.info("sampler_path=%s", cfg.sampler_path)
    logger.info("temperature=%g  max_tokens=%d", cfg.temperature, cfg.max_tokens)
    logger.info("out_dir=%s", out_dir)

    template = PANEL_TEMPLATE if cfg.variant == "panel" else NO_PANEL_TEMPLATE
    tokenizer = get_tokenizer(cfg.model_name)

    # --- Load problems.
    ds24 = load_dataset("HuggingFaceH4/aime_2024", split="train")
    ds25 = load_dataset("yentinglin/aime_2025", split="train")
    rows: list[dict] = []
    for i in range(cfg.n_per_year):
        r = ds24[i]
        rows.append({"year": 2024, "idx": i, "problem": r["problem"], "gold": str(r["answer"])})
    for i in range(cfg.n_per_year):
        r = ds25[i]
        rows.append({"year": 2025, "idx": i, "problem": r["problem"], "gold": str(r["answer"])})
    logger.info("problems: %d (AIME24×%d + AIME25×%d)", len(rows), cfg.n_per_year, cfg.n_per_year)

    # --- Build group builders (one per problem, group_size=n_samples).
    builders = [
        AIMEGroupBuilder(
            env_thunk=partial(AIMEEnv, r["problem"], r["gold"], tokenizer, template),
            num_envs=cfg.n_samples,
            year=r["year"],
            idx=r["idx"],
        )
        for r in rows
    ]

    # --- Sampling client from checkpoint.
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(model_path=cfg.sampler_path)
    policy = TinkerTokenCompleter(
        sampling_client=sampling_client,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )

    # --- Roll out problems in small concurrent windows.
    per_problem: list[dict] = []
    t_start = time.time()
    with open(rollouts_path, "w") as fh:
        for wstart in range(0, len(builders), cfg.concurrent_problems):
            wend = min(wstart + cfg.concurrent_problems, len(builders))
            window_builders = builders[wstart:wend]
            window_meta = rows[wstart:wend]
            logger.info("window %d..%d / %d", wstart, wend - 1, len(builders))

            groups = await asyncio.gather(
                *[do_group_rollout(gb, policy) for gb in window_builders]
            )

            for r, group in zip(window_meta, groups, strict=True):
                n_correct = 0
                for si, traj in enumerate(group.trajectories_G):
                    tokens: list[int] = []
                    for tr in traj.transitions:
                        tokens.extend(tr.ac.tokens)
                    text = tokenizer.decode(tokens)
                    text_for_grading = text
                    if "</answer>" not in text_for_grading and "<answer>" in text_for_grading:
                        text_for_grading = text + "</answer>"
                    pred = extract_int(text_for_grading)
                    try:
                        gold_int = int(r["gold"])
                        correct = pred is not None and pred == gold_int
                    except ValueError:
                        correct = False
                    n_correct += int(correct)
                    fh.write(json.dumps({
                        "year": r["year"],
                        "problem_idx": r["idx"],
                        "sample_idx": si,
                        "gold": r["gold"],
                        "pred": pred,
                        "correct": bool(correct),
                        "completion_tokens": len(tokens),
                        "completion_chars": len(text),
                        "completion": text,
                    }) + "\n")
                per_problem.append({
                    "year": r["year"],
                    "problem_idx": r["idx"],
                    "gold": r["gold"],
                    "n_samples": cfg.n_samples,
                    "n_correct": n_correct,
                    "avg": n_correct / cfg.n_samples,
                    "pass_at_n": bool(n_correct > 0),
                })
                logger.info("  AIME%d #%d  correct=%d/%d", r["year"], r["idx"], n_correct, cfg.n_samples)
            elapsed = time.time() - t_start
            logger.info("  window done; elapsed=%.0fs", elapsed)

    # --- Aggregate.
    total_correct = sum(p["n_correct"] for p in per_problem)
    total_samples = sum(p["n_samples"] for p in per_problem)
    avg_rate = total_correct / total_samples if total_samples else 0.0

    # Wilson 95% binomial CI.
    z = 1.96
    n = total_samples
    if n > 0:
        p = avg_rate
        denom = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        half = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / denom
        wilson = [center - half, center + half]
    else:
        wilson = [0.0, 0.0]

    pass_n = sum(1 for p in per_problem if p["pass_at_n"])
    n_probs = len(per_problem)

    summary = {
        "tag": cfg.tag,
        "variant": cfg.variant,
        "sampler_path": cfg.sampler_path,
        "model_name": cfg.model_name,
        "n_problems": n_probs,
        "n_samples_per_problem": cfg.n_samples,
        "total_samples": total_samples,
        "n_correct_samples": total_correct,
        "avg_hit_rate": avg_rate,
        "wilson_ci95": wilson,
        "pass_at_n_count": pass_n,
        "pass_at_n_rate": pass_n / n_probs if n_probs else 0.0,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "per_problem": per_problem,
        "elapsed_sec": time.time() - t_start,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("=== FINAL for tag=%s variant=%s ===", cfg.tag, cfg.variant)
    logger.info("avg_hit_rate=%.4f  Wilson95=[%.4f, %.4f]   (%d/%d)",
                avg_rate, wilson[0], wilson[1], total_correct, total_samples)
    logger.info("pass@%d: %d/%d problems (%.2f)",
                cfg.n_samples, pass_n, n_probs, pass_n / n_probs if n_probs else 0.0)
    logger.info("wrote %s", summary_path)
    logger.info("wrote %s", rollouts_path)


if __name__ == "__main__":
    cfg = chz.entrypoint(EvalConfig)
    asyncio.run(_eval(cfg))
