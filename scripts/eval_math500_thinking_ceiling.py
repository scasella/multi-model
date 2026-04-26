"""Thinking-model ceiling: Qwen3-30B-A3B (hybrid) on the same MATH500 sample.

This is the right reference for our <multipersonaDebate> panel scaffold:
a production-grade post-trained Qwen that natively emits
<think>...</think> before the answer. Orders of magnitude more
post-training than any matched RL-from-Base <think> control we could
run with our compute budget.

Implementation note: Tinker does not currently support sampling from
Qwen/Qwen3-30B-A3B-Thinking-2507, Qwen/QwQ-32B, or the DeepSeek-R1
distills (all return BadRequestError: "Sampling is not supported"). It
*does* support the ORIGINAL hybrid release `Qwen/Qwen3-30B-A3B`, which
is architecturally identical to the model family our panel adapter was
trained on (Qwen3-30B-A3B-Base → +panel-LoRA) and is post-trained by
Qwen themselves to toggle between thinking and non-thinking modes via
the `enable_thinking` flag in the chat template. That makes this
actually a cleaner comparison than -2507 would have been: same base
architecture, same parameter count, Qwen's own post-training vs. ours.

The hybrid chat template behavior with `add_generation_prompt=True`:
  - `enable_thinking=True` : assistant header is left empty; the model
    is trained to emit `<think>...</think>` itself, then the answer.
  - `enable_thinking=False`: template pre-injects an empty closed
    `<think></think>` block, suppressing thinking.

We use `enable_thinking=True` — this is the ceiling we want.

Runs the same 50-problem stratified MATH500 slice (10 per level × 5) at
n=4 samples.

Produces a single number directly comparable to:
  - reports/eval_math500_vibecheck/final_panel/         (pre-MATH panel adapter)
  - reports/eval_math500_vibecheck/final_no_panel/      (pre-MATH no-panel)
  - reports/eval_math500_vibecheck/math_final_panel/    (post-MATH panel adapter)
  - reports/eval_math500_vibecheck/math_final_no_panel/ (post-MATH no-panel)
  - reports/eval_math500_vibecheck/instruct_native/     (Instruct-2507 ceiling)

Usage:
    python scripts/eval_math500_thinking_ceiling.py tag=thinking_native
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

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

# Reuse the MATH grader + stratified sampler from the vibe-check script.
from scripts.eval_math500_vibecheck import check_correct_math, _pick_stratified  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s:%(lineno)d [%(levelname)s] %(message)s",
)
logger = logging.getLogger("eval_math500_thinking_ceiling")


# Stop on Qwen3 turn-end. The thinking template closes </think> then
# produces the final answer, then <|im_end|>.
STOP_SEQUENCES: list[str] = ["<|im_end|>"]


class ThinkingMathEnv(Env):
    """Single-turn eval env for a native-thinking model.

    We do NOT provide a system prompt. The hybrid Qwen3-30B-A3B is
    post-trained to, given `enable_thinking=True` in the chat template,
    emit `<think>...reasoning...</think>` followed by the final answer
    (typically `\\boxed{…}`). Injecting our own "show your work"
    instructions could interfere with that post-training. Just pass the
    problem.
    """

    def __init__(self, problem: str, gold: str, tokenizer):
        self.problem = problem
        self.gold = gold
        self.tokenizer = tokenizer

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        messages = [{"role": "user", "content": self.problem}]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=True, tokenize=False
        )
        tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        return tinker.ModelInput.from_ints(list(tokens)), STOP_SEQUENCES

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        text = self.tokenizer.decode(action)
        correct = check_correct_math(text, self.gold)
        return StepResult(
            reward=float(correct),
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=STOP_SEQUENCES,
            metrics={"correct": float(correct)},
        )


@dataclass(frozen=True)
class ThinkingMathGroupBuilder(EnvGroupBuilder):
    env_thunk: Callable[[], ThinkingMathEnv]
    num_envs: int
    unique_id: str
    level: str
    subject: str

    async def make_envs(self) -> Sequence[Env]:
        return [self.env_thunk() for _ in range(self.num_envs)]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        return [(0.0, {}) for _ in trajectory_group]

    def logging_tags(self) -> list[str]:
        return ["math500_thinking_ceiling", f"level{self.level}", self.subject]


@chz.chz
class EvalConfig:
    tag: str = "thinking_native"
    base_model: str = "Qwen/Qwen3-30B-A3B"
    n_per_level: int | None = 10   # None/0 = all at each level
    level_filter: int | None = None  # restrict to a single level (e.g. 5)
    n_samples: int = 4
    # Thinking traces can be very long — our probe at 256 tokens didn't
    # even finish the <think> block on a trivial arithmetic question.
    # 16k is a generous budget that still lets us parallelize.
    max_tokens: int = 16384
    # Match the rest of the eval suite (instruct ceiling, panel/no_panel
    # adapter variants all use T=1.0). Qwen recommends T=0.6 for thinking
    # mode at inference time — this gives a conservative ceiling estimate
    # vs. their recommended settings.
    temperature: float = 1.0
    # Thinking traces are much longer than instruct answers; halve the
    # concurrency to keep memory reasonable.
    concurrent_problems: int = 4
    seed: int = 20260423
    out_root: str = str(ROOT / "reports" / "eval_math500_vibecheck")


async def _eval(cfg: EvalConfig) -> None:
    out_dir = Path(cfg.out_root) / cfg.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    rollouts_path = out_dir / "rollouts.jsonl"
    summary_path = out_dir / "summary.json"

    logger.info("tag=%s  base_model=%s", cfg.tag, cfg.base_model)
    logger.info("n_per_level=%d  n_samples=%d  T=%.2f  max_tokens=%d",
                cfg.n_per_level, cfg.n_samples, cfg.temperature, cfg.max_tokens)
    logger.info("out_dir=%s", out_dir)

    tokenizer = get_tokenizer(cfg.base_model)

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows = _pick_stratified(ds, cfg.n_per_level, cfg.seed, level_filter=cfg.level_filter)
    levels_sampled = sorted({str(r["level"]).strip() for r in rows})
    logger.info("sampled %d problems across levels %s", len(rows), levels_sampled)

    builders = [
        ThinkingMathGroupBuilder(
            env_thunk=partial(ThinkingMathEnv, r["problem"], r["answer"], tokenizer),
            num_envs=cfg.n_samples,
            unique_id=r["unique_id"],
            level=str(r["level"]).strip(),
            subject=r["subject"],
        )
        for r in rows
    ]

    # Base-model sampling client — no adapter path.
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=cfg.base_model)
    policy = TinkerTokenCompleter(
        sampling_client=sampling_client,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )

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
                    correct = check_correct_math(text, r["answer"])
                    n_correct += int(correct)
                    fh.write(json.dumps({
                        "unique_id": r["unique_id"],
                        "level": str(r["level"]).strip(),
                        "subject": r["subject"],
                        "sample_idx": si,
                        "gold": r["answer"],
                        "correct": bool(correct),
                        "completion_tokens": len(tokens),
                        "completion": text,
                    }) + "\n")
                per_problem.append({
                    "unique_id": r["unique_id"],
                    "level": str(r["level"]).strip(),
                    "subject": r["subject"],
                    "gold": r["answer"],
                    "n_samples": cfg.n_samples,
                    "n_correct": n_correct,
                    "avg": n_correct / cfg.n_samples,
                    "pass_at_n": bool(n_correct > 0),
                })
                logger.info("  %-40s L%-1s  correct=%d/%d",
                            r["unique_id"], r["level"], n_correct, cfg.n_samples)
            elapsed = time.time() - t_start
            logger.info("  window done; elapsed=%.0fs", elapsed)

    total_correct = sum(p["n_correct"] for p in per_problem)
    total_samples = sum(p["n_samples"] for p in per_problem)
    avg_rate = total_correct / total_samples if total_samples else 0.0

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

    by_level: dict[str, dict] = {}
    for p in per_problem:
        lv = p["level"]
        d = by_level.setdefault(lv, {"n_probs": 0, "n_samples": 0, "n_correct": 0, "pass_n": 0})
        d["n_probs"] += 1
        d["n_samples"] += p["n_samples"]
        d["n_correct"] += p["n_correct"]
        d["pass_n"] += int(p["pass_at_n"])
    for lv, d in by_level.items():
        d["avg"] = d["n_correct"] / d["n_samples"] if d["n_samples"] else 0.0
        d["pass_rate"] = d["pass_n"] / d["n_probs"] if d["n_probs"] else 0.0

    summary = {
        "tag": cfg.tag,
        "variant": "thinking_native",
        "base_model": cfg.base_model,
        "system_prompt": None,
        "chat_template": "Qwen3-30B-A3B hybrid via tokenizer.apply_chat_template(enable_thinking=True); model emits <think>...</think> itself",
        "n_problems": len(per_problem),
        "n_samples_per_problem": cfg.n_samples,
        "total_samples": total_samples,
        "n_correct_samples": total_correct,
        "avg_hit_rate": avg_rate,
        "wilson_ci95": wilson,
        "pass_at_n_count": pass_n,
        "pass_at_n_rate": pass_n / len(per_problem) if per_problem else 0.0,
        "per_level": by_level,
        "per_problem": per_problem,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "elapsed_sec": time.time() - t_start,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("=== FINAL for tag=%s ===", cfg.tag)
    logger.info("avg_hit_rate=%.4f  Wilson95=[%.4f, %.4f]   (%d/%d)",
                avg_rate, wilson[0], wilson[1], total_correct, total_samples)
    logger.info("pass@%d: %d/%d problems (%.2f)",
                cfg.n_samples, pass_n, len(per_problem),
                pass_n / len(per_problem) if per_problem else 0.0)
    for lv in sorted(by_level):
        d = by_level[lv]
        logger.info("  level %s: n=%d probs (%d samples)  avg=%.3f  pass@%d=%d/%d",
                    lv, d["n_probs"], d["n_samples"], d["avg"],
                    cfg.n_samples, d["pass_n"], d["n_probs"])
    logger.info("wrote %s", summary_path)


if __name__ == "__main__":
    cfg = chz.entrypoint(EvalConfig)
    asyncio.run(_eval(cfg))
