"""Thinking-model ceiling on AIME 2024 + AIME 2025.

Companion to `eval_math500_thinking_ceiling.py` — same hybrid Qwen3-30B-A3B
with `enable_thinking=True`, evaluated on the AIME slice that
`eval_aime_vibecheck.py` uses.

This gives us a non-saturated benchmark (thinking on MATH500 stratified-50
already hits pass@3 = 1.0, i.e. ceiling). On AIME both models fail often,
so:
  - pass@k curves have room to grow
  - diversity measurements are stress-tested on problems where neither
    model is converging on a single known solution
  - potential for pass@k crossover (panel's semantic spread may yield
    wider coverage where no single strategy dominates)

Uses the SAME 20-problem AIME slice as `eval_aime_vibecheck.py`:
AIME24 first 10 + AIME25 first 10.

Usage:
    python scripts/eval_aime_thinking_ceiling.py tag=aime_thinking_n16 n_samples=16
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

# Reuse AIME grader (integer-exact match, with <answer>/\boxed{}/fallback priority).
from scripts.eval_aime_vibecheck import extract_int  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s:%(lineno)d [%(levelname)s] %(message)s",
)
logger = logging.getLogger("eval_aime_thinking_ceiling")


# Stop on Qwen3 turn-end. Hybrid thinking model emits <think>...</think>
# then the final answer then <|im_end|>.
STOP_SEQUENCES: list[str] = ["<|im_end|>"]


def check_correct_aime(completion: str, gold: str) -> bool:
    try:
        gold_int = int(str(gold).strip())
    except ValueError:
        return False
    pred = extract_int(completion)
    return pred is not None and pred == gold_int


class ThinkingAIMEEnv(Env):
    """Single-turn eval env for the hybrid thinking model on AIME.

    No system prompt — the model's own post-training handles the
    <think>...</think>answer format given `enable_thinking=True`.
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
        correct = check_correct_aime(text, self.gold)
        return StepResult(
            reward=float(correct),
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=STOP_SEQUENCES,
            metrics={"correct": float(correct)},
        )


@dataclass(frozen=True)
class ThinkingAIMEGroupBuilder(EnvGroupBuilder):
    env_thunk: Callable[[], ThinkingAIMEEnv]
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
        return ["aime_thinking_ceiling", f"aime{self.year}", f"p{self.idx}"]


@chz.chz
class EvalConfig:
    tag: str = "aime_thinking_n16"
    base_model: str = "Qwen/Qwen3-30B-A3B"
    n_per_year: int = 10
    n_samples: int = 16
    max_tokens: int = 16384
    temperature: float = 1.0
    # AIME thinking traces are long. Keep concurrent problems modest.
    concurrent_problems: int = 4
    out_root: str = str(ROOT / "reports" / "eval_aime_vibecheck")


async def _eval(cfg: EvalConfig) -> None:
    out_dir = Path(cfg.out_root) / cfg.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    rollouts_path = out_dir / "rollouts.jsonl"
    summary_path = out_dir / "summary.json"

    logger.info("tag=%s  base_model=%s", cfg.tag, cfg.base_model)
    logger.info("n_per_year=%d  n_samples=%d  T=%.2f  max_tokens=%d",
                cfg.n_per_year, cfg.n_samples, cfg.temperature, cfg.max_tokens)
    logger.info("out_dir=%s", out_dir)

    tokenizer = get_tokenizer(cfg.base_model)

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

    builders = [
        ThinkingAIMEGroupBuilder(
            env_thunk=partial(ThinkingAIMEEnv, r["problem"], r["gold"], tokenizer),
            num_envs=cfg.n_samples,
            year=r["year"],
            idx=r["idx"],
        )
        for r in rows
    ]

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
                    pred = extract_int(text)
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

    summary = {
        "tag": cfg.tag,
        "variant": "thinking_native",
        "base_model": cfg.base_model,
        "system_prompt": None,
        "chat_template": "Qwen3-30B-A3B hybrid via tokenizer.apply_chat_template(enable_thinking=True)",
        "n_problems": len(per_problem),
        "n_samples_per_problem": cfg.n_samples,
        "total_samples": total_samples,
        "n_correct_samples": total_correct,
        "avg_hit_rate": avg_rate,
        "wilson_ci95": wilson,
        "pass_at_n_count": pass_n,
        "pass_at_n_rate": pass_n / len(per_problem) if per_problem else 0.0,
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
    logger.info("wrote %s", summary_path)


if __name__ == "__main__":
    cfg = chz.entrypoint(EvalConfig)
    asyncio.run(_eval(cfg))
