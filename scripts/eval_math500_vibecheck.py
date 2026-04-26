"""Vibe-check transfer eval on MATH500 (stratified sample).

Samples n=4 completions per problem across 50 MATH500 problems (10 per
level, levels 1–5) at T=1.0 against a single trained adapter, under two
prompt variants:

  panel     — full multi-persona panel template (the training prompt)
  no_panel  — stripped User/Assistant framing requesting <answer> tags

Reports:
  - overall hit rate (correct-sample rate) with Wilson 95% binomial CI
  - pass@n: #problems where at least one of the n samples was correct
  - per-level breakdown
  - per-problem avg and correct count

Grading uses math_verify.parse/verify — handles \\boxed{} extraction,
LaTeX normalization, and form equivalence.

Usage:
    python scripts/eval_math500_vibecheck.py \\
        sampler_path=tinker://<sess>:train:0/sampler_weights/final \\
        variant=panel tag=final_panel
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
from typing import Literal

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import chz
import tinker
from datasets import load_dataset
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s:%(lineno)d [%(levelname)s] %(message)s",
)
logger = logging.getLogger("eval_math500_vibecheck")


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

NO_PANEL_TEMPLATE = (
    "User: {problem}\n\n"
    "Please solve this problem. Put your final answer inside "
    "<answer>...</answer> tags, using LaTeX if needed (e.g. \\frac{{3}}{{4}}).\n\n"
    "Assistant: "
)

STOP_SEQUENCES: list[str] = ["</answer>"]


# -------------------------------------------------------------------------
# Grading
# -------------------------------------------------------------------------


def check_correct_math(completion: str, gold: str) -> bool:
    """Use math_verify to compare prediction to gold, tolerant of LaTeX form.

    math_verify.parse handles \\boxed{...} extraction automatically.
    We additionally try the <answer>...</answer> body as a fallback.
    """
    # Try full completion first — math_verify finds \boxed{} anywhere.
    try:
        gold_parsed = mv_parse("$" + gold + "$")
    except Exception:
        return False

    try:
        pred_parsed = mv_parse(completion)
        if mv_verify(gold_parsed, pred_parsed):
            return True
    except Exception:
        pass

    # Fallback: extract <answer>...</answer> body and re-try.
    import re
    m = re.search(r"<answer>(.*?)(?:</answer>|$)", completion, re.DOTALL)
    if m and m.group(1).strip():
        body = m.group(1).strip()
        try:
            pred_parsed = mv_parse(body)
            if mv_verify(gold_parsed, pred_parsed):
                return True
            # Also try wrapping as LaTeX-math so parse treats it as expression.
            pred_parsed = mv_parse("$" + body + "$")
            if mv_verify(gold_parsed, pred_parsed):
                return True
        except Exception:
            pass
    return False


# -------------------------------------------------------------------------
# Env + GroupBuilder
# -------------------------------------------------------------------------


class MathEnv(Env):
    def __init__(self, problem: str, gold: str, tokenizer, template: str):
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
        correct = check_correct_math(text_for_grading, self.gold)
        return StepResult(
            reward=float(correct),
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=STOP_SEQUENCES,
            metrics={"correct": float(correct)},
        )


@dataclass(frozen=True)
class MathGroupBuilder(EnvGroupBuilder):
    env_thunk: Callable[[], MathEnv]
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
        return ["math500_vibecheck", f"level{self.level}", self.subject]


@chz.chz
class EvalConfig:
    sampler_path: str
    variant: Literal["panel", "no_panel"]
    tag: str
    model_name: str = "Qwen/Qwen3-30B-A3B-Base"
    n_per_level: int | None = 10  # 10/level × 5 = 50; 0/None = all at each level
    # If set, only use problems at this level (e.g. 5 for hard-problem sweep).
    level_filter: int | None = None
    n_samples: int = 4
    max_tokens: int = 4096
    temperature: float = 1.0
    concurrent_problems: int = 8
    seed: int = 20260423
    out_root: str = str(ROOT / "reports" / "eval_math500_vibecheck")


def _pick_stratified(
    ds,
    n_per_level: int | None,
    seed: int,
    level_filter: int | None = None,
) -> list[dict]:
    """Deterministically pick the first `n_per_level` problems for each of
    levels 1..5. Falls back to taking whatever is available at each level.

    - ``n_per_level=None`` (or 0) means "take all problems" at each level.
    - ``level_filter`` restricts to a single level (used for hard-problem
      L5 sweeps). When set, returns *every* problem at that level unless
      ``n_per_level`` also caps it.
    """
    by_level: dict[str, list[dict]] = {}
    for i, row in enumerate(ds):
        by_level.setdefault(str(row["level"]).strip(), []).append({**row, "_orig_idx": i})
    levels = [str(level_filter)] if level_filter is not None else ["1", "2", "3", "4", "5"]
    rows: list[dict] = []
    for lv in levels:
        pool = by_level.get(lv, [])
        if n_per_level is None or n_per_level == 0:
            rows.extend(pool)
        else:
            rows.extend(pool[:n_per_level])
    return rows


async def _eval(cfg: EvalConfig) -> None:
    out_dir = Path(cfg.out_root) / cfg.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    rollouts_path = out_dir / "rollouts.jsonl"
    summary_path = out_dir / "summary.json"

    logger.info("variant=%s  tag=%s  n_per_level=%d  n_samples=%d",
                cfg.variant, cfg.tag, cfg.n_per_level, cfg.n_samples)
    logger.info("sampler_path=%s", cfg.sampler_path)
    logger.info("out_dir=%s", out_dir)

    template = PANEL_TEMPLATE if cfg.variant == "panel" else NO_PANEL_TEMPLATE
    tokenizer = get_tokenizer(cfg.model_name)

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows = _pick_stratified(ds, cfg.n_per_level, cfg.seed, level_filter=cfg.level_filter)
    levels_sampled = sorted({str(r["level"]).strip() for r in rows})
    logger.info("sampled %d problems across levels %s", len(rows), levels_sampled)

    builders = [
        MathGroupBuilder(
            env_thunk=partial(MathEnv, r["problem"], r["answer"], tokenizer, template),
            num_envs=cfg.n_samples,
            unique_id=r["unique_id"],
            level=str(r["level"]).strip(),
            subject=r["subject"],
        )
        for r in rows
    ]

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(model_path=cfg.sampler_path)
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
                    text_for_grading = text
                    if "</answer>" not in text_for_grading and "<answer>" in text_for_grading:
                        text_for_grading = text + "</answer>"
                    correct = check_correct_math(text_for_grading, r["answer"])
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
                logger.info("  %-20s L%-1s  correct=%d/%d",
                            r["unique_id"], r["level"], n_correct, cfg.n_samples)
            elapsed = time.time() - t_start
            logger.info("  window done; elapsed=%.0fs", elapsed)

    # Aggregate.
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

    # Per-level aggregates.
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
        "variant": cfg.variant,
        "sampler_path": cfg.sampler_path,
        "model_name": cfg.model_name,
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
    logger.info("=== FINAL for tag=%s variant=%s ===", cfg.tag, cfg.variant)
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
