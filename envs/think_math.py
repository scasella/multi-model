"""Matched <think> control — MATH stage.

Direct mirror of envs/multipersona_math.py with the think scaffold.

Initialization: the stage-1 think-GSM8K-trained checkpoint (`load_checkpoint_path`
at the launcher level). Continues LoRA training on the same MATH train
distribution (round-robin category-balanced, 7 subjects) used by the
panel MATH run, with MATH-500 excluded from the held-out eval.

Everything except the scaffold (template, tag regex, tag_valid check)
matches the panel MATH env. Correctness grader (`math_verify`), gold
extraction (`extract_last_boxed`), dataset loader, eval builder — all
reused unchanged.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import cast

import chz
import tinker
from datasets import Dataset, load_dataset
from math_verify import parse as mv_parse, verify as mv_verify

from tinker_cookbook.completers import StopCondition
from tinker_cookbook.rl.types import (
    Action,
    ActionExtra,
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
    Trajectory,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import logtree

# Scaffold-specific bits: mirror the panel-MATH design but import from
# the think-scaffold GSM8K module.
from envs.think_gsm8k import (
    PROMPT_TEMPLATE,
    STOP_SEQUENCES,
    _ANSWER_CLOSE,
    check_tag_structure,
    extract_answer_text,
)

# Scaffold-independent bits (reused from the panel MATH env).
from envs.multipersona_math import (
    SUBJECTS,
    extract_last_boxed,
    _normalize_level,
    _load_subject_train_rows,
)

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Correctness grader (math_verify) — scaffold-independent but uses this
# module's extract_answer_text, so we redefine to pin that dependency.
# -------------------------------------------------------------------------


def check_correct_math(completion: str, gold: str) -> bool:
    body = extract_answer_text(completion)
    if body is None:
        return False
    try:
        gold_parsed = mv_parse("$" + gold + "$")
    except Exception:
        return False

    candidates: list[str] = [body]
    inner = extract_last_boxed(body)
    if inner is not None:
        candidates.append(inner)

    for cand in candidates:
        try:
            pred = mv_parse("$" + cand + "$")
            if mv_verify(gold_parsed, pred):
                return True
        except Exception:
            pass

    try:
        pred = mv_parse(completion)
        if mv_verify(gold_parsed, pred):
            return True
    except Exception:
        pass

    return False


# -------------------------------------------------------------------------
# Env
# -------------------------------------------------------------------------


class ThinkMathEnv(Env):
    """Single-turn MATH env with <think> scaffold."""

    def __init__(
        self,
        problem: str,
        gold: str,
        tokenizer,
        *,
        tag_coef: float = 0.2,
    ):
        self.problem = problem
        self.gold = gold
        self.tokenizer = tokenizer
        self.tag_coef = tag_coef

    def _prompt_text(self) -> str:
        return PROMPT_TEMPLATE.format(problem=self.problem)

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        prompt = self._prompt_text()
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        return tinker.ModelInput.from_ints(tokens), STOP_SEQUENCES

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        text = self.tokenizer.decode(action)
        stop_reason = (extra or {}).get("stop_reason") if extra else None
        text_for_grading = text + _ANSWER_CLOSE if stop_reason == "stop" else text

        tag_valid = check_tag_structure(text_for_grading)
        correct = check_correct_math(text_for_grading, self.gold)
        reward = float(correct) + self.tag_coef * (float(tag_valid) - 1.0)

        with logtree.scope_header("Prompt"):
            logtree.log_text(self._prompt_text())
        with logtree.scope_header("Completion"):
            logtree.log_text(text)
        with logtree.scope_header("Reward"):
            logtree.table_from_dict(
                {
                    "gold": self.gold,
                    "tag_valid": tag_valid,
                    "correct": correct,
                    "reward": f"{reward:.3f}",
                    "tag_coef": self.tag_coef,
                },
                caption="think_math reward",
            )

        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=STOP_SEQUENCES,
            metrics={
                "correct": float(correct),
                "tag_valid": float(tag_valid),
            },
        )


# -------------------------------------------------------------------------
# Group builder
# -------------------------------------------------------------------------


@dataclass(frozen=True)
class ThinkMathGroupBuilder(EnvGroupBuilder):
    env_thunk: Callable[[], ThinkMathEnv]
    num_envs: int
    subject: str
    level: str

    async def make_envs(self) -> Sequence[Env]:
        return [self.env_thunk() for _ in range(self.num_envs)]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        return [(0.0, {}) for _ in trajectory_group]

    def logging_tags(self) -> list[str]:
        return ["math_train_think", f"subj_{self.subject}", f"level_{self.level}"]


# -------------------------------------------------------------------------
# Round-robin train dataset (category-balanced)
# -------------------------------------------------------------------------


class RoundRobinThinkMathTrainDataset(RLDataset):
    """Category-balanced round-robin training dataset, <think> variant.

    Identical data and ordering to envs.multipersona_math; only the env
    (and thus prompt template + tag checks) differs.
    """

    def __init__(
        self,
        batch_size: int,
        group_size: int,
        tokenizer,
        seed: int,
        tag_coef: float,
        virtual_len: int = 100_000,
        n_train_per_subject: int | None = None,
    ):
        import random

        self.batch_size = batch_size
        self.group_size = group_size
        self.tokenizer = tokenizer
        self.tag_coef = tag_coef
        self.virtual_len = virtual_len
        self._rng = random.Random(seed)

        self.per_subject: dict[str, list[dict]] = {}
        total = 0
        for subj in SUBJECTS:
            rows = _load_subject_train_rows(subj)
            if n_train_per_subject is not None:
                rows = rows[:n_train_per_subject]
            self._rng.shuffle(rows)
            self.per_subject[subj] = rows
            total += len(rows)
            logger.info(f"  loaded subject {subj}: {len(rows)} rows")
        logger.info(
            f"MATH train pool (think): {total} rows across {len(SUBJECTS)} subjects "
            f"(per-subject range: {min(len(v) for v in self.per_subject.values())}"
            f"..{max(len(v) for v in self.per_subject.values())})"
        )

        self._ptrs: dict[str, int] = {s: 0 for s in SUBJECTS}
        self._rotation_idx = 0

    def _next_row(self, subject: str) -> dict:
        rows = self.per_subject[subject]
        if not rows:
            raise RuntimeError(f"No rows available for subject {subject!r}")
        if self._ptrs[subject] >= len(rows):
            self._rng.shuffle(rows)
            self._ptrs[subject] = 0
        row = rows[self._ptrs[subject]]
        self._ptrs[subject] += 1
        return row

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        out: list[EnvGroupBuilder] = []
        for _ in range(self.batch_size):
            subj = SUBJECTS[self._rotation_idx % len(SUBJECTS)]
            self._rotation_idx += 1
            row = self._next_row(subj)
            out.append(self._make_builder(row, self.group_size))
        return out

    def __len__(self) -> int:
        return self.virtual_len

    def _make_builder(self, row: dict, group_size: int) -> ThinkMathGroupBuilder:
        return ThinkMathGroupBuilder(
            env_thunk=partial(
                ThinkMathEnv,
                row["problem"],
                row["gold"],
                self.tokenizer,
                tag_coef=self.tag_coef,
            ),
            num_envs=group_size,
            subject=row["subject"],
            level=row["level"],
        )


# -------------------------------------------------------------------------
# Holdout eval dataset (think variant of MathHoldoutEvalDataset)
# -------------------------------------------------------------------------


class ThinkMathHoldoutEvalDataset(RLDataset):
    """256-problem MATH-test slice (MATH-500 excluded), think variant."""

    def __init__(
        self,
        batch_size: int,
        tokenizer,
        seed: int,
        tag_coef: float,
        n_examples: int = 256,
    ):
        import random

        self.batch_size = batch_size
        self.group_size = 1
        self.tokenizer = tokenizer
        self.tag_coef = tag_coef

        m500 = cast(Dataset, load_dataset("HuggingFaceH4/MATH-500", split="test"))
        exclude: set[str] = {r["problem"] for r in m500}

        rows: list[dict] = []
        for subj in SUBJECTS:
            ds = cast(
                Dataset,
                load_dataset("EleutherAI/hendrycks_math", subj, split="test"),
            )
            for row in ds:
                if row["problem"] in exclude:
                    continue
                gold = extract_last_boxed(row.get("solution") or "")
                if gold is None or not gold.strip():
                    continue
                rows.append(
                    {
                        "problem": row["problem"],
                        "gold": gold,
                        "level": _normalize_level(row.get("level")),
                        "subject": subj,
                    }
                )
        rng = random.Random(seed + 1)
        rng.shuffle(rows)
        self.rows = rows[:n_examples]
        logger.info(
            f"MATH holdout eval (think): {len(self.rows)} problems "
            f"(from {len(rows)} available post-exclusion; "
            f"{len(exclude)} MATH-500 problems excluded)"
        )

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.rows))
        if start >= end:
            return []
        return [self._make_builder(self.rows[i]) for i in range(start, end)]

    def __len__(self) -> int:
        return math.ceil(len(self.rows) / self.batch_size)

    def _make_builder(self, row: dict) -> ThinkMathGroupBuilder:
        return ThinkMathGroupBuilder(
            env_thunk=partial(
                ThinkMathEnv,
                row["problem"],
                row["gold"],
                self.tokenizer,
                tag_coef=self.tag_coef,
            ),
            num_envs=1,
            subject=row["subject"],
            level=row["level"],
        )


# -------------------------------------------------------------------------
# Builder
# -------------------------------------------------------------------------


@chz.chz
class ThinkMathDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    group_size: int
    seed: int = 0
    tag_coef: float = 0.2
    n_eval: int = 256
    n_train_per_subject: int | None = None
    virtual_train_len: int = 100_000

    async def __call__(
        self,
    ) -> tuple[RoundRobinThinkMathTrainDataset, ThinkMathHoldoutEvalDataset]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        train = RoundRobinThinkMathTrainDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            tokenizer=tokenizer,
            seed=self.seed,
            tag_coef=self.tag_coef,
            virtual_len=self.virtual_train_len,
            n_train_per_subject=self.n_train_per_subject,
        )
        test = ThinkMathHoldoutEvalDataset(
            batch_size=self.batch_size,
            tokenizer=tokenizer,
            seed=self.seed,
            tag_coef=self.tag_coef,
            n_examples=self.n_eval,
        )
        return train, test
