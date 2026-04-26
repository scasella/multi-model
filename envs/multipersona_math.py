"""Multi-Persona Panel of Experts — RL continuation on the MATH train split.

Initialization: loads weights from the GSM8K-trained LoRA checkpoint
(batch 128, run e1a9b8bf-...). Continues training the same rank-32 LoRA
with a fresh optimizer state on MATH train problems.

Dataset: `EleutherAI/hendrycks_math`, 7 subjects loaded separately and
served in strict round-robin order so every training batch carries equal
representation from each subject. Within a subject, problems are drawn
from a shuffled deque; when exhausted, the deque is reshuffled and
cycled — so every problem appears ~evenly often, regardless of subject
pool size.

Prompt template: identical to the GSM8K training scaffold (imported
verbatim from envs.multipersona_gsm8k). Only the grader differs.

Gold extraction: the last brace-balanced `\\boxed{...}` in the
`solution` field. Rows without a parseable boxed answer are dropped at
load time.

Correctness grader: `math_verify.parse` + `verify` for LaTeX-equivalent
matching (handles fractions, radicals, `\\boxed`, 0.5 ↔ 1/2, etc.).
This is the same grader used in `scripts/eval_math500_vibecheck.py`.

Reward shaping (unchanged from GSM8K run):
    reward = correct_answer + tag_coef * (tag_structure - 1)
with tag_coef = 0.2.
"""
from __future__ import annotations

import logging
import math
import random
import re
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

# Reuse the exact prompt template, stop sequences, tag regex, and tag
# structure check from the GSM8K env. The only thing that changes here
# is the correctness grader.
from envs.multipersona_gsm8k import (
    PROMPT_TEMPLATE,
    STOP_SEQUENCES,
    _ANSWER_CLOSE,
    check_tag_structure,
    extract_answer_text,
)

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Dataset constants
# -------------------------------------------------------------------------

# EleutherAI/hendrycks_math subset names (snake_case). These correspond
# 1:1 with MATH's `type` field values (in title-case).
SUBJECTS: list[str] = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


# -------------------------------------------------------------------------
# Gold extraction from MATH solutions
# -------------------------------------------------------------------------


def extract_last_boxed(text: str) -> str | None:
    """Return the contents of the last `\\boxed{...}` in `text`, brace-balanced.

    Handles nested braces (e.g. `\\boxed{\\frac{1}{2}}`). Returns None if no
    balanced boxed expression is present.
    """
    if not text:
        return None
    key = "\\boxed{"
    idx = text.rfind(key)
    if idx < 0:
        return None
    i = idx + len(key)
    depth = 1
    start = i
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
        i += 1
    return None


# -------------------------------------------------------------------------
# Correctness grader (math_verify)
# -------------------------------------------------------------------------


def check_correct_math(completion: str, gold: str) -> bool:
    """True iff the `<answer>…</answer>` body matches gold via math_verify.

    Tries: (a) the extracted answer body as-is; (b) the boxed expression
    within the answer body, if any; (c) the whole completion as last-
    resort (math_verify can sometimes detect `\\boxed{}` in raw form).
    """
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

    # Last resort: pass math_verify the full completion (it can detect
    # raw `\\boxed{}` without the dollar wrapping).
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


class MultipersonaMathEnv(Env):
    """Single-turn env: prompt → rollout → (correctness + tag-structure) reward.

    Identical scaffolding to MultipersonaGsm8kEnv; only the grader differs.
    """

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
        # Re-append </answer> if the sampler stopped on it (sampler strips
        # the stop sequence from the returned action).
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
                caption="multipersona_math reward",
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
class MultipersonaMathGroupBuilder(EnvGroupBuilder):
    env_thunk: Callable[[], MultipersonaMathEnv]
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
        return ["math_train", f"subj_{self.subject}", f"level_{self.level}"]


# -------------------------------------------------------------------------
# Round-robin, category-balanced training dataset
# -------------------------------------------------------------------------


def _normalize_level(raw: object) -> str:
    """MATH level field is like 'Level 3' or 'Level ?' — normalize to digit str or '?'."""
    s = str(raw or "").strip()
    s = s.replace("Level", "").strip()
    return s or "?"


def _load_subject_train_rows(subject: str) -> list[dict]:
    """Load one MATH subject's train split as a list of {problem, gold, level, subject}."""
    ds: Dataset = cast(
        Dataset,
        load_dataset("EleutherAI/hendrycks_math", subject, split="train"),
    )
    rows: list[dict] = []
    dropped = 0
    for row in ds:
        gold = extract_last_boxed(row.get("solution") or "")
        if gold is None or not gold.strip():
            dropped += 1
            continue
        rows.append(
            {
                "problem": row["problem"],
                "gold": gold,
                "level": _normalize_level(row.get("level")),
                "subject": subject,
            }
        )
    if dropped:
        logger.info(f"  {subject}: dropped {dropped} rows without parseable \\boxed{{}}")
    return rows


class RoundRobinMathTrainDataset(RLDataset):
    """Category-balanced round-robin training dataset.

    Maintains one shuffled deque per subject. For each batch of
    `batch_size` groups, rotates through SUBJECTS one slot at a time,
    pulling the next problem from that subject's deque (reshuffled and
    cycled when exhausted). With batch_size=16 and 7 subjects, each
    batch contains 2 groups from each of 7 subjects + 2 extras from the
    top of the rotation — which rotates across batches, so over 7 batches
    every subject has contributed 16 groups (perfectly balanced over
    longer windows).

    `__len__` returns a large virtual length; the trainer stops at
    `max_steps`. This matches the "infinite sampling with replacement"
    semantics the user requested ('sample each category equally and
    randomize').
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
            f"MATH train pool: {total} rows across {len(SUBJECTS)} subjects "
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

    def _make_builder(self, row: dict, group_size: int) -> MultipersonaMathGroupBuilder:
        return MultipersonaMathGroupBuilder(
            env_thunk=partial(
                MultipersonaMathEnv,
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
# In-training eval dataset (MATH test minus MATH-500)
# -------------------------------------------------------------------------


class MathHoldoutEvalDataset(RLDataset):
    """A fixed held-out slice of MATH-test that EXCLUDES all MATH-500 problems.

    This keeps MATH-500 clean as a final post-training eval while still
    letting us watch an in-distribution held-out curve during training.
    """

    def __init__(
        self,
        batch_size: int,
        tokenizer,
        seed: int,
        tag_coef: float,
        n_examples: int = 256,
    ):
        self.batch_size = batch_size
        self.group_size = 1
        self.tokenizer = tokenizer
        self.tag_coef = tag_coef

        # Collect problems-to-exclude from MATH-500 (by problem text).
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
        rng = random.Random(seed + 1)  # distinct stream from train shuffle
        rng.shuffle(rows)
        self.rows = rows[:n_examples]
        logger.info(
            f"MATH holdout eval: {len(self.rows)} problems "
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

    def _make_builder(self, row: dict) -> MultipersonaMathGroupBuilder:
        return MultipersonaMathGroupBuilder(
            env_thunk=partial(
                MultipersonaMathEnv,
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
class MultipersonaMathDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    group_size: int
    seed: int = 0
    tag_coef: float = 0.2
    n_eval: int = 256
    # Primarily for smoke testing — cap each subject's train pool.
    n_train_per_subject: int | None = None
    virtual_train_len: int = 100_000

    async def __call__(
        self,
    ) -> tuple[RoundRobinMathTrainDataset, MathHoldoutEvalDataset]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        train = RoundRobinMathTrainDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            tokenizer=tokenizer,
            seed=self.seed,
            tag_coef=self.tag_coef,
            virtual_len=self.virtual_train_len,
            n_train_per_subject=self.n_train_per_subject,
        )
        test = MathHoldoutEvalDataset(
            batch_size=self.batch_size,
            tokenizer=tokenizer,
            seed=self.seed,
            tag_coef=self.tag_coef,
            n_examples=self.n_eval,
        )
        return train, test
