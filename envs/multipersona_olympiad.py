"""Panel-of-experts RL env on the olympiad union pool.

Reads:
    data/olympiad_pool/train.jsonl
    data/olympiad_pool/heldout_eval.jsonl

Scaffold: `<mutipersonaDebate>…</mutipersonaDebate><answer>…</answer>`,
identical to envs.multipersona_math (same prompt + tag regex + reward
shaping). Only the data source and the grader wrapper differ:
we use `math_verify` through envs.olympiad_pool.grade_math_verify,
which tries the answer body, the last `\\boxed{}`, and the raw
completion as candidates. This makes the grader robust across
integer-answer (AIME/HMMT) and LaTeX-answer (OlympiadBench) sources.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import chz
import tinker

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

from envs.multipersona_gsm8k import (
    PROMPT_TEMPLATE,
    STOP_SEQUENCES,
    _ANSWER_CLOSE,
    check_tag_structure,
    extract_answer_text,
)
from envs.olympiad_pool import (
    DEFAULT_HELDOUT_PATH,
    DEFAULT_TRAIN_PATH,
    RoundRobinSourceSampler,
    grade_math_verify,
    load_pool_rows,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Env
# ----------------------------------------------------------------------


class PanelOlympiadEnv(Env):
    def __init__(self, problem: str, gold: str, tokenizer, *, tag_coef: float = 0.2):
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
        body = extract_answer_text(text_for_grading)
        correct = grade_math_verify(body, text_for_grading, self.gold)
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
                caption="panel_olympiad reward",
            )

        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=STOP_SEQUENCES,
            metrics={"correct": float(correct), "tag_valid": float(tag_valid)},
        )


@dataclass(frozen=True)
class PanelOlympiadGroupBuilder(EnvGroupBuilder):
    env_thunk: Callable[[], PanelOlympiadEnv]
    num_envs: int
    source: str
    uid: str

    async def make_envs(self) -> Sequence[Env]:
        return [self.env_thunk() for _ in range(self.num_envs)]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        return [(0.0, {}) for _ in trajectory_group]

    def logging_tags(self) -> list[str]:
        return ["olympiad_panel", f"src_{self.source}", self.uid]


# ----------------------------------------------------------------------
# Train dataset (round-robin over sources)
# ----------------------------------------------------------------------


class PanelOlympiadTrainDataset(RLDataset):
    def __init__(
        self,
        rows: list[dict],
        batch_size: int,
        group_size: int,
        tokenizer,
        seed: int,
        tag_coef: float,
        virtual_len: int = 100_000,
    ):
        self.batch_size = batch_size
        self.group_size = group_size
        self.tokenizer = tokenizer
        self.tag_coef = tag_coef
        self.virtual_len = virtual_len
        self.sampler = RoundRobinSourceSampler(rows, seed=seed)

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        out: list[EnvGroupBuilder] = []
        for _ in range(self.batch_size):
            row = self.sampler.next_row()
            out.append(self._make_builder(row, self.group_size))
        return out

    def __len__(self) -> int:
        return self.virtual_len

    def _make_builder(self, row: dict, group_size: int) -> PanelOlympiadGroupBuilder:
        return PanelOlympiadGroupBuilder(
            env_thunk=partial(
                PanelOlympiadEnv,
                row["problem"],
                row["gold"],
                self.tokenizer,
                tag_coef=self.tag_coef,
            ),
            num_envs=group_size,
            source=row["source"],
            uid=row["uid"],
        )


# ----------------------------------------------------------------------
# Heldout eval (static, one sample per problem)
# ----------------------------------------------------------------------


class PanelOlympiadHeldoutDataset(RLDataset):
    def __init__(self, rows: list[dict], batch_size: int, tokenizer, tag_coef: float):
        self.rows = rows
        self.batch_size = batch_size
        self.group_size = 1
        self.tokenizer = tokenizer
        self.tag_coef = tag_coef

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.rows))
        if start >= end:
            return []
        return [self._make_builder(self.rows[i]) for i in range(start, end)]

    def __len__(self) -> int:
        return math.ceil(len(self.rows) / self.batch_size)

    def _make_builder(self, row: dict) -> PanelOlympiadGroupBuilder:
        return PanelOlympiadGroupBuilder(
            env_thunk=partial(
                PanelOlympiadEnv,
                row["problem"],
                row["gold"],
                self.tokenizer,
                tag_coef=self.tag_coef,
            ),
            num_envs=1,
            source=row["source"],
            uid=row["uid"],
        )


# ----------------------------------------------------------------------
# Builder (chz entry point for the trainer)
# ----------------------------------------------------------------------


@chz.chz
class PanelOlympiadDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    group_size: int
    seed: int = 0
    tag_coef: float = 0.2
    train_path: str = str(DEFAULT_TRAIN_PATH)
    heldout_path: str = str(DEFAULT_HELDOUT_PATH)
    virtual_train_len: int = 100_000

    async def __call__(
        self,
    ) -> tuple[PanelOlympiadTrainDataset, PanelOlympiadHeldoutDataset]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        train_rows = load_pool_rows(Path(self.train_path))
        heldout_rows = load_pool_rows(Path(self.heldout_path))
        train = PanelOlympiadTrainDataset(
            rows=train_rows,
            batch_size=self.batch_size,
            group_size=self.group_size,
            tokenizer=tokenizer,
            seed=self.seed,
            tag_coef=self.tag_coef,
            virtual_len=self.virtual_train_len,
        )
        heldout = PanelOlympiadHeldoutDataset(
            rows=heldout_rows,
            batch_size=self.batch_size,
            tokenizer=tokenizer,
            tag_coef=self.tag_coef,
        )
        return train, heldout
