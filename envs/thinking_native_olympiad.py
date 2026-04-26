"""Qwen3 native-thinking RL env on the olympiad union pool.

Same data as envs.multipersona_olympiad (both arms read
data/olympiad_pool/train.jsonl + heldout_eval.jsonl). Two differences:

  1. Prompt construction uses the model's native chat template with
     `enable_thinking=True`. No custom `<mutipersonaDebate>` scaffold.
     The model produces `<think>…</think>…<|im_end|>` in its own format.
  2. Grader extracts the final answer from the last `\\boxed{}` in the
     completion (standard Qwen3-thinking post-`</think>` convention),
     falling back to raw-completion math_verify. No tag-structure
     reward — the native template already enforces structure.

Reward = 1.0 if correct, else 0.0. No format penalty (the native
template is already well-structured; adding a tag_coef here would
be apples-to-oranges vs the panel arm).
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

from envs.olympiad_pool import (
    DEFAULT_HELDOUT_PATH,
    DEFAULT_TRAIN_PATH,
    RoundRobinSourceSampler,
    grade_math_verify,
    load_pool_rows,
)

logger = logging.getLogger(__name__)

STOP_SEQUENCES: list[str] = ["<|im_end|>"]


# ----------------------------------------------------------------------
# Env
# ----------------------------------------------------------------------


class ThinkingNativeOlympiadEnv(Env):
    def __init__(self, problem: str, gold: str, tokenizer):
        self.problem = problem
        self.gold = gold
        self.tokenizer = tokenizer

    def _prompt_tokens(self) -> list[int]:
        messages = [{"role": "user", "content": self.problem}]
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=True,
            tokenize=False,
        )
        return self.tokenizer.encode(prompt_text, add_special_tokens=False)

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        tokens = self._prompt_tokens()
        return tinker.ModelInput.from_ints(list(tokens)), STOP_SEQUENCES

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        text = self.tokenizer.decode(action)
        # No answer_body for native thinking — just try the last \boxed{} and the raw text.
        correct = grade_math_verify(answer_body=None, completion_fallback=text, gold=self.gold)
        reward = float(correct)

        with logtree.scope_header("Prompt"):
            logtree.log_text(self.problem)
        with logtree.scope_header("Completion"):
            logtree.log_text(text)
        with logtree.scope_header("Reward"):
            logtree.table_from_dict(
                {"gold": self.gold, "correct": correct, "reward": f"{reward:.3f}"},
                caption="thinking_native_olympiad reward",
            )

        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=STOP_SEQUENCES,
            metrics={"correct": float(correct)},
        )


@dataclass(frozen=True)
class ThinkingNativeOlympiadGroupBuilder(EnvGroupBuilder):
    env_thunk: Callable[[], ThinkingNativeOlympiadEnv]
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
        return ["olympiad_thinking", f"src_{self.source}", self.uid]


# ----------------------------------------------------------------------
# Train + heldout datasets
# ----------------------------------------------------------------------


class ThinkingNativeOlympiadTrainDataset(RLDataset):
    def __init__(
        self,
        rows: list[dict],
        batch_size: int,
        group_size: int,
        tokenizer,
        seed: int,
        virtual_len: int = 100_000,
    ):
        self.batch_size = batch_size
        self.group_size = group_size
        self.tokenizer = tokenizer
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

    def _make_builder(self, row: dict, group_size: int) -> ThinkingNativeOlympiadGroupBuilder:
        return ThinkingNativeOlympiadGroupBuilder(
            env_thunk=partial(
                ThinkingNativeOlympiadEnv, row["problem"], row["gold"], self.tokenizer
            ),
            num_envs=group_size,
            source=row["source"],
            uid=row["uid"],
        )


class ThinkingNativeOlympiadHeldoutDataset(RLDataset):
    def __init__(self, rows: list[dict], batch_size: int, tokenizer):
        self.rows = rows
        self.batch_size = batch_size
        self.group_size = 1
        self.tokenizer = tokenizer

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.rows))
        if start >= end:
            return []
        return [self._make_builder(self.rows[i]) for i in range(start, end)]

    def __len__(self) -> int:
        return math.ceil(len(self.rows) / self.batch_size)

    def _make_builder(self, row: dict) -> ThinkingNativeOlympiadGroupBuilder:
        return ThinkingNativeOlympiadGroupBuilder(
            env_thunk=partial(
                ThinkingNativeOlympiadEnv, row["problem"], row["gold"], self.tokenizer
            ),
            num_envs=1,
            source=row["source"],
            uid=row["uid"],
        )


# ----------------------------------------------------------------------
# Builder
# ----------------------------------------------------------------------


@chz.chz
class ThinkingNativeOlympiadDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    group_size: int
    seed: int = 0
    train_path: str = str(DEFAULT_TRAIN_PATH)
    heldout_path: str = str(DEFAULT_HELDOUT_PATH)
    virtual_train_len: int = 100_000

    async def __call__(
        self,
    ) -> tuple[ThinkingNativeOlympiadTrainDataset, ThinkingNativeOlympiadHeldoutDataset]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        train_rows = load_pool_rows(Path(self.train_path))
        heldout_rows = load_pool_rows(Path(self.heldout_path))
        train = ThinkingNativeOlympiadTrainDataset(
            rows=train_rows,
            batch_size=self.batch_size,
            group_size=self.group_size,
            tokenizer=tokenizer,
            seed=self.seed,
            virtual_len=self.virtual_train_len,
        )
        heldout = ThinkingNativeOlympiadHeldoutDataset(
            rows=heldout_rows,
            batch_size=self.batch_size,
            tokenizer=tokenizer,
        )
        return train, heldout
