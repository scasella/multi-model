"""Matched <think> control for the Multi-Persona Panel of Experts experiment.

Identical pure-RL-from-base recipe as envs/multipersona_gsm8k.py — same
base model, same LoRA rank, same RL algorithm, same hyperparameters,
same GSM8K dataset, same grader, same tag_coef, same stop conditions,
same reward shape. The ONLY deliberate difference is the scaffold:

  panel:  <mutipersonaDebate>...</mutipersonaDebate><answer>...</answer>
  think:  <think>...</think><answer>...</answer>

Prompt template (verbatim DeepSeek-R1-Zero phrasing, matched in
structure/length/depth to the panel template):

    A conversation between User and Assistant. The user asks a question,
    and the Assistant solves it. The Assistant first thinks about the
    reasoning process in its mind and then provides the user with the
    answer. The reasoning process and answer are enclosed within
    <think>...</think> and <answer>...</answer> tags, respectively,
    i.e., <think> reasoning process here </think> <answer>answer here
    </answer>. User: {problem}. Assistant:

Purpose: this is the scientific control for the scaffold-vs-scaffold
comparison. We already know panel > no_panel under our adapter. We need
this control to know whether panel > matched-think under the same RL
recipe — i.e. is it "multipersona specifically" or "any structured
scaffold" that produces the observed RL gains.
"""
from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Literal, cast

import chz
import tinker
from datasets import Dataset, load_dataset

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

# Universal grader utilities (scaffold-independent).
from envs.multipersona_gsm8k import (
    _normalize_answer,
    _NUMBER_RE,
    extract_gsm8k_gold,
)

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Prompt template + scaffold tags (the only deliberate deviation from
# envs/multipersona_gsm8k.py)
# -------------------------------------------------------------------------

PROMPT_TEMPLATE = (
    "A conversation between User and Assistant. "
    "The user asks a question, and the Assistant solves it. "
    "The Assistant first thinks about the reasoning process in its mind "
    "and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within "
    "<think>...</think> and <answer>...</answer> tags, respectively, i.e., "
    "<think> reasoning process here </think> "
    "<answer>answer here </answer>. "
    "User: {problem}. Assistant: "
)

STOP_SEQUENCES: list[str] = ["</answer>"]

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_ANSWER_OPEN = "<answer>"
_ANSWER_CLOSE = "</answer>"

_TAG_RE = re.compile(
    r"<think>(?P<think>.*?)</think>\s*<answer>(?P<answer>.*?)(?:</answer>|$)",
    re.DOTALL,
)


def check_tag_structure(completion: str) -> bool:
    """True iff the completion contains a non-empty
    <think>...</think><answer>...</answer> pair in that order.
    (</answer> may be stripped by the stop sequence; end-of-string accepted.)"""
    m = _TAG_RE.search(completion)
    if not m:
        return False
    return bool(m.group("think").strip()) and bool(m.group("answer").strip())


def extract_answer_text(completion: str) -> str | None:
    """Return the text inside the first <answer>...</answer> block, or None."""
    m = _TAG_RE.search(completion)
    if m and m.group("answer").strip():
        return m.group("answer").strip()
    i = completion.find(_ANSWER_OPEN)
    if i < 0:
        return None
    rest = completion[i + len(_ANSWER_OPEN):]
    j = rest.find(_ANSWER_CLOSE)
    body = rest[:j] if j >= 0 else rest
    body = body.strip()
    return body or None


def check_correct(completion: str, gold: str) -> bool:
    """True iff <answer>…</answer> numerically matches gold. Same GSM8K
    numeric grader logic as the panel env (scaffold-independent)."""
    text = extract_answer_text(completion)
    if text is None:
        return False
    pred_norm = _normalize_answer(text)
    gold_norm = _normalize_answer(gold)
    if pred_norm == gold_norm:
        return True

    def last_number(x: str) -> float | None:
        nums = _NUMBER_RE.findall(x)
        if not nums:
            return None
        try:
            return float(nums[-1])
        except ValueError:
            return None

    pv = last_number(pred_norm)
    gv = last_number(gold_norm)
    if pv is None or gv is None:
        return False
    if pv == gv:
        return True
    return math.isclose(pv, gv, rel_tol=0, abs_tol=1e-6)


# -------------------------------------------------------------------------
# Env (direct mirror of MultipersonaGsm8kEnv)
# -------------------------------------------------------------------------


class ThinkGsm8kEnv(Env):
    """Single-turn env: prompt → rollout → (correctness + tag-structure) reward."""

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
        correct = check_correct(text_for_grading, self.gold)
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
                caption="think reward",
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
# Group builder + dataset (direct mirror)
# -------------------------------------------------------------------------


@dataclass(frozen=True)
class ThinkGroupBuilder(EnvGroupBuilder):
    env_thunk: Callable[[], ThinkGsm8kEnv]
    num_envs: int
    dataset_name: str = "gsm8k_think"

    async def make_envs(self) -> Sequence[Env]:
        return [self.env_thunk() for _ in range(self.num_envs)]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        return [(0.0, {}) for _ in trajectory_group]

    def logging_tags(self) -> list[str]:
        return [self.dataset_name]


class ThinkGsm8kDataset(RLDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        tokenizer,
        split: Literal["train", "test"] = "train",
        seed: int = 0,
        n_examples: int | None = None,
        tag_coef: float = 0.2,
    ):
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        ds = cast(Dataset, load_dataset("openai/gsm8k", name="main", split=split))
        if split == "train":
            ds = ds.shuffle(seed=seed)
        if n_examples is not None:
            ds = ds.select(range(min(n_examples, len(ds))))
        self.ds = ds
        self.batch_size = batch_size
        self.group_size = group_size if split == "train" else 1
        self.tokenizer = tokenizer
        self.tag_coef = tag_coef

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.ds))
        assert start < end, f"Batch index {index} out of range"
        out = []
        for row in self.ds.select(range(start, end)):
            b = self._make_builder(row, self.group_size)
            if b is not None:
                out.append(b)
        return out

    def __len__(self) -> int:
        return math.ceil(len(self.ds) / self.batch_size)

    def _make_builder(self, x: dict, group_size: int) -> ThinkGroupBuilder | None:
        try:
            problem = x["question"]
            gold = extract_gsm8k_gold(x["answer"])
        except Exception as e:
            logger.warning(f"Failed to parse GSM8K row: {e}")
            return None
        return ThinkGroupBuilder(
            env_thunk=partial(
                ThinkGsm8kEnv, problem, gold, self.tokenizer, tag_coef=self.tag_coef
            ),
            num_envs=group_size,
        )


@chz.chz
class ThinkGsm8kDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    group_size: int
    seed: int = 0
    n_train: int | None = None
    n_test: int | None = None
    tag_coef: float = 0.2

    async def __call__(self) -> tuple[ThinkGsm8kDataset, ThinkGsm8kDataset]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        train = ThinkGsm8kDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            tokenizer=tokenizer,
            split="train",
            seed=self.seed,
            n_examples=self.n_train,
            tag_coef=self.tag_coef,
        )
        test = ThinkGsm8kDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            tokenizer=tokenizer,
            split="test",
            seed=self.seed,
            n_examples=self.n_test,
            tag_coef=self.tag_coef,
        )
        return train, test
