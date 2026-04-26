"""Multi-Persona Panel of Experts — pure-RL-from-base on GSM8K.

Design intent (see reports/NEW_PREREGISTRATION.md for the full rationale):
- Follow the DeepSeek-R1-Zero recipe: pure RL from a Base model, no SFT
  warmup, outcome-based rewards only, with a prompt template that
  setup-shapes the rollout but does NOT impose a process reward.
- The one deliberate deviation from R1-Zero: instead of the singular
  "Assistant" with a <think>...</think><answer>...</answer> scaffold,
  the template posits a plural deliberator — a "Multi-Persona Panel of
  Experts" — with a <mutipersonaDebate>...</mutipersonaDebate><answer>...
  </answer> scaffold. This tests whether plural-actor framing under
  outcome-only RL induces emergent multi-voice reasoning.

Prompt template (verbatim, including the user's spelling of
`<mutipersonaDebate>` — with a single "mu", not "multi"):

    A conversation between User and Multi-Persona Panel of Experts.
    The user asks a question, and the Multi-Persona Panel of Experts
    solves it. The Multi-Persona Panel of Experts first deliberates
    and debates the reasoning process with each other and then
    provides the user with the answer. The deliberation process and
    answer are enclosed within <mutipersonaDebate>...</mutipersonaDebate>
    and <answer>...</answer> tags, respectively, i.e.,
    <mutipersonaDebate> deliberation process here </mutipersonaDebate>
    <answer>answer here </answer>. User: {problem}. Assistant:

Rewards (two components, outcome-based):
  1. `tag_structure`     : binary. The completion must contain a
                           well-formed <mutipersonaDebate>...
                           </mutipersonaDebate> block followed by a
                           <answer>...</answer> block, each with
                           non-empty interior, in that order.
  2. `correct_answer`    : binary. The value inside <answer> matches
                           the GSM8K gold numeric answer after light
                           normalization (strip whitespace, commas,
                           dollar signs, trailing period, optional
                           `\\boxed{...}` wrapper).

Combined scalar reward (ProblemEnv-style additive with format penalty):

    reward = correct_answer + tag_coef * (tag_structure - 1)

With tag_coef = 0.2 this gives:
    good tags + correct : 1.0
    good tags + wrong   : 0.0
    bad tags + correct  : 0.8   (unlikely — wrong-tag answers won't parse)
    bad tags + wrong    : -0.2
so the format signal is a small negative pull away from no-tag gibberish
without crowding out the outcome signal.
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

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Prompt template
# -------------------------------------------------------------------------

PROMPT_TEMPLATE = (
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

STOP_SEQUENCES: list[str] = ["</answer>"]

# -------------------------------------------------------------------------
# Answer extraction and correctness grading
# -------------------------------------------------------------------------

_DEBATE_OPEN = "<mutipersonaDebate>"
_DEBATE_CLOSE = "</mutipersonaDebate>"
_ANSWER_OPEN = "<answer>"
_ANSWER_CLOSE = "</answer>"

_TAG_RE = re.compile(
    r"<mutipersonaDebate>(?P<debate>.*?)</mutipersonaDebate>\s*<answer>(?P<answer>.*?)(?:</answer>|$)",
    re.DOTALL,
)


def check_tag_structure(completion: str) -> bool:
    """True iff the completion contains a non-empty
    <mutipersonaDebate>...</mutipersonaDebate><answer>...</answer>
    pair in that order.

    The answer block may be closed by </answer> OR by end-of-string
    (because </answer> is a stop sequence, the sampler will strip it
    from some completions; we accept either).
    """
    m = _TAG_RE.search(completion)
    if not m:
        return False
    return bool(m.group("debate").strip()) and bool(m.group("answer").strip())


def extract_answer_text(completion: str) -> str | None:
    """Return the raw text inside the first <answer>...</answer> block,
    or None if no such block exists. Accepts unclosed </answer> (stop-sequence-stripped)."""
    m = _TAG_RE.search(completion)
    if m and m.group("answer").strip():
        return m.group("answer").strip()
    # fallback: any <answer>…
    i = completion.find(_ANSWER_OPEN)
    if i < 0:
        return None
    rest = completion[i + len(_ANSWER_OPEN):]
    j = rest.find(_ANSWER_CLOSE)
    body = rest[:j] if j >= 0 else rest
    body = body.strip()
    return body or None


_BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]*)\}")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _normalize_answer(s: str) -> str:
    """Normalize a free-form answer string for GSM8K comparison."""
    s = s.strip()
    # unwrap \boxed{...}
    m = _BOXED_RE.search(s)
    if m:
        s = m.group(1).strip()
    # strip surrounding punctuation
    s = s.strip().strip(".")
    # drop a $ prefix / % suffix / common units
    if s.startswith("$"):
        s = s[1:]
    if s.endswith("%"):
        s = s[:-1]
    # drop thousands-commas and whitespace inside numbers
    s = s.replace(",", "").replace(" ", "")
    return s


def check_correct(completion: str, gold: str) -> bool:
    """True iff the <answer>…</answer> content numerically matches gold."""
    text = extract_answer_text(completion)
    if text is None:
        return False
    pred_norm = _normalize_answer(text)
    gold_norm = _normalize_answer(gold)

    # exact string match after normalization
    if pred_norm == gold_norm:
        return True

    # numeric match (int or float) — pull the last number from each
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
    # GSM8K answers are exact — no float tolerance beyond int equality
    if pv == gv:
        return True
    # allow tiny float slack (e.g. 3.0 vs 3)
    return math.isclose(pv, gv, rel_tol=0, abs_tol=1e-6)


def extract_gsm8k_gold(answer_field: str) -> str:
    """GSM8K reference field ends with '#### <answer>'. Return that tail."""
    # last line that starts with '####'
    for line in reversed(answer_field.splitlines()):
        s = line.strip()
        if s.startswith("####"):
            content = s[4:].strip().lstrip(":").strip()
            return content.replace(",", "")
    # fallback regex
    matches = re.findall(r"####\s*(.+)", answer_field)
    if matches:
        return matches[-1].strip().replace(",", "")
    raise ValueError(f"No GSM8K '####' marker in: {answer_field!r}")


# -------------------------------------------------------------------------
# Env
# -------------------------------------------------------------------------


class MultipersonaGsm8kEnv(Env):
    """Single-turn env: prompt → rollout → (correctness + tag-structure) reward."""

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
        # Qwen3-Base tokenizer: start from a clean slate (no BOS prepended in the
        # cookbook renderers for Qwen3; encode directly).
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        obs = tinker.ModelInput.from_ints(tokens)
        return obs, STOP_SEQUENCES

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        # Decode the action into text. We need to RE-APPEND the stop sequence because
        # the sampler strips it from `action` (it's still reflected in `extra.stop_reason`
        # if we want to know why it stopped).
        text = self.tokenizer.decode(action)
        # If the sampler hit `</answer>` it won't be in `text`; add it back so the
        # regex can find a closed <answer> block.
        stop_reason = (extra or {}).get("stop_reason") if extra else None
        # StopReason is Literal["length","stop"]; "stop" = hit a stop sequence.
        if stop_reason == "stop":
            text_for_grading = text + _ANSWER_CLOSE
        else:
            text_for_grading = text

        tag_valid = check_tag_structure(text_for_grading)
        correct = check_correct(text_for_grading, self.gold)

        # ProblemEnv-style additive reward with a format pull-down.
        reward = float(correct) + self.tag_coef * (float(tag_valid) - 1.0)

        # Log for training traces
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
                caption="multipersona reward",
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
# EnvGroupBuilder + Dataset
# -------------------------------------------------------------------------


@dataclass(frozen=True)
class MultipersonaGroupBuilder(EnvGroupBuilder):
    env_thunk: Callable[[], MultipersonaGsm8kEnv]
    num_envs: int
    dataset_name: str = "gsm8k_multipersona"

    async def make_envs(self) -> Sequence[Env]:
        return [self.env_thunk() for _ in range(self.num_envs)]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        return [(0.0, {}) for _ in trajectory_group]

    def logging_tags(self) -> list[str]:
        return [self.dataset_name]


class MultipersonaGsm8kDataset(RLDataset):
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

    def _make_builder(self, x: dict, group_size: int) -> MultipersonaGroupBuilder | None:
        try:
            problem = x["question"]
            gold = extract_gsm8k_gold(x["answer"])
        except Exception as e:
            logger.warning(f"Failed to parse GSM8K row: {e}")
            return None
        return MultipersonaGroupBuilder(
            env_thunk=partial(
                MultipersonaGsm8kEnv, problem, gold, self.tokenizer, tag_coef=self.tag_coef
            ),
            num_envs=group_size,
        )


@chz.chz
class MultipersonaGsm8kDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    group_size: int
    seed: int = 0
    n_train: int | None = None
    n_test: int | None = None
    tag_coef: float = 0.2

    async def __call__(self) -> tuple[MultipersonaGsm8kDataset, MultipersonaGsm8kDataset]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        train = MultipersonaGsm8kDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            tokenizer=tokenizer,
            split="train",
            seed=self.seed,
            n_examples=self.n_train,
            tag_coef=self.tag_coef,
        )
        test = MultipersonaGsm8kDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            tokenizer=tokenizer,
            split="test",
            seed=self.seed,
            n_examples=self.n_test,
            tag_coef=self.tag_coef,
        )
        return train, test
