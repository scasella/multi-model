"""Shared utilities for olympiad-pool RL envs.

Both `multipersona_olympiad.py` (panel) and `thinking_native_olympiad.py`
(Qwen3 native thinking) read the same JSONL files:

    data/olympiad_pool/train.jsonl        — RL training problems
    data/olympiad_pool/heldout_eval.jsonl — frozen in-training eval

produced by `scripts/intersect_variance_bands.py`.

Each row has:
    { "uid", "source", "year", "problem", "gold", "answer_type",
      "panel_band", "thinking_band" }

Grader: `math_verify` for LaTeX/integer equivalence. Same grader both arms.
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from math_verify import parse as mv_parse, verify as mv_verify

from envs.multipersona_math import extract_last_boxed

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN_PATH = ROOT / "data" / "olympiad_pool" / "train.jsonl"
DEFAULT_HELDOUT_PATH = ROOT / "data" / "olympiad_pool" / "heldout_eval.jsonl"


def load_pool_rows(path: Path | str) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"olympiad pool split not found at {path}. "
            "Did you run build_olympiad_pool + filter_variance_band + intersect_variance_bands?"
        )
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("problem") and r.get("gold"):
                rows.append(r)
    logger.info("loaded %d rows from %s", len(rows), path)
    return rows


def group_by_source(rows: list[dict]) -> dict[str, list[dict]]:
    by_src: dict[str, list[dict]] = {}
    for r in rows:
        by_src.setdefault(r["source"], []).append(r)
    return by_src


def grade_math_verify(answer_body: str | None, completion_fallback: str, gold: str) -> bool:
    """Return True iff any candidate parses to gold under math_verify.

    Candidates, in order:
      1. answer_body (if not None)
      2. last \\boxed{} inside answer_body
      3. last \\boxed{} inside the full completion
      4. the full completion (raw)
    Gold is wrapped as "$gold$".
    """
    try:
        gold_parsed = mv_parse("$" + gold + "$")
    except Exception:
        return False

    candidates: list[str] = []
    if answer_body:
        candidates.append(answer_body)
        inner = extract_last_boxed(answer_body)
        if inner:
            candidates.append(inner)
    inner_full = extract_last_boxed(completion_fallback)
    if inner_full:
        candidates.append(inner_full)

    for cand in candidates:
        for wrapped in (f"${cand}$", cand):
            try:
                pred = mv_parse(wrapped)
                if mv_verify(gold_parsed, pred):
                    return True
            except Exception:
                continue

    # last resort: raw completion
    try:
        pred = mv_parse(completion_fallback)
        if mv_verify(gold_parsed, pred):
            return True
    except Exception:
        pass
    return False


class RoundRobinSourceSampler:
    """Round-robin over sources with per-source shuffled deques.

    Mirrors the category-balanced sampler in envs.multipersona_math, but
    keyed on `source` (hmmt / aime / olympiadbench / amc) instead of
    MATH subject. Guarantees every source contributes evenly across
    training batches regardless of pool size asymmetry.
    """

    def __init__(self, rows: list[dict], seed: int):
        self._rng = random.Random(seed)
        self.by_source: dict[str, list[dict]] = group_by_source(rows)
        for src in self.by_source:
            self._rng.shuffle(self.by_source[src])
        self.sources: list[str] = sorted(self.by_source)
        self._ptrs: dict[str, int] = {s: 0 for s in self.sources}
        self._rot = 0
        logger.info(
            "RoundRobinSourceSampler: %d sources, pool sizes %s",
            len(self.sources),
            {s: len(v) for s, v in self.by_source.items()},
        )

    def next_row(self) -> dict:
        src = self.sources[self._rot % len(self.sources)]
        self._rot += 1
        rows = self.by_source[src]
        if self._ptrs[src] >= len(rows):
            self._rng.shuffle(rows)
            self._ptrs[src] = 0
        row = rows[self._ptrs[src]]
        self._ptrs[src] += 1
        return row
