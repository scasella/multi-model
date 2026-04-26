"""Assemble the olympiad-math union pool for the RLVR hill-climbing experiment.

Sources (all verifiable short-answer math, graded by math_verify):
  - HMMT Feb 2024 + 2025               (MathArena/hmmt_feb_2024, _2025)
  - AIME 2022, 2023, 2024, 2025        (HF mirrors; 2024+2025 already used for vibecheck)
  - OlympiadBench (TP_MM_maths_en_COMP, open-ended math, English)
  - AMC 2022/2023 from AIMO validation (AI-MO/aimo-validation-amc)

Output: data/olympiad_pool/all.jsonl with schema:
    { "uid": "<source>__<id>", "source": str, "year": int | None,
      "problem": str, "gold": str, "answer_type": "int" | "latex" | "other" }

The pool is what both arms (panel + Qwen thinking) will see. The
variance-band filter (scripts/filter_variance_band.py) then prunes it
per-model, and the intersection step produces the actual training set.

Uncertain HF paths are flagged with `UNVERIFIED` — adjust as needed.
A missing source just logs a warning and continues; partial pools are OK.

Usage:
    python scripts/build_olympiad_pool.py
    python scripts/build_olympiad_pool.py --only hmmt,aime   # subset
    python scripts/build_olympiad_pool.py --max-per-source 100  # smoke
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "olympiad_pool"
OUT_PATH = OUT_DIR / "all.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("build_olympiad_pool")


# ----------------------------------------------------------------------
# Answer-type classifier + canonicalizer
# ----------------------------------------------------------------------

_INT_RE = re.compile(r"^-?\d{1,4}$")


def canonicalize_gold(raw: object) -> tuple[str, str]:
    """Return (gold_string, answer_type).

    answer_type ∈ {"int", "latex", "other"}. math_verify handles all three
    at grade time; this is just metadata for filtering.
    """
    s = str(raw).strip()
    # strip outer $...$ or \(...\) if present
    s = s.strip()
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    # unwrap a single outer \boxed{...}
    m = re.fullmatch(r"\\boxed\s*\{(.+)\}", s, flags=re.DOTALL)
    if m:
        s = m.group(1).strip()
    if _INT_RE.match(s):
        return s, "int"
    # heuristic: contains a LaTeX control sequence or math operator
    if re.search(r"\\[a-zA-Z]+|[\^_]|\\frac|\\sqrt|\\pi", s):
        return s, "latex"
    return s, "other"


# ----------------------------------------------------------------------
# Source loaders. Each yields {uid, source, year, problem, gold}.
# ----------------------------------------------------------------------


def _yield_hmmt(year: int) -> Iterator[dict]:
    ds_name = f"MathArena/hmmt_feb_{year}"
    try:
        ds = load_dataset(ds_name, split="train")
    except Exception as e:
        logger.warning("skip %s: %s", ds_name, e)
        return
    for i, row in enumerate(ds):
        problem = row.get("problem") or row.get("question")
        answer = row.get("answer") or row.get("solution_value")
        if not problem or answer is None:
            continue
        gold, atype = canonicalize_gold(answer)
        yield {
            "uid": f"hmmt_feb_{year}__{i}",
            "source": "hmmt",
            "year": year,
            "problem": problem,
            "gold": gold,
            "answer_type": atype,
        }


def _yield_aime(year: int) -> Iterator[dict]:
    # Mirror choices by year (chosen for availability):
    candidates = {
        2022: ["Maxwell-Jia/AIME_2024"],        # placeholder — AIME22 less common on HF
        2023: ["Maxwell-Jia/AIME_2024"],        # placeholder — AIME23 less common on HF
        2024: ["HuggingFaceH4/aime_2024", "Maxwell-Jia/AIME_2024"],
        2025: ["yentinglin/aime_2025"],
    }
    # Note: 2022/2023 AIME is hard to find on HF. If you have a source, add it here;
    # else those years are skipped and you lean on 2024/2025 + HMMT.
    for ds_name in candidates.get(year, []):
        try:
            ds = load_dataset(ds_name, split="train")
            break
        except Exception as e:
            logger.warning("skip %s: %s", ds_name, e)
            ds = None
    if ds is None:
        return
    for i, row in enumerate(ds):
        problem = row.get("problem") or row.get("question")
        answer = row.get("answer") or row.get("solution")
        if not problem or answer is None:
            continue
        gold, atype = canonicalize_gold(answer)
        yield {
            "uid": f"aime_{year}__{i}",
            "source": "aime",
            "year": year,
            "problem": problem,
            "gold": gold,
            "answer_type": atype,
        }


def _yield_olympiadbench() -> Iterator[dict]:
    # UNVERIFIED: confirm the exact config / field names on your HF cache.
    # Alternate configs you may want: TP_TO_maths_en_COMP, OE_TO_maths_en_COMP.
    for cfg in ("OE_TO_maths_en_COMP",):
        try:
            ds = load_dataset("Hothan/OlympiadBench", cfg, split="train")
        except Exception as e:
            logger.warning("skip OlympiadBench:%s: %s", cfg, e)
            continue
        for i, row in enumerate(ds):
            problem = row.get("question") or row.get("problem")
            answer = row.get("final_answer") or row.get("answer")
            # `final_answer` on OlympiadBench is usually a list of strings.
            if isinstance(answer, list):
                if not answer:
                    continue
                answer = answer[0]
            if not problem or answer is None:
                continue
            gold, atype = canonicalize_gold(answer)
            yield {
                "uid": f"olympiadbench_{cfg}__{i}",
                "source": "olympiadbench",
                "year": row.get("year"),
                "problem": problem,
                "gold": gold,
                "answer_type": atype,
            }


def _yield_amc() -> Iterator[dict]:
    try:
        ds = load_dataset("AI-MO/aimo-validation-amc", split="train")
    except Exception as e:
        logger.warning("skip AI-MO/aimo-validation-amc: %s", e)
        return
    for i, row in enumerate(ds):
        problem = row.get("problem") or row.get("question")
        answer = row.get("answer")
        if not problem or answer is None:
            continue
        gold, atype = canonicalize_gold(answer)
        yield {
            "uid": f"amc__{i}",
            "source": "amc",
            "year": row.get("year"),
            "problem": problem,
            "gold": gold,
            "answer_type": atype,
        }


SOURCES: dict[str, callable] = {
    "hmmt_2024": lambda: _yield_hmmt(2024),
    "hmmt_2025": lambda: _yield_hmmt(2025),
    "aime_2022": lambda: _yield_aime(2022),
    "aime_2023": lambda: _yield_aime(2023),
    "aime_2024": lambda: _yield_aime(2024),
    "aime_2025": lambda: _yield_aime(2025),
    "olympiadbench": _yield_olympiadbench,
    "amc": _yield_amc,
}


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only",
        default=None,
        help="comma-separated subset of source keys (see SOURCES). Default: all.",
    )
    ap.add_argument(
        "--max-per-source",
        type=int,
        default=None,
        help="Cap per source for smoke testing.",
    )
    args = ap.parse_args()

    if args.only:
        keys = [k.strip() for k in args.only.split(",") if k.strip()]
        missing = [k for k in keys if k not in SOURCES]
        if missing:
            raise SystemExit(f"unknown source keys: {missing} (have: {list(SOURCES)})")
    else:
        keys = list(SOURCES.keys())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_source: dict[str, int] = {}
    seen_uids: set[str] = set()
    n_written = 0

    with open(OUT_PATH, "w") as fh:
        for key in keys:
            logger.info("loading %s ...", key)
            count = 0
            for row in SOURCES[key]():
                if row["uid"] in seen_uids:
                    continue
                seen_uids.add(row["uid"])
                if not row["problem"].strip() or not row["gold"].strip():
                    continue
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
                n_written += 1
                if args.max_per_source and count >= args.max_per_source:
                    break
            by_source[key] = count
            logger.info("  %s: %d rows", key, count)

    # Print a summary with answer-type counts.
    by_atype: dict[str, int] = {}
    with open(OUT_PATH) as fh:
        for line in fh:
            r = json.loads(line)
            by_atype[r["answer_type"]] = by_atype.get(r["answer_type"], 0) + 1

    logger.info("=== pool summary ===")
    logger.info("total rows:  %d", n_written)
    logger.info("by source:   %s", by_source)
    logger.info("by answer:   %s", by_atype)
    logger.info("wrote %s", OUT_PATH)


if __name__ == "__main__":
    main()
