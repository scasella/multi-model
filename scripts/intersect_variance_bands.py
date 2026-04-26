"""Intersect the variance bands of two arms and emit train + heldout splits.

Given per-problem band classifications for the panel arm and the thinking
arm (produced by `filter_variance_band.py`), this computes:

  TRAIN_POOL      = { uid : panel.band == variance_band AND thinking.band == variance_band }
  HELDOUT_POOL    = everything else that's *potentially* useful:
      - problems where at least one arm is in the variance band (asymmetric trainable)
      - problems where both arms are all_one  (ceiling — good for sanity)
      - problems where both arms are all_zero (floor — good for documenting failure)

We stratify HELDOUT by source and by joint-band to get a ~100-problem
frozen eval set. The remainder goes to an auxiliary held-out pool for
post-hoc analysis.

Outputs (data/olympiad_pool/):
    train.jsonl         — joint variance band; used for RL training
    heldout_eval.jsonl  — frozen, used for step-wise pass@1/pass@16
    heldout_aux.jsonl   — remainder; post-hoc analysis only
    split_summary.json  — counts by source × joint_band

Usage:
    python scripts/intersect_variance_bands.py \\
        --panel reports/variance_band/panel_g8/per_problem.jsonl \\
        --thinking reports/variance_band/thinking_g8/per_problem.jsonl \\
        --pool data/olympiad_pool/all.jsonl \\
        [--heldout-size 100] [--seed 20260424]
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "olympiad_pool"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("intersect_variance_bands")


def load_bands(path: Path) -> dict[str, str]:
    m: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            m[r["uid"]] = r["band"]
    return m


def load_pool(path: Path) -> dict[str, dict]:
    m: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            m[r["uid"]] = r
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True, type=Path)
    ap.add_argument("--thinking", required=True, type=Path)
    ap.add_argument("--pool", default=ROOT / "data/olympiad_pool/all.jsonl", type=Path)
    ap.add_argument("--heldout-size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260424)
    args = ap.parse_args()

    panel_bands = load_bands(args.panel)
    think_bands = load_bands(args.thinking)
    pool = load_pool(args.pool)

    logger.info("panel:    %d problems scored", len(panel_bands))
    logger.info("thinking: %d problems scored", len(think_bands))
    logger.info("pool:     %d problems total", len(pool))

    shared_uids = sorted(set(panel_bands) & set(think_bands) & set(pool))
    logger.info("shared (all three): %d", len(shared_uids))

    joint_buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    for uid in shared_uids:
        key = (panel_bands[uid], think_bands[uid])
        joint_buckets[key].append(uid)

    logger.info("=== joint-band buckets ===")
    for (p, t), uids in sorted(joint_buckets.items()):
        logger.info("  panel=%-13s  thinking=%-13s  %5d problems", p, t, len(uids))

    # TRAIN POOL: joint variance band.
    train_uids = list(joint_buckets.get(("variance_band", "variance_band"), []))
    logger.info("joint variance band (TRAIN): %d", len(train_uids))

    # Build heldout eval: stratify across interesting joint-bands.
    rng = random.Random(args.seed)
    heldout_eval_uids: list[str] = []
    used: set[str] = set()

    # Priorities: problems where BOTH arms are in variance band are most
    # informative for eval — we can see per-problem differences. Then
    # asymmetric variance-band problems. Then mixed (one is ceiling, one is variance).
    strata: list[tuple[str, str]] = [
        ("variance_band", "variance_band"),
        ("variance_band", "all_zero"),
        ("all_zero", "variance_band"),
        ("variance_band", "all_one"),
        ("all_one", "variance_band"),
        ("all_zero", "all_zero"),
        ("all_one", "all_one"),
    ]
    # We want to reserve a chunk of the (variance, variance) bucket for
    # training, not eval — take at most 25% of it for eval.
    max_eval_from_joint_vb = max(0, len(train_uids) // 4)
    stratum_caps = {
        ("variance_band", "variance_band"): max_eval_from_joint_vb,
    }

    for stratum in strata:
        cap = stratum_caps.get(stratum, args.heldout_size)
        pool_for_stratum = [u for u in joint_buckets.get(stratum, []) if u not in used]
        rng.shuffle(pool_for_stratum)
        take = pool_for_stratum[: min(cap, args.heldout_size - len(heldout_eval_uids))]
        heldout_eval_uids.extend(take)
        used.update(take)
        if len(heldout_eval_uids) >= args.heldout_size:
            break

    # Remove any heldout-eval uids from the training pool.
    train_uids = [u for u in train_uids if u not in used]
    heldout_aux_uids = [u for u in shared_uids if u not in used and u not in set(train_uids)]

    logger.info("TRAIN (joint variance band, post-heldout): %d", len(train_uids))
    logger.info("HELDOUT eval:                              %d", len(heldout_eval_uids))
    logger.info("HELDOUT aux (remainder):                   %d", len(heldout_aux_uids))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def write(uids: list[str], path: Path) -> None:
        with open(path, "w") as f:
            for uid in uids:
                r = dict(pool[uid])
                r["panel_band"] = panel_bands[uid]
                r["thinking_band"] = think_bands[uid]
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write(train_uids, OUT_DIR / "train.jsonl")
    write(heldout_eval_uids, OUT_DIR / "heldout_eval.jsonl")
    write(heldout_aux_uids, OUT_DIR / "heldout_aux.jsonl")

    by_source_train: dict[str, int] = defaultdict(int)
    by_source_heldout: dict[str, int] = defaultdict(int)
    for uid in train_uids:
        by_source_train[pool[uid]["source"]] += 1
    for uid in heldout_eval_uids:
        by_source_heldout[pool[uid]["source"]] += 1

    summary = {
        "panel_file": str(args.panel),
        "thinking_file": str(args.thinking),
        "pool_file": str(args.pool),
        "n_shared": len(shared_uids),
        "joint_bands": {
            f"{p}__{t}": len(uids) for (p, t), uids in joint_buckets.items()
        },
        "train": {
            "n": len(train_uids),
            "by_source": dict(by_source_train),
        },
        "heldout_eval": {
            "n": len(heldout_eval_uids),
            "by_source": dict(by_source_heldout),
        },
        "heldout_aux": {"n": len(heldout_aux_uids)},
        "seed": args.seed,
    }
    (OUT_DIR / "split_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("wrote train/heldout to %s", OUT_DIR)
    logger.info("%s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
