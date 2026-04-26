"""Build per-arm train splits + shared held-out from variance-band outputs.

Given per-problem band classifications for both arms, writes:

    data/olympiad_pool/
        panel_train.jsonl       — panel VB, minus held-out UIDs
        thinking_train.jsonl    — thinking VB, minus held-out UIDs
        heldout_eval.jsonl      — shared, stratified-by-source sample of the
                                  full pool (default 100 problems). Both arms
                                  evaluate on this set identically.
        heldout_aux.jsonl       — remainder (not in either train, not in
                                  heldout) for post-hoc analysis.
        split_summary.json      — counts, joint-band stats, per-source

Design note:
    On Qwen3-30B-A3B the per-arm variance bands are largely disjoint:
    panel concentrates in OlympiadBench/AMC; thinking in HMMT/AIME. Rather
    than force both arms onto the shrinking joint intersection, each arm
    trains on its OWN variance band and both arms are scored on the same
    stratified held-out eval. This tests hill-climbing efficiency per arm
    with gradient signal guaranteed at step 0, without coupling the two
    training pools. See `intersect_variance_bands.py` for the prior
    joint-band design.

Usage:
    python scripts/build_per_arm_splits.py \\
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
logger = logging.getLogger("build_per_arm_splits")


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


def stratified_sample_by_source(
    pool: dict[str, dict],
    allowed_uids: set[str],
    target: int,
    rng: random.Random,
) -> list[str]:
    """Draw ~target UIDs from `allowed_uids`, stratified uniformly by source.

    If a source has fewer problems than its quota, we take all of them and
    spill the deficit to the remaining sources.
    """
    by_source: dict[str, list[str]] = defaultdict(list)
    for uid in allowed_uids:
        by_source[pool[uid]["source"]].append(uid)

    sources = sorted(by_source.keys())
    per_source_quota = {s: target // len(sources) for s in sources}
    # distribute remainder deterministically
    leftover = target - sum(per_source_quota.values())
    for s in sources[:leftover]:
        per_source_quota[s] += 1

    picked: list[str] = []
    spill = 0
    for s in sources:
        rng.shuffle(by_source[s])
        q = per_source_quota[s]
        take = by_source[s][:q]
        picked.extend(take)
        if len(by_source[s]) < q:
            spill += q - len(by_source[s])
        by_source[s] = by_source[s][q:]

    # spill any deficit across the remaining per-source tails (largest first)
    if spill:
        tails = sorted(sources, key=lambda s: -len(by_source[s]))
        for s in tails:
            if spill <= 0:
                break
            take = by_source[s][:spill]
            picked.extend(take)
            by_source[s] = by_source[s][len(take):]
            spill -= len(take)

    return picked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True, type=Path,
                    help="per_problem.jsonl from filter_variance_band.py (panel arm)")
    ap.add_argument("--thinking", required=True, type=Path,
                    help="per_problem.jsonl from filter_variance_band.py (thinking arm)")
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
    logger.info("shared (panel ∩ thinking ∩ pool): %d", len(shared_uids))

    # Diagnostic: joint-band contingency table (this is why per-arm splits exist).
    joint_counts: dict[tuple[str, str], int] = defaultdict(int)
    for uid in shared_uids:
        joint_counts[(panel_bands[uid], think_bands[uid])] += 1
    logger.info("=== joint-band contingency (diagnostic) ===")
    for (p, t), n in sorted(joint_counts.items()):
        logger.info("  panel=%-13s  thinking=%-13s  %5d", p, t, n)

    panel_vb = {u for u in shared_uids if panel_bands[u] == "variance_band"}
    think_vb = {u for u in shared_uids if think_bands[u] == "variance_band"}
    logger.info("panel VB:    %d problems", len(panel_vb))
    logger.info("thinking VB: %d problems", len(think_vb))
    logger.info("VB intersection (would have been joint train): %d",
                len(panel_vb & think_vb))
    logger.info("VB union:                                      %d",
                len(panel_vb | think_vb))

    # Shared held-out: stratified random sample from the full shared pool.
    # Does NOT depend on either arm's band — both arms will be scored on it.
    rng = random.Random(args.seed)
    heldout_eval_uids = stratified_sample_by_source(
        pool={u: pool[u] for u in shared_uids},
        allowed_uids=set(shared_uids),
        target=args.heldout_size,
        rng=rng,
    )
    heldout_set = set(heldout_eval_uids)
    logger.info("HELDOUT eval: %d (stratified by source)", len(heldout_eval_uids))

    # Per-arm train sets: each arm's VB minus any held-out UIDs.
    panel_train_uids = sorted(panel_vb - heldout_set)
    think_train_uids = sorted(think_vb - heldout_set)
    logger.info("PANEL train (VB \\ heldout):    %d", len(panel_train_uids))
    logger.info("THINKING train (VB \\ heldout): %d", len(think_train_uids))

    # Auxiliary held-out = everything not in any train set and not in heldout_eval.
    used = set(panel_train_uids) | set(think_train_uids) | heldout_set
    heldout_aux_uids = sorted(set(shared_uids) - used)
    logger.info("HELDOUT aux (remainder): %d", len(heldout_aux_uids))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def write(uids: list[str], path: Path) -> None:
        with open(path, "w") as f:
            for uid in uids:
                r = dict(pool[uid])
                r["panel_band"] = panel_bands[uid]
                r["thinking_band"] = think_bands[uid]
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write(panel_train_uids, OUT_DIR / "panel_train.jsonl")
    write(think_train_uids, OUT_DIR / "thinking_train.jsonl")
    write(heldout_eval_uids, OUT_DIR / "heldout_eval.jsonl")
    write(heldout_aux_uids, OUT_DIR / "heldout_aux.jsonl")

    def source_counts(uids: list[str]) -> dict[str, int]:
        d: dict[str, int] = defaultdict(int)
        for uid in uids:
            d[pool[uid]["source"]] += 1
        return dict(d)

    # Record paths as repo-relative when possible so split_summary.json is
    # portable across machines (no /Users/... leaks into the public artifact).
    def _rel(p: Path) -> str:
        try:
            return str(Path(p).resolve().relative_to(ROOT))
        except ValueError:
            return str(p)

    summary = {
        "panel_file": _rel(args.panel),
        "thinking_file": _rel(args.thinking),
        "pool_file": _rel(args.pool),
        "seed": args.seed,
        "n_shared": len(shared_uids),
        "joint_bands": {f"{p}__{t}": n for (p, t), n in joint_counts.items()},
        "panel_vb_size": len(panel_vb),
        "thinking_vb_size": len(think_vb),
        "vb_intersection_size": len(panel_vb & think_vb),
        "vb_union_size": len(panel_vb | think_vb),
        "panel_train": {
            "n": len(panel_train_uids),
            "by_source": source_counts(panel_train_uids),
        },
        "thinking_train": {
            "n": len(think_train_uids),
            "by_source": source_counts(think_train_uids),
        },
        "heldout_eval": {
            "n": len(heldout_eval_uids),
            "by_source": source_counts(heldout_eval_uids),
        },
        "heldout_aux": {
            "n": len(heldout_aux_uids),
            "by_source": source_counts(heldout_aux_uids),
        },
    }
    (OUT_DIR / "split_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("wrote splits to %s", OUT_DIR)
    logger.info("%s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
