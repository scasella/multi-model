"""Track B analysis: does panel's divergent exploration catch problems thinking fails?

Inputs: two MATH500-L5 full-sweep result dirs:
  - thinking_l5_full (Qwen3-30B-A3B hybrid, enable_thinking=True, n=4)
  - panel_l5_full     (our post-MATH panel adapter, n=8)

Pipeline:
  1. Join on unique_id (all 134 L5 MATH500 problems).
  2. Classify each problem by thinking-difficulty: hard_fail (0/4),
     hard_rare (1/4), hard_coin (2/4), easy_maj (3/4), easy (4/4).
  3. For each difficulty bucket, compute panel's hit rate and pass@n.
  4. Specifically answer: for problems where thinking's pass@4 is False,
     is panel's pass@8 better than chance?
  5. Identify "panel-saves" — problems where thinking completely failed
     (0/4) but panel solved at least once (≥1/8).
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
THINK_DIR = ROOT / "reports/eval_math500_vibecheck/thinking_l5_full"
PANEL_DIR = ROOT / "reports/eval_math500_vibecheck/panel_l5_full"
OUT_DIR = ROOT / "reports/hard_problem_analysis"


def load_per_problem(summary_path: Path) -> dict[str, dict]:
    s = json.load(open(summary_path))
    return {p["unique_id"]: p for p in s["per_problem"]}


def bucket(n_correct: int, n: int) -> str:
    if n_correct == 0:
        return "hard_fail"     # 0% — thinking always fails
    if n_correct == 1 and n == 4:
        return "hard_rare"     # 25% — thinking rarely catches
    if n_correct == 2 and n == 4:
        return "hard_coin"     # 50% — coin-flip
    if n_correct == 3 and n == 4:
        return "easy_maj"      # 75% — thinking usually catches
    if n_correct == n:
        return "easy"          # 100% — always solves
    return "other"


def main() -> None:
    think = load_per_problem(THINK_DIR / "summary.json")
    panel = load_per_problem(PANEL_DIR / "summary.json")
    uids = sorted(set(think) & set(panel))
    print(f"thinking problems: {len(think)}, panel problems: {len(panel)}, shared: {len(uids)}")

    # Classify each problem by thinking difficulty
    joined = []
    for uid in uids:
        t = think[uid]
        p = panel[uid]
        joined.append({
            "uid": uid,
            "subject": t["subject"],
            "level": t["level"],
            "gold": t["gold"],
            "think_n": t["n_samples"],
            "think_correct": t["n_correct"],
            "think_pass_n": t["pass_at_n"],
            "panel_n": p["n_samples"],
            "panel_correct": p["n_correct"],
            "panel_pass_n": p["pass_at_n"],
            "bucket": bucket(t["n_correct"], t["n_samples"]),
        })

    # === Bucket-level aggregation ===
    by_bucket: dict[str, list] = defaultdict(list)
    for r in joined:
        by_bucket[r["bucket"]].append(r)

    print("\n=== Panel performance by thinking-difficulty bucket ===\n")
    hdr = ("bucket", "n_probs", "think_hit", "panel_hit", "panel_pass@8", "panel_saves")
    print("  " + "  ".join(f"{h:>14s}" for h in hdr))
    bucket_order = ["hard_fail", "hard_rare", "hard_coin", "easy_maj", "easy"]
    for b in bucket_order:
        recs = by_bucket.get(b, [])
        if not recs:
            continue
        n = len(recs)
        think_hit = sum(r["think_correct"] / r["think_n"] for r in recs) / n
        panel_hit = sum(r["panel_correct"] / r["panel_n"] for r in recs) / n
        panel_pn = sum(r["panel_pass_n"] for r in recs) / n
        # "panel_saves": problems where panel pass@8 = True (AT LEAST ONE correct)
        saves = sum(r["panel_pass_n"] for r in recs)
        print(f"  {b:>14s}  {n:>14d}  {think_hit:>14.3f}  {panel_hit:>14.3f}  {panel_pn:>14.3f}  {saves:>14d}")

    # === Focus on thinking-failures ===
    think_fail = [r for r in joined if not r["think_pass_n"]]
    print(f"\n=== Thinking-failure problems (pass@4 = False): {len(think_fail)} ===\n")
    print("  " + "  ".join(f"{h:>10s}" for h in ("subject","panel_hits","panel_pass@8","think_hits")))
    for r in sorted(think_fail, key=lambda x: (-x["panel_correct"], x["uid"])):
        print(f"  {r['subject'][:18]:>18s}  {r['panel_correct']}/{r['panel_n']}        "
              f"{str(r['panel_pass_n']):>10s}  {r['think_correct']}/{r['think_n']}  {r['uid']}")

    panel_saves = [r for r in think_fail if r["panel_pass_n"]]
    panel_also_fails = [r for r in think_fail if not r["panel_pass_n"]]
    print(f"\n  Panel saved {len(panel_saves)} / {len(think_fail)} thinking-failures")
    if think_fail:
        rate = len(panel_saves) / len(think_fail)
        # Wilson CI for save rate
        import math
        z = 1.96
        n = len(think_fail)
        p = rate
        denom = 1 + z*z/n
        center = (p + z*z/(2*n)) / denom
        half = z * math.sqrt(max(p*(1-p)/n + z*z/(4*n*n), 0)) / denom
        print(f"  Save rate: {rate:.3f}  Wilson95=[{center-half:.3f}, {center+half:.3f}]")

    print(f"\n  Panel-also-fails ({len(panel_also_fails)}): problems NEITHER model solved at n=8/4")
    for r in panel_also_fails:
        print(f"    {r['uid']}  ({r['subject']})  think=0/4  panel=0/8  gold={r['gold']!r}")

    # === Inverse direction: panel-fails that thinking solved ===
    panel_fail = [r for r in joined if not r["panel_pass_n"]]
    print(f"\n=== Panel-failure problems (pass@8 = False): {len(panel_fail)} ===")
    print(f"  Of these, thinking pass@4 = True on {sum(r['think_pass_n'] for r in panel_fail)}/{len(panel_fail)} "
          f"(problems thinking solves but panel can't, even with 2× the sample budget)")

    # === Correlation / joint failures ===
    both_fail = [r for r in joined if not r["panel_pass_n"] and not r["think_pass_n"]]
    print(f"\n=== Joint-failure set (both fail at their respective n): {len(both_fail)} problems ===")

    # === Write outputs ===
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "joined.jsonl").write_text(
        "\n".join(json.dumps(r) for r in joined) + "\n"
    )
    summary = {
        "n_problems": len(joined),
        "think_pass_at_4_count": sum(1 for r in joined if r["think_pass_n"]),
        "panel_pass_at_8_count": sum(1 for r in joined if r["panel_pass_n"]),
        "bucket_counts": {b: len(v) for b, v in by_bucket.items()},
        "panel_saves": {
            "n_thinking_failures": len(think_fail),
            "n_panel_saves": len(panel_saves),
            "n_joint_failures": len(panel_also_fails),
            "save_rate": len(panel_saves) / len(think_fail) if think_fail else None,
            "saved_uids": [r["uid"] for r in panel_saves],
            "joint_failure_uids": [r["uid"] for r in panel_also_fails],
        },
        "panel_fail_thinking_succeed": {
            "n": sum(1 for r in panel_fail if r["think_pass_n"]),
            "uids": [r["uid"] for r in panel_fail if r["think_pass_n"]],
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT_DIR/'summary.json'}")


if __name__ == "__main__":
    main()
