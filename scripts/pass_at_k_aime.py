"""Pass@k curves on AIME 2024+2025: panel (post-MATH) vs Qwen3-30B-A3B thinking.

Companion to scripts/pass_at_k_crossover.py (MATH500 version).

Uses the unbiased pass@k estimator:
    pass@k = 1 - C(n-c, k) / C(n, k)

Inputs (both at n=16):
  - reports/eval_aime_vibecheck/aime_panel_postmath_n16/rollouts.jsonl
  - reports/eval_aime_vibecheck/aime_thinking_n16/rollouts.jsonl

Reports:
  - pass@k curve for k=1..16, both variants
  - paired delta (panel - thinking) per problem at each k + t-stat
  - gap-closure rate (how much does the gap narrow from k=1 to k=16?)
  - problem-level overlap (which 20 problems: both/thinking-only/panel-only/neither)
  - comparison to MATH500 gap-closure to test whether diversity effect
    transfers to the non-saturated distribution
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "reports/eval_aime_vibecheck/aime_panel_postmath_n16/rollouts.jsonl"
THINK = ROOT / "reports/eval_aime_vibecheck/aime_thinking_n16/rollouts.jsonl"
OUT_DIR = ROOT / "reports/pass_at_k_aime"


def load(path: Path) -> dict[str, tuple[int, int]]:
    """uid (year_idx) -> (n_samples, n_correct)."""
    by_uid: dict[str, list[int]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            uid = f"{r['year']}_{r['problem_idx']}"
            by_uid[uid].append(int(r["correct"]))
    return {u: (len(v), sum(v)) for u, v in by_uid.items()}


def pass_at_k(n: int, c: int, k: int) -> float:
    if k > n:
        raise ValueError(f"k={k} > n={n}")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def curves_and_deltas(panel, think, ks):
    shared = sorted(set(panel) & set(think))
    rows = []
    for k in ks:
        p_vals = [pass_at_k(*panel[u], k) for u in shared]
        t_vals = [pass_at_k(*think[u], k) for u in shared]
        deltas = [p - t for p, t in zip(p_vals, t_vals, strict=True)]
        n = len(deltas)
        mean = sum(deltas) / n
        if n > 1:
            var = sum((d - mean) ** 2 for d in deltas) / (n - 1)
            se = math.sqrt(var / n)
            t = mean / se if se > 0 else (float("inf") if mean > 0 else (float("-inf") if mean < 0 else 0.0))
        else:
            se, t = float("nan"), float("nan")
        rows.append({
            "k": k,
            "n_paired": n,
            "panel_pass_at_k": sum(p_vals) / n,
            "thinking_pass_at_k": sum(t_vals) / n,
            "delta_mean": mean,
            "delta_se": se,
            "t": t,
        })
    return rows, shared


def main() -> None:
    panel = load(PANEL)
    think = load(THINK)
    print(f"panel:    {len(panel)} problems, n/problem = {next(iter(panel.values()))[0]}")
    print(f"thinking: {len(think)} problems, n/problem = {next(iter(think.values()))[0]}")
    ks = list(range(1, 17))
    rows, shared = curves_and_deltas(panel, think, ks)

    print(f"\n=== Matched pass@k (both n=16) on AIME 2024+2025, {len(shared)} shared problems ===\n")
    print(f"{'k':>3} {'panel':>8} {'thinking':>10} {'Δ':>9} {'SE':>7} {'t':>7} {'winner':>9}")
    for r in rows:
        winner = "panel" if r["delta_mean"] > 0.005 else ("thinking" if r["delta_mean"] < -0.005 else "~tie")
        print(f"{r['k']:>3} {r['panel_pass_at_k']:>8.4f} {r['thinking_pass_at_k']:>10.4f} "
              f"{r['delta_mean']:>+9.4f} {r['delta_se']:>7.4f} {r['t']:>+7.2f} {winner:>9}")

    k1_gap = rows[0]["delta_mean"]
    k16_gap = rows[-1]["delta_mean"]
    closure = k16_gap - k1_gap  # positive = gap closed (panel caught up)
    closure_pp = closure * 100
    print(f"\nGap closure k=1 → k=16: {closure_pp:+.1f}pp (positive = panel closing gap)")

    crossover = next((r["k"] for r in rows if r["delta_mean"] >= 0), None)
    print(f"Crossover k (panel ≥ thinking): {crossover}")

    # Problem-level overlap
    p_solved = {u for u, (n, c) in panel.items() if c > 0}
    t_solved = {u for u, (n, c) in think.items() if c > 0}
    both = p_solved & t_solved
    panel_only = p_solved - t_solved
    think_only = t_solved - p_solved
    neither = set(shared) - (p_solved | t_solved)
    print("\n=== Problem-level ensemble (n=20) ===")
    print(f"  both solve:         {len(both)}")
    print(f"  thinking-only:      {len(think_only)}")
    print(f"  panel-only:         {len(panel_only)}")
    print(f"  neither:            {len(neither)}")
    print(f"  panel alone:        {len(p_solved)}/20 = {len(p_solved)/20:.3f}")
    print(f"  thinking alone:     {len(t_solved)}/20 = {len(t_solved)/20:.3f}")
    print(f"  union (ensemble):   {len(p_solved | t_solved)}/20 = {len(p_solved | t_solved)/20:.3f}")
    if panel_only:
        print(f"  panel-only UIDs:    {sorted(panel_only)}")
    if neither:
        print(f"  joint-failure UIDs: {sorted(neither)}")

    # Compare gap-closure rate on MATH500 vs AIME
    print("\n=== Gap-closure comparison ===")
    print("  MATH500 L5-full (matched k=1..4): -32.5pp → -17.8pp  (closure +14.7pp over 4x sampling)")
    print(f"  AIME 16-matched  (matched k=1..16): {rows[0]['delta_mean']*100:+.1f}pp → "
          f"{rows[-1]['delta_mean']*100:+.1f}pp  (closure {closure_pp:+.1f}pp over 16x sampling)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(json.dumps({
        "n_panel_problems": len(panel),
        "n_thinking_problems": len(think),
        "n_shared": len(shared),
        "rows": rows,
        "gap_closure_k1_to_k16_pp": closure_pp,
        "crossover_k": crossover,
        "problem_level_overlap": {
            "both": len(both),
            "thinking_only": len(think_only),
            "panel_only": len(panel_only),
            "neither": len(neither),
            "panel_alone_pass": len(p_solved) / 20,
            "thinking_alone_pass": len(t_solved) / 20,
            "union_pass": len(p_solved | t_solved) / 20,
        },
        "panel_only_uids": sorted(panel_only),
        "joint_failure_uids": sorted(neither),
    }, indent=2))
    print(f"\nwrote {OUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
