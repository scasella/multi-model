"""Pass@k curves: panel (post-MATH) vs Qwen3-30B-A3B thinking.

Uses the unbiased pass@k estimator from the Codex paper:
    pass@k = 1 - C(n-c, k) / C(n, k)
where c = number of correct samples out of n total draws.

Two slices:
  (A) MATH500 stratified-50 (L1..L5, 10/level):
      - panel:     reports/eval_math500_vibecheck/math_final_panel/rollouts.jsonl      (n=4)
      - thinking:  reports/eval_math500_vibecheck/thinking_native/rollouts.jsonl       (n=4)
  (B) MATH500 L5 full-sweep (134 problems):
      - panel:     reports/eval_math500_vibecheck/panel_l5_full/rollouts.jsonl         (n=8)
      - thinking:  reports/eval_math500_vibecheck/thinking_l5_full/rollouts.jsonl      (n=4)

For each problem × variant we count n and c, then average pass@k across problems.
Report:
  - pass@k curve per variant (k from 1..min(panel_n, thinking_n) for A; 1..4 for B)
  - paired delta (panel - thinking) per problem at each k, with SE and t-stat
  - crossover k (if any) where panel ≥ thinking
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(path: Path) -> dict[str, tuple[int, int]]:
    """uid -> (n_samples, n_correct)."""
    by_uid: dict[str, list[int]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            by_uid[r["unique_id"]].append(int(r["correct"]))
    return {u: (len(v), sum(v)) for u, v in by_uid.items()}


def pass_at_k(n: int, c: int, k: int) -> float:
    if k > n:
        raise ValueError(f"k={k} > n={n}")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def curve(stats: dict[str, tuple[int, int]], ks: list[int]) -> dict[int, list[float]]:
    out: dict[int, list[tuple[str, float]]] = {k: [] for k in ks}
    for uid, (n, c) in stats.items():
        for k in ks:
            if k <= n:
                out[k].append((uid, pass_at_k(n, c, k)))
    return out


def paired_delta(panel_curve, think_curve, ks):
    """Paired (panel - thinking) over problems that exist in both curves at k."""
    rows = []
    for k in ks:
        p_by_uid = dict(panel_curve[k])
        t_by_uid = dict(think_curve[k])
        shared = sorted(set(p_by_uid) & set(t_by_uid))
        deltas = [p_by_uid[u] - t_by_uid[u] for u in shared]
        n = len(deltas)
        if n == 0:
            continue
        mean = sum(deltas) / n
        if n > 1:
            var = sum((d - mean) ** 2 for d in deltas) / (n - 1)
            se = math.sqrt(var / n)
            t = mean / se if se > 0 else float("inf") if mean > 0 else (float("-inf") if mean < 0 else 0.0)
        else:
            se, t = float("nan"), float("nan")
        panel_mean = sum(p_by_uid[u] for u in shared) / n
        think_mean = sum(t_by_uid[u] for u in shared) / n
        rows.append({
            "k": k,
            "n_paired": n,
            "panel_pass_at_k": panel_mean,
            "thinking_pass_at_k": think_mean,
            "delta_mean": mean,
            "delta_se": se,
            "t": t,
        })
    return rows


def report(name: str, panel_path: Path, think_path: Path, ks: list[int]) -> dict:
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    p_stats = load(panel_path)
    t_stats = load(think_path)
    shared = sorted(set(p_stats) & set(t_stats))
    print(f"panel:    {len(p_stats)} problems, samples/problem = {next(iter(p_stats.values()))[0]}")
    print(f"thinking: {len(t_stats)} problems, samples/problem = {next(iter(t_stats.values()))[0]}")
    print(f"shared:   {len(shared)} problems")

    p_curve = curve({u: p_stats[u] for u in shared}, ks)
    t_curve = curve({u: t_stats[u] for u in shared}, ks)
    rows = paired_delta(p_curve, t_curve, ks)

    print(f"\n{'k':>3} {'n':>4} {'panel':>10} {'thinking':>10} {'Δ':>10} {'SE':>8} {'t':>7} {'winner':>10}")
    for r in rows:
        winner = "panel" if r["delta_mean"] > 0 else ("thinking" if r["delta_mean"] < 0 else "tie")
        print(f"{r['k']:>3} {r['n_paired']:>4} {r['panel_pass_at_k']:>10.4f} {r['thinking_pass_at_k']:>10.4f} "
              f"{r['delta_mean']:>+10.4f} {r['delta_se']:>8.4f} {r['t']:>+7.2f} {winner:>10}")

    crossover = next((r["k"] for r in rows if r["delta_mean"] >= 0), None)
    print(f"\nCrossover k (panel >= thinking): {crossover}")
    return {"slice": name, "rows": rows, "crossover_k": crossover}


def main() -> None:
    results = []
    results.append(report(
        "MATH500 stratified-50 (L1-L5, n=4 each)",
        ROOT / "reports/eval_math500_vibecheck/math_final_panel/rollouts.jsonl",
        ROOT / "reports/eval_math500_vibecheck/thinking_native/rollouts.jsonl",
        ks=[1, 2, 3, 4],
    ))
    results.append(report(
        "MATH500 L5-full (134 problems, panel n=8, thinking n=4)",
        ROOT / "reports/eval_math500_vibecheck/panel_l5_full/rollouts.jsonl",
        ROOT / "reports/eval_math500_vibecheck/thinking_l5_full/rollouts.jsonl",
        ks=[1, 2, 3, 4],  # limited by thinking n=4
    ))

    # Extra: panel-only pass@k up to 8 on L5, alongside thinking's pass@4 ceiling,
    # to see if panel's budget-doubled regime catches up.
    print(f"\n{'=' * 70}\nL5-full: panel pass@k for k=1..8 vs thinking pass@4 = ceiling\n{'=' * 70}")
    p_stats = load(ROOT / "reports/eval_math500_vibecheck/panel_l5_full/rollouts.jsonl")
    t_stats = load(ROOT / "reports/eval_math500_vibecheck/thinking_l5_full/rollouts.jsonl")
    shared = sorted(set(p_stats) & set(t_stats))
    p_curve = curve({u: p_stats[u] for u in shared}, [1, 2, 3, 4, 5, 6, 7, 8])
    t_pass4 = sum(pass_at_k(*t_stats[u], 4) for u in shared) / len(shared)
    print(f"\n{'k':>3} {'panel pass@k':>14} {'thinking pass@4 ref':>22}")
    for k in [1, 2, 3, 4, 5, 6, 7, 8]:
        vals = [v for _, v in p_curve[k]]
        m = sum(vals) / len(vals)
        print(f"{k:>3} {m:>14.4f} {t_pass4:>22.4f}")

    out_dir = ROOT / "reports/pass_at_k_crossover"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
