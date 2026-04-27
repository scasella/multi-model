"""Token-efficiency comparison between the multi-persona-debate adapter and
vanilla Qwen3-thinking, using existing eval rollouts.

Question this script answers:
  When the panel and Qwen3-thinking each emit one sample on the same problem,
  and BOTH samples are correct, how do their token counts compare?

The case-study gallery (n=15 shared-correct items) shows the panel using
~6× fewer chars on every shared-correct problem. This script tests whether
that pattern holds at population scale on the full MATH-500 L5 and AIME
24+25 rollout sets.

The rollouts.jsonl files already include `completion_tokens` from sampling
time, so this is a few-seconds-of-CPU analysis, no GPU and no re-tokenization.

Usage:
    python scripts/analyze_token_efficiency.py
    python scripts/analyze_token_efficiency.py --bench math
    python scripts/analyze_token_efficiency.py --bench aime --json reports/token_efficiency/aime.json

Outputs (to stdout + optional --json):
  - n paired (problem, sample_idx) pairs
  - per-bucket stats: both-correct, panel-only, thinking-only, neither
  - mean / median / geometric-mean token-ratio (thinking / panel)
  - paired Wilcoxon test on the both-correct subset
  - mean tokens-per-correct (cost of producing one correct answer)
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent

# Eval-output paths. Both arms sampled at temperature 1.0 with matched
# max_tokens budgets; see reports/{eval_*}/summary.json for the exact knobs.
PAIRS = {
    "math": {
        "panel": ROOT / "reports/eval_math500_vibecheck/panel_l5_full/rollouts.jsonl",
        "think": ROOT / "reports/eval_math500_vibecheck/thinking_l5_full/rollouts.jsonl",
        "key": ("unique_id", "sample_idx"),
        "label": "MATH-500 L5",
    },
    "aime": {
        "panel": ROOT / "reports/eval_aime_vibecheck/aime_panel_postmath_n16/rollouts.jsonl",
        "think": ROOT / "reports/eval_aime_vibecheck/aime_thinking_n16/rollouts.jsonl",
        "key": ("year", "problem_idx", "sample_idx"),
        "label": "AIME 24+25",
    },
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def index_by_key(rows: Iterable[dict], key: tuple[str, ...]) -> dict[tuple, dict]:
    return {tuple(r[k] for k in key): r for r in rows}


def pair_rollouts(panel_path: Path, think_path: Path, key: tuple[str, ...]) -> list[dict]:
    """Inner-join panel and thinking rollouts on (problem-id, sample_idx)."""
    panel = index_by_key(load_jsonl(panel_path), key)
    think = index_by_key(load_jsonl(think_path), key)
    paired = []
    for k in panel.keys() & think.keys():
        p, t = panel[k], think[k]
        paired.append({
            "key": k,
            "panel_tokens": p["completion_tokens"],
            "think_tokens": t["completion_tokens"],
            "panel_correct": bool(p["correct"]),
            "think_correct": bool(t["correct"]),
        })
    return paired


def bucket_stats(name: str, items: list[dict]) -> dict:
    """Return token-stats for a list of paired items. Includes ratio stats."""
    n = len(items)
    if n == 0:
        return {"name": name, "n": 0}
    p = [it["panel_tokens"] for it in items]
    t = [it["think_tokens"] for it in items]
    ratios = [t[i] / p[i] for i in range(n) if p[i] > 0]
    log_ratios = [math.log(r) for r in ratios if r > 0]
    n_thinking_longer = sum(1 for r in ratios if r > 1)
    return {
        "name": name,
        "n": n,
        "panel_tokens_mean":     statistics.fmean(p),
        "panel_tokens_median":   statistics.median(p),
        "think_tokens_mean":     statistics.fmean(t),
        "think_tokens_median":   statistics.median(t),
        "ratio_mean":            statistics.fmean(ratios) if ratios else None,
        "ratio_median":          statistics.median(ratios) if ratios else None,
        "ratio_geomean":         math.exp(statistics.fmean(log_ratios)) if log_ratios else None,
        "ratio_min":             min(ratios) if ratios else None,
        "ratio_max":             max(ratios) if ratios else None,
        "global_ratio":          sum(t) / sum(p) if sum(p) > 0 else None,
        "thinking_longer_in":    n_thinking_longer,
        "thinking_longer_frac":  n_thinking_longer / n if n else None,
    }


def wilcoxon_signed_rank(diffs: list[float]) -> dict | None:
    """Two-sided Wilcoxon signed-rank test on the diffs.
    Returns None if scipy isn't installed; the headline numbers don't depend
    on a p-value, this is just a defensible add-on."""
    try:
        from scipy.stats import wilcoxon  # type: ignore
    except ImportError:
        return None
    nz = [d for d in diffs if d != 0]
    if len(nz) < 5:
        return None
    res = wilcoxon(nz, alternative="two-sided")
    return {"statistic": float(res.statistic), "pvalue": float(res.pvalue), "n_nonzero": len(nz)}


def fmt_stat(s: dict) -> list[str]:
    if s["n"] == 0:
        return [f"  {s['name']:<22} (n=0, no items)"]
    lines = [
        f"  {s['name']:<22} n = {s['n']}",
        f"    panel  tokens (mean / median): {s['panel_tokens_mean']:>7.0f} / {s['panel_tokens_median']:>7.0f}",
        f"    think  tokens (mean / median): {s['think_tokens_mean']:>7.0f} / {s['think_tokens_median']:>7.0f}",
        f"    ratio (think / panel)        :",
        f"        mean      {s['ratio_mean']:>5.2f}x",
        f"        median    {s['ratio_median']:>5.2f}x",
        f"        geom mean {s['ratio_geomean']:>5.2f}x",
        f"        range     {s['ratio_min']:>5.2f}x - {s['ratio_max']:.2f}x",
        f"        global Σt/Σp = {s['global_ratio']:.2f}x",
        f"    thinking longer in: {s['thinking_longer_in']} / {s['n']}  ({s['thinking_longer_frac']:.0%})",
    ]
    return lines


def analyze(bench: str) -> dict:
    cfg = PAIRS[bench]
    paired = pair_rollouts(cfg["panel"], cfg["think"], cfg["key"])

    both = [it for it in paired if it["panel_correct"] and it["think_correct"]]
    panel_only = [it for it in paired if it["panel_correct"] and not it["think_correct"]]
    think_only = [it for it in paired if not it["panel_correct"] and it["think_correct"]]
    neither = [it for it in paired if not it["panel_correct"] and not it["think_correct"]]

    # Cost per correct answer: total tokens spent / correct samples.
    # Lower = better. Apples-to-apples because both arms saw the same
    # (problem, sample_idx) draw.
    panel_total_tokens = sum(it["panel_tokens"] for it in paired)
    think_total_tokens = sum(it["think_tokens"] for it in paired)
    panel_correct_n = len(both) + len(panel_only)
    think_correct_n = len(both) + len(think_only)

    cost_per_correct = {
        "panel_tokens_per_correct":    panel_total_tokens / panel_correct_n if panel_correct_n else None,
        "thinking_tokens_per_correct": think_total_tokens / think_correct_n if think_correct_n else None,
    }

    # Wilcoxon on the both-correct subset (paired by problem×sample_idx).
    diffs = [it["think_tokens"] - it["panel_tokens"] for it in both]
    wilcoxon = wilcoxon_signed_rank(diffs)

    return {
        "bench": cfg["label"],
        "n_paired": len(paired),
        "buckets": {
            "both_correct":  bucket_stats("both correct",   both),
            "panel_only":    bucket_stats("panel only",     panel_only),
            "thinking_only": bucket_stats("thinking only",  think_only),
            "neither":       bucket_stats("neither correct", neither),
            "all_paired":    bucket_stats("all paired",     paired),
        },
        "cost_per_correct": cost_per_correct,
        "wilcoxon_both_correct": wilcoxon,
        "panel_correct_count": panel_correct_n,
        "thinking_correct_count": think_correct_n,
    }


def print_report(report: dict) -> None:
    print(f"\n{'='*60}")
    print(f" {report['bench']} — token-efficiency analysis")
    print(f"{'='*60}")
    print(f" {report['n_paired']} paired (problem, sample_idx) draws")
    print(f" panel correct on    {report['panel_correct_count']:>5} / {report['n_paired']}  ({report['panel_correct_count']/report['n_paired']:.1%})")
    print(f" thinking correct on {report['thinking_correct_count']:>5} / {report['n_paired']}  ({report['thinking_correct_count']/report['n_paired']:.1%})")
    print()
    for bucket in ["both_correct", "panel_only", "thinking_only", "neither", "all_paired"]:
        for ln in fmt_stat(report["buckets"][bucket]):
            print(ln)
        print()
    cpc = report["cost_per_correct"]
    print(" cost per correct answer (total tokens / correct count):")
    if cpc["panel_tokens_per_correct"] is not None:
        print(f"   panel   : {cpc['panel_tokens_per_correct']:>8.0f} tokens / correct")
    if cpc["thinking_tokens_per_correct"] is not None:
        print(f"   thinking: {cpc['thinking_tokens_per_correct']:>8.0f} tokens / correct")
    if cpc["panel_tokens_per_correct"] and cpc["thinking_tokens_per_correct"]:
        ratio = cpc["thinking_tokens_per_correct"] / cpc["panel_tokens_per_correct"]
        print(f"   thinking is {ratio:.2f}x more expensive per correct answer")
    print()
    if report["wilcoxon_both_correct"]:
        w = report["wilcoxon_both_correct"]
        print(f" Wilcoxon signed-rank (both-correct subset, paired think - panel diffs):")
        print(f"   n_nonzero = {w['n_nonzero']}, W = {w['statistic']:.1f}, p = {w['pvalue']:.2e}")
    else:
        print(" (scipy not installed; Wilcoxon test skipped)")
    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", choices=["math", "aime", "all"], default="all")
    p.add_argument("--json", type=Path, default=None,
                   help="Optional path to write the full report as JSON.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    benches = ["math", "aime"] if args.bench == "all" else [args.bench]
    reports = {}
    for b in benches:
        r = analyze(b)
        reports[b] = r
        print_report(r)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(reports, indent=2))
        print(f"  wrote full report to {args.json}")


if __name__ == "__main__":
    main()
