# reports

Eval + analysis outputs. Large per-sample files (`rollouts.jsonl`) are gitignored;
aggregated `summary.json` / `results.json` / `per_problem.jsonl` files are tracked
so the blog figures and tables can be regenerated without rerunning a full eval.

## Folders

| Folder | Written by | Contents |
|---|---|---|
| `blog_post/` | (hand-written) | `blog_multipersona.html` — the writeup |
| `eval_math500_vibecheck/` | `scripts/eval_math500_vibecheck.py`, `eval_math500_thinking_ceiling.py`, `eval_math500_instruct_ceiling.py` | Per-tag run dirs: `summary.json` (tracked), `rollouts.jsonl` (ignored) |
| `eval_aime_vibecheck/` | `scripts/eval_aime_vibecheck.py`, `eval_aime_thinking_ceiling.py` | Same layout, AIME 24 + 25 |
| `eval_gsm8k/` | `scripts/eval_gsm8k.py` | GSM8K sanity checks |
| `diversity_analysis/` | `scripts/analyze_diversity.py` | MATH-500 diversity: summary + per-problem |
| `diversity_analysis_aime/` | `scripts/analyze_diversity.py` (AIME inputs) | AIME diversity |
| `diversity_analysis_longhead/` | `scripts/analyze_diversity.py` | Sensitivity pass at a longer trace-head cutoff |
| `pass_at_k_crossover/` | `scripts/pass_at_k_crossover.py` | MATH-500 L5 pass@k curves + deltas |
| `pass_at_k_aime/` | `scripts/pass_at_k_aime.py` | AIME pass@k + gap-closure |
| `hard_problem_analysis/` | `scripts/analyze_hard_problems.py` | Per-problem overlap: panel-only / think-only / both / neither |
| `variance_band/` | `scripts/filter_variance_band.py` | Per-problem `pass@G` classification per arm (`panel_g8/`, `thinking_g8/` are the full-pool runs; `*_smoke/`, `*_mixed_smoke/` are kept for traceability of the disjoint-frontier discovery) |
| `olympiad_panel_only/` | `scripts/rl_panel_olympiad.py` + post-hoc analysis | Held-out trajectory across 100 RL steps + per-batch metrics + length-CV diversity proxy |

## Blog figure → artifact map

Every claim in the blog cites a tracked file in this directory. To verify or
regenerate any number you see in the writeup:

| section in blog | artifact |
|---|---|
| Headline diversity (`+78.2%`/`+75.6%`) | `diversity_analysis/summary.json`, `diversity_analysis_aime/summary.json` |
| Pass@k closure on AIME — Figure 1 | `pass_at_k_aime/` |
| Pass@k closure on MATH500 L5 | `pass_at_k_crossover/` |
| Variance-band sizes (panel 382 / thinking 209) | `variance_band/panel_g8/summary.json`, `variance_band/thinking_g8/summary.json` |
| Joint-band contingency table | `../data/olympiad_pool/split_summary.json` |
| Hill-climbing curve — Figure 2 + per-source gains | `olympiad_panel_only/summary.json` (+ sibling `heldout_trajectory.jsonl`, `train_per_batch.jsonl`, `diversity_proxy.jsonl`) |
| Vanilla Qwen3-thinking baseline (~60% pass@1) | `eval_aime_vibecheck/aime_thinking_n16/summary.json`, `eval_math500_vibecheck/thinking_l5_full/summary.json` |
| Per-problem solve overlap (panel-only vs think-only) | `hard_problem_analysis/summary.json` |

## A note on `tinker://` URIs in these files

Most `summary.json` files here record the `sampler_path` or `checkpoint`
URI of the original run that produced them — useful for traceability,
but bear in mind:

- The URIs are session IDs from a private Tinker account; they aren't
  secrets, but they aren't resolvable for anyone else either.
- All but two of those sessions (`44722365-…` panel-MATH and
  `e1a9b8bf-…` panel-GSM8K stage-2 final) were deleted in a
  post-publication cleanup. The retained pair are the load-bearing
  artifacts cited in the blog and used as RL inits downstream.
- To reproduce, run the recipe from your own account; the URIs you
  produce will end up in your own regenerated `summary.json` files.

## Regenerating rollouts

If you want the raw `rollouts.jsonl` back, rerun the corresponding eval script.
Each script writes to a `tag`-named subfolder of the matching `reports/eval_*`
directory. Example:

```bash
python scripts/eval_aime_vibecheck.py \
    tag=aime_panel_postmath_n16 \
    variant=panel \
    n_samples=16
# writes reports/eval_aime_vibecheck/aime_panel_postmath_n16/{rollouts.jsonl,summary.json}
```

Analyzers (`analyze_diversity.py`, `pass_at_k_*`, `analyze_hard_problems.py`)
read the `rollouts.jsonl` files directly, so run evals first.
