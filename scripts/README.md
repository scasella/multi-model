# scripts

Flat layout, grouped by filename prefix. Run from repo root, e.g.:

```bash
python scripts/eval_math500_vibecheck.py tag=panel_l5_full variant=panel
bash   scripts/rl_multipersona_math.sh
```

## Training (`rl_*`)

RL drivers built on `tinker_cookbook`. Panel = `<mutipersonaDebate>` scaffold, think = native Qwen3 `<think>` thinking.

| Script | What it trains |
|---|---|
| `rl_multipersona_math.py` / `.sh` | **Main run.** Qwen3-30B-A3B-Base + `<mutipersonaDebate>` on MATH-500-style categories. LoRA rank 32, ~200 steps. |
| `rl_multipersona_math_smoke.sh` | Single-step smoke test of the math driver. |
| `rl_multipersona_gsm8k.py` / `.sh` | Panel scaffold on GSM8K (warmup). |
| `rl_multipersona_gsm8k_extend_160.sh` | Extend the GSM8K run to 160 steps. |
| `rl_multipersona_gsm8k_resume.sh` | Resume a GSM8K checkpoint. |
| `rl_think_math.py` / `.sh` | `<think>`-format baseline on MATH, matched to the panel run. |
| `rl_think_gsm8k.py` / `.sh` | `<think>` baseline on GSM8K. |
| `rl_think_gsm8k_smoke.sh` | Single-step smoke test of the think driver. |
| `smoke_multipersona.py` | Sampling-only smoke for the panel env. |
| `rl_panel_olympiad.py` / `.sh` | **Olympiad hill-climbing exp, panel arm.** RL continuation on the joint-variance-band olympiad pool. |
| `rl_thinking_olympiad.py` / `.sh` | **Olympiad hill-climbing exp, thinking arm.** Same pool + hyperparams; Qwen3-30B-A3B native thinking template. |

### RLVR hill-climbing experiment (run order)

Tests the hypothesis that the panel scaffold's wider per-sample diversity
translates to faster / more efficient RLVR hill-climbing against Qwen3-thinking
on non-saturated olympiad math.

**Design note.** On Qwen3-30B-A3B the two arms' variance bands are largely
disjoint — panel concentrates in OlympiadBench/AMC where thinking is
saturated; thinking concentrates in HMMT/AIME where panel is all_zero.
Rather than force both arms onto a near-empty joint intersection, each
arm trains on its **own** variance band and both arms score on the same
stratified held-out eval. `scripts/build_per_arm_splits.py` is the
canonical splitter; `scripts/intersect_variance_bands.py` is preserved
for the prior joint-band design.

```bash
# 1. assemble the union pool (HMMT + AIME + OlympiadBench + AMC)
python scripts/build_olympiad_pool.py

# 2. variance-band filter, BOTH arms (samples G=8 per problem)
python scripts/filter_variance_band.py --arm panel    --tag panel_g8
python scripts/filter_variance_band.py --arm thinking --tag thinking_g8

# 3. per-arm split + shared held-out ->
#    data/olympiad_pool/{panel_train, thinking_train, heldout_eval, heldout_aux}.jsonl
python scripts/build_per_arm_splits.py \
    --panel    reports/variance_band/panel_g8/per_problem.jsonl \
    --thinking reports/variance_band/thinking_g8/per_problem.jsonl

# 4. train both arms (matched hyperparameters, different train pools)
bash scripts/rl_panel_olympiad.sh
bash scripts/rl_thinking_olympiad.sh

# 5. compare on the shared heldout_eval.jsonl (reuse existing analyzers)
```

## Evaluation (`eval_*`)

All evals emit `rollouts.jsonl` (per-sample) + `summary.json` (aggregated) under `reports/`.

| Script | Benchmark | Notes |
|---|---|---|
| `eval_math500_vibecheck.py` / `_sweep.sh` | MATH-500 | Panel / think / baseline variants. Used for the main result. |
| `eval_math500_after_math_sweep.sh` | MATH-500 | Post-RL sweep wrapper. |
| `eval_math500_thinking_ceiling.py` / `.sh` | MATH-500 | Qwen3-30B-A3B **thinking** ceiling (no fine-tune). |
| `eval_math500_instruct_ceiling.py` | MATH-500 | Qwen3-30B-A3B instruct ceiling (no thinking). |
| `eval_aime_vibecheck.py` / `_sweep.sh` | AIME 24 + 25 | Non-saturated benchmark — pass@k has room to grow. |
| `eval_aime_thinking_ceiling.py` | AIME 24 + 25 | Thinking ceiling on the same 20-problem slice. |
| `eval_gsm8k.py` / `_sweep.sh` | GSM8K | Sanity check. |

## Analysis (`analyze_*`, `pass_at_k_*`)

| Script | What it computes |
|---|---|
| `analyze_diversity.py` | Mean pairwise cosine distance across n samples per problem — our primary diversity metric. |
| `analyze_hard_problems.py` | Per-problem breakdown: which problems does panel solve that thinking doesn't, and vice versa. |
| `pass_at_k_crossover.py` | Unbiased pass@k curves for panel vs thinking on MATH-500 L5. |
| `pass_at_k_aime.py` | Same for AIME — tests whether the diversity benefit transfers off-saturated benchmarks. |

## Pool assembly + variance-band filtering (olympiad exp)

| Script | What it does |
|---|---|
| `build_olympiad_pool.py` | Assembles `data/olympiad_pool/all.jsonl` from HMMT / AIME / OlympiadBench / AMC via HF datasets. |
| `filter_variance_band.py` | Samples G=8 per problem for one arm; classifies each into all_zero / variance_band / all_one. |
| `build_per_arm_splits.py` | **Canonical.** Per-arm train splits + shared stratified held-out: writes `panel_train.jsonl`, `thinking_train.jsonl`, `heldout_eval.jsonl`, `heldout_aux.jsonl`, `split_summary.json`. |
| `intersect_variance_bands.py` | *Legacy / diagnostic.* Joint-variance-band design (one shared `train.jsonl`). Near-empty on this pool; preserved for reference. |
