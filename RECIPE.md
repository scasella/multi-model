# Recipe — panel-of-experts scaffold on Qwen3-30B-A3B-Base

This recipe produces a LoRA adapter on `Qwen/Qwen3-30B-A3B-Base` that emits a
panel-of-experts debate (`<mutipersonaDebate>…</mutipersonaDebate>`) in place
of a `<think>` monologue before answering. The headline finding is
**wider search per sample** — see `reports/blog_post/blog_multipersona.html`.

For the full pre-pivot history (Qwen3-8B-Base with `<debate>/<proposer>/<skeptic>/<arbiter>`),
see `archive/RECIPE_pre_pivot.md`.

---

## 1. Environment

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .  # if a pyproject exists; otherwise install tinker_cookbook deps directly
```

Required env vars in `.env` (gitignored):

```
TINKER_API_KEY=...                # Tinker auth — required for any RL or sampling
HF_TOKEN=...                      # HuggingFace — for dataset downloads (MATH, AIME, OlympiadBench, AMC)
```

Optional env vars for chaining stages (set after each stage finishes; export for use by downstream scripts):

```
# Set after rl_multipersona_gsm8k.sh completes — its weights/final + sampler_weights/final URIs:
PANEL_GSM8K_CHECKPOINT=tinker://<your-gsm8k-session>:train:0/weights/final
PANEL_GSM8K_CHECKPOINT_SAMPLER=tinker://<your-gsm8k-session>:train:0/sampler_weights/final

# Set after rl_multipersona_math.sh completes — used by olympiad RL + post-MATH evals:
PANEL_MATH_CHECKPOINT=tinker://<your-math-session>:train:0/weights/final
PANEL_MATH_CHECKPOINT_SAMPLER=tinker://<your-math-session>:train:0/sampler_weights/final
```

The Python defaults all fall back to `tinker://<your-…-session>:…` placeholders if the env var
isn't set, which makes them obvious in any log line if you forget to export.

All scripts are run from the repo root.

## 2. Environments (`envs/`)

| File | Format |
|---|---|
| `envs/multipersona_math.py` | MATH-500-style problems → panel scaffold |
| `envs/multipersona_gsm8k.py` | GSM8K → panel scaffold (warmup) |
| `envs/think_math.py` | MATH → `<think>` baseline, matched prompt/grader |
| `envs/think_gsm8k.py` | GSM8K → `<think>` baseline |

Panel envs inject a system prompt that describes the three-persona format and
reward only when `\boxed{…}` inside `<answer>…</answer>` matches.

## 3. Training

Two stages, run in order:

```bash
# Stage 1: GSM8K warmup, 80 steps (the scaffold learns its tag structure here)
bash scripts/rl_multipersona_gsm8k.sh

# Export the resulting weights/final URI before stage 2:
export PANEL_GSM8K_CHECKPOINT=tinker://<your-gsm8k-session>:train:0/weights/final

# Stage 2: MATH train split, 128 steps continuation
bash scripts/rl_multipersona_math.sh
```

Both are LoRA rank-32 fine-tunes on a single node via Tinker. The
`rl_multipersona_math.sh` launcher refuses to start if `PANEL_GSM8K_CHECKPOINT`
isn't set — that's intentional, since it can't init from anywhere else.

`<think>` baseline matched to the panel run (used in pass@k comparisons):

```bash
bash scripts/rl_think_math.sh
```

After stage 2, export the post-MATH URI so the olympiad section (§6) and the
post-MATH eval sweeps can find it:

```bash
export PANEL_MATH_CHECKPOINT=tinker://<your-math-session>:train:0/weights/final
export PANEL_MATH_CHECKPOINT_SAMPLER=tinker://<your-math-session>:train:0/sampler_weights/final
```

## 4. Evaluation

Run from repo root — all evals emit `rollouts.jsonl` + `summary.json` under `reports/`.

**MATH-500** (panel, think, baseline):

```bash
python scripts/eval_math500_vibecheck.py tag=panel_l5_full       variant=panel
python scripts/eval_math500_vibecheck.py tag=no_panel_l5_full    variant=no_panel
python scripts/eval_math500_thinking_ceiling.py tag=thinking_l5_full
```

**AIME 24 + 25** (non-saturated slice, pass@k has room to grow):

```bash
python scripts/eval_aime_vibecheck.py          tag=aime_panel_postmath_n16 n_samples=16
python scripts/eval_aime_thinking_ceiling.py   tag=aime_thinking_n16       n_samples=16
```

## 5. Analysis

```bash
python scripts/analyze_diversity.py       # mean pairwise cosine distance per problem
python scripts/pass_at_k_crossover.py     # pass@k vs k on MATH-500 L5 (panel vs thinking)
python scripts/pass_at_k_aime.py          # same for AIME
python scripts/analyze_hard_problems.py   # per-problem overlap: panel-only / think-only / both / neither
```

Each analyzer writes its own folder under `reports/` (e.g. `reports/diversity_analysis/`,
`reports/pass_at_k_aime/`). The blog post `reports/blog_post/blog_multipersona.html`
reads those files directly.

## 6. Olympiad RLVR hill-climbing experiment

**Hypothesis.** A panel-scaffolded policy's wider per-sample diversity
raises the fraction of training prompts with `0 < pass@G < 1` (the
"variance band"), which is the only regime where group-relative RLVR
produces gradient signal. If true, the panel should hill-climb faster
than Qwen3-30B-A3B native thinking under matched hyperparameters on an
off-saturated benchmark.

**Design.** Assemble an olympiad-math union pool (HMMT + AIME + OlympiadBench
+ AMC), classify every problem per-model via G=8 sampling, and train each
arm on its **own variance band** (not the joint intersection). Both arms
score on the same stratified held-out eval.

**Why per-arm, not joint.** On this pool the two arms' variance bands are
largely disjoint: panel's concentrates in OlympiadBench/AMC (where
thinking is saturated at all_one) and thinking's concentrates in
HMMT/AIME (where panel is all_zero). Forcing both arms onto the joint
intersection would (a) yield a pool too thin to train on and
(b) discard precisely the signal that the hypothesis predicts — that
each scaffold has a *different*, and for panel *wider*, region of
gradient-signal-bearing problems. The split-summary reports
`panel_vb_size`, `thinking_vb_size`, and joint-band counts so this
asymmetry is visible in the writeup. `scripts/intersect_variance_bands.py`
is preserved for the joint-band reference design.

```bash
# Step A — build the pool (877 problems)
python scripts/build_olympiad_pool.py

# Step B — variance-band filter for each arm (in parallel if you have budget)
python scripts/filter_variance_band.py --arm panel    --tag panel_g8
python scripts/filter_variance_band.py --arm thinking --tag thinking_g8

# Step C — per-arm train + shared held-out splits
python scripts/build_per_arm_splits.py \
    --panel    reports/variance_band/panel_g8/per_problem.jsonl \
    --thinking reports/variance_band/thinking_g8/per_problem.jsonl

# Step D — train both arms with MATCHED hyperparameters
bash scripts/rl_panel_olympiad.sh       # trains on panel_train.jsonl
bash scripts/rl_thinking_olympiad.sh    # trains on thinking_train.jsonl
```

**What to look at after training.**

1. **Variance-band size per arm** (step 0 reading) — is
   `panel_vb_size > thinking_vb_size` on the full pool? This is a
   direct first-order test of the hypothesis.
2. **Variance-band trajectory** — at each checkpoint, rerun the filter
   on that arm's training pool and plot the fraction still in the
   variance band vs step. Panel's should stay wider longer.
3. **Held-out pass@1 and pass@16 vs step** on the shared
   `heldout_eval.jsonl` — the hill-climbing curve. Both arms are scored
   identically on the same 100 problems.
4. **Mean pairwise cosine distance vs step** — does panel preserve
   spread while thinking collapses? Use `scripts/analyze_diversity.py`
   pointed at each checkpoint's held-out rollouts.

**Files.**

- Envs: `envs/olympiad_pool.py` (shared loader + grader), `envs/multipersona_olympiad.py` (panel arm), `envs/thinking_native_olympiad.py` (thinking arm)
- Scripts: `scripts/build_olympiad_pool.py`, `scripts/filter_variance_band.py`, `scripts/build_per_arm_splits.py`, `scripts/rl_panel_olympiad.{py,sh}`, `scripts/rl_thinking_olympiad.{py,sh}`
- Data: `data/olympiad_pool/{all, panel_train, thinking_train, heldout_eval, heldout_aux}.jsonl` (generated)
- Filter outputs: `reports/variance_band/{panel_g8,thinking_g8}/`

## 7. What gets tracked vs ignored

- ✅ tracked: scripts, envs, `summary.json` under `reports/**/`, `data/**/SAMPLE.jsonl`
- ❌ ignored: `rollouts.jsonl` (large; regenerate via eval scripts), `logs/`, `archive/`

See `.gitignore` for the exact list and `reports/README.md` for the mapping
between scripts and the output folders they write.
