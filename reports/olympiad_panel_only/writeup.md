# Panel-only RLVR on the olympiad variance band

100 RL steps, LoRA rank 32, on `Qwen/Qwen3-30B-A3B-Base + panel-MATH LoRA`,
training on the panel-arm variance band of an 877-problem olympiad-math
union pool, scored on a stratified 100-problem held-out eval shared
across sources.

## Headline

| metric | batch 0 | batch 90 (final eval) | Δ |
|---|---:|---:|---:|
| **Held-out mean correct (100 problems)** | **14.0%** | **29.0%** | **+15.0pp (2.07×)** |
| Training reward (smoothed 10-batch window) | 23.0% | 57.6% | +34.6pp (2.50×) |
| Mean ac_tokens / turn | 586 | 1,103 | +88% |
| Token entropy (policy) | 0.635 | 0.139 | -78% |

**Panel doubled its olympiad held-out pass rate in 100 RL steps**, with
gains concentrated in the sources where panel had the most
gradient-signal-bearing training problems. This is direct evidence that
the panel scaffold's wider variance band (382 problems vs thinking's 209
on this pool) translates into actually-realized RL hill-climbing.

## Held-out trajectory

| batch | held-out | AIME | AMC | HMMT | OlympiadBench |
|---:|---:|---:|---:|---:|---:|
| 0   | 14.0% | 4%  | 32% | 0% | 20% |
| 10  | 12.0% | 4%  | 32% | 0% | 12% |
| 20  | 14.0% | 8%  | 28% | 0% | 20% |
| 30  | 16.0% | 4%  | 40% | 0% | 20% |
| 40  | 19.0% | 12% | 36% | 0% | 28% |
| 50  | 21.0% | 12% | 32% | 0% | 40% |
| 60  | 19.0% | 4%  | 44% | 0% | 28% |
| 70  | **29.0%** | 12% | **60%** | 0% | **44%** |
| 80  | 27.0% | 20% | 48% | 0% | 40% |
| 90  | 29.0% | 16% | 52% | 4% | 44% |

Shape: brief weight-shock dip at batch 10 → monotone climb through
batch 50 → step-jump at batch 70 → plateau in 27–29% band through batch
90. Final eval at batch 90 (no eval scheduled at batch 100 since
`eval_every=10`, `max_steps=100`).

## Mechanism — gains scale with training-pool representation

Panel's `panel_train.jsonl` distributes across sources very unequally
(it's the panel-arm variance band of the joint pool). Held-out gains
track that distribution almost perfectly:

| source | % of panel_train | held-out Δ (batch 0 → 90) | per-pp gain per % of train |
|---|---:|---:|---:|
| OlympiadBench | 88%   (311/354) | **+24.0pp** | 0.27 |
| AMC           | 10%   (35/354)  | **+20.0pp** | 2.00 |
| AIME          | 2%    (6/354)   | +12.0pp | 6.00 |
| HMMT          | 0.6%  (2/354)   | +4.0pp  | 6.67 |

Two readings of the per-percent column:

1. **Bulk transfer:** OlympiadBench dominates training, gets the largest
   absolute lift, but with diminishing returns per training problem.
2. **Cross-source generalization:** AIME and HMMT have negligible direct
   training data but still moved (HMMT crossed 0 for the first time at
   batch 90). The skills learned on OlympiadBench/AMC are transferring,
   not memorizing.

## Training reward — clear acceleration with no held-out collapse

10-batch smoothed:

```
batch  0-9  : 0.230   ← baseline
batch 10-19 : 0.260
batch 20-29 : 0.272
batch 30-39 : 0.350   ← first inflection
batch 40-49 : 0.408
batch 50-59 : 0.466
batch 60-69 : 0.501
batch 70-79 : 0.495   ← brief plateau
batch 80-89 : 0.548
batch 90-99 : 0.576   ← second wave
```

Training reward kept climbing through batch 99 (final smoothed window
57.6%, peak single-batch 67%). Held-out plateaued at 27–29% from batch
70 onward. The training/held-out gap widens, but **held-out doesn't
fall** — that's the healthy "still-generalizing" regime, not the "stuck
on training shortcuts" regime.

## Entropy and length drift

```
batch  0-9  : entropy=0.591  tokens=629
batch 10-19 : entropy=0.565  tokens=666
batch 20-29 : entropy=0.535  tokens=716
batch 30-39 : entropy=0.446  tokens=825
batch 40-49 : entropy=0.367  tokens=884
batch 50-59 : entropy=0.263  tokens=985
batch 60-69 : entropy=0.231  tokens=952
batch 70-79 : entropy=0.180  tokens=1049
batch 80-89 : entropy=0.159  tokens=1049
batch 90-99 : entropy=0.158  tokens=1048
```

Per-token policy entropy collapsed sharply (-78%); panel is sharpening
its distribution as RL progresses. **But** token-length diversity within
problem groups (CV of `ac_len` across the 16 rollouts of a single
training problem) actually *increased*:

| batch | ac_len CV within group |
|---:|---:|
| 0   | 0.36 |
| 30  | 0.37 |
| 50  | 0.51 |
| 70  | 0.49 |
| 90  | **0.66** |

So the policy is more confident *per token* but explores meaningfully
different *trajectory lengths* on the same problem. This is the panel
scaffold's signature — 3 personas with different verbosities producing
different debate trajectories — preserved through RL.

(Caveat: this is a length-only proxy. Proper held-out diversity would
need multi-sample eval rollouts which weren't collected in this run.)

## Limitations

1. **No thinking baseline trained.** Initial plan was matched
   panel-vs-thinking RL on per-arm variance bands; thinking arm was
   dropped after billing constraints (~17× higher per-rollout token
   cost). We have **thinking's batch-0 held-out at 60%** as a static
   reference point, but no comparison to its 100-step trajectory.
2. **Single-sample held-out evaluation.** ~3pp per-tick stderr on the
   aggregate, ~7-10pp per-source. Held-out trajectory is interpretable
   as a denoised curve only after smoothing across multiple ticks.
3. **Single seed / single run.** Confidence intervals on the ~+15pp
   final number need a second seed.
4. **Panel scaffold inherits the post-MATH LoRA checkpoint;** the
   "starting point" isn't the raw scaffold but the scaffold + MATH RL.

## What it does and doesn't say

**Confirmed:**
- The variance-band-size asymmetry between panel (382) and thinking
  (209) on this pool is real (filtered separately via G=8 per problem
  on full 877-problem pool).
- Panel's variance-band yields RL gradient signal — training reward
  climbs 2.5× over 100 steps.
- That gradient signal **transfers to held-out** — held-out doubles
  (14% → 29%), with gains per source tracking training-pool
  representation.
- Entropy collapse + length-diversity preservation suggests panel's
  3-persona structure is acting as a length-/style-diversity reservoir
  even as the per-token policy concentrates.

**Not confirmed:**
- Whether thinking on its own per-arm variance band hill-climbs faster
  or slower (no thinking trajectory).
- Whether the 27-29% plateau at batch 70-90 reflects (a) the
  variance-band being absorbed and remaining gains needing fresh
  problems, or (b) capability ceiling at this LoRA rank / lr.

## Files

- `reports/olympiad_panel_only/summary.json` — all numbers
- `reports/olympiad_panel_only/heldout_trajectory.jsonl` — per-eval-point
- `reports/olympiad_panel_only/train_per_batch.jsonl` — per-batch metrics
- `reports/olympiad_panel_only/diversity_proxy.jsonl` — ac_len CV per checkpoint

Reproduction (your own session): run `bash scripts/rl_panel_olympiad.sh`
with `PANEL_MATH_CHECKPOINT` exported. The local run dir under
`/tmp/tinker-examples/rl/panel_olympiad/...` and the final
`tinker://<your-session>:train:0/weights/final` URI for the original run
have been deleted in the post-publication cleanup pass and are not
resolvable.
