# Panel-format adapter vs Qwen3 thinking on Qwen3-30B-A3B

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Backbone: Qwen3-30B-A3B](https://img.shields.io/badge/backbone-Qwen3--30B--A3B-ff6b6b.svg)](https://huggingface.co/Qwen/Qwen3-30B-A3B-Base)
[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Model-multipersona--debate--lora-FFD21E.svg)](https://huggingface.co/scasella91/qwen3-30b-a3b-multipersona-debate-lora)
[![RL: Tinker](https://img.shields.io/badge/RL-Tinker-blueviolet.svg)](https://tinker-console.thinkingmachines.ai/)

This repository contains a panel-format Qwen adapter and evaluations of its
accuracy, completion length, and between-trace embedding dispersion. The
comparison uses Qwen's production thinking model, so it does not isolate the
effect of the reasoning format. Earlier interpretations involving wider search
have been withdrawn.

The adapter is a rank-32 LoRA on `Qwen/Qwen3-30B-A3B-Base`, trained with RL only
to write a panel-of-experts debate (`<mutipersonaDebate>…</mutipersonaDebate>`)
before its answer. The comparison model is `Qwen/Qwen3-30B-A3B` with
`enable_thinking=True`: same architecture, but Qwen's full post-training rather
than these two RL stages, and a larger token limit (16,384 vs 4,096 on MATH-500 /
8,192 on AIME).

**What was measured** (details and caveats in the write-up,
[casella.dev/blog_multipersona.html](https://casella.dev/blog_multipersona.html)):

- **Accuracy, one sample.** The panel is less accurate: MATH-500 level 5
  **58.4% vs 90.9%**, AIME 2024 + 2025 **22.5% vs 73.1%**. It stays below the
  thinking model at every k measured (AIME pass@16: 55% vs 90%).
- **Completion length.** The panel writes much less. Tokens per correct sampled
  answer (total completion tokens ÷ correct samples) are 1,652 vs 8,376 on
  MATH-500 L5 (thinking 5.07×) and 6,257 vs 15,491 on AIME (2.48×). On pairs
  where both were correct, the median length ratio is 6.91× (MATH-500 L5) and
  5.89× (AIME). Many wrong thinking samples stop at the 16,384-token limit,
  which raises its tokens per correct.
- **Between-trace embedding dispersion (exploratory).** Mean pairwise cosine
  distance between a model's samples on the same problem (all-mpnet-base-v2) is
  0.095 vs 0.053 on a 50-problem MATH-500 slice (L1–L5) and 0.119 vs 0.068 on AIME. The embedder reads only
  about the first 384 word pieces of each trace, and persona names are included,
  so this is not a measure of different solution strategies.

**RL follow-up (Apr 25).** On an 877-problem olympiad-math pool the panel has
more mixed-outcome ("variance-band") problems than Qwen3-thinking (382 vs 209),
but 280 of the panel's are problems thinking already solves 8/8, so the counts
locate each model's learning frontier rather than showing wider exploration. One
exploratory 100-step RL run on the panel's band, starting from the GSM8K-stage
adapter (per the run logs), took it from **14% → 29%** on a shared 100-problem
held-out set; the matching thinking arm was not completed. Write-up:
[casella.dev/blog_multipersona_rl.html](https://casella.dev/blog_multipersona_rl.html).

> **Correction (2026-09-23).** Earlier versions of this README presented
> embedding dispersion and pass@k gap closure as evidence of "wider search per
> sample", described tokens per correct answer as a compute or deployment-cost
> advantage, quoted a ~10⁴–10⁵× post-training compute estimate with no
> derivation, reported a per-character dispersion ratio computed from token
> counts, and cited a longer-window dispersion check that did not test what it
> claimed (all-mpnet-base-v2 truncates at 384 word pieces, so the 2,000- and
> 8,000-character runs are identical). Those readings are withdrawn; the numbers
> above are what was measured. The file's git history keeps the earlier text.

**Adapter on Hugging Face:** [`scasella91/qwen3-30b-a3b-multipersona-debate-lora`](https://huggingface.co/scasella91/qwen3-30b-a3b-multipersona-debate-lora)

---

## Repo layout

```
.
├── envs/               four RL environments: panel + think × math + gsm8k
├── scripts/            training, eval, and analysis drivers (see scripts/README.md)
├── reports/            eval + analysis outputs (summary.json tracked, rollouts.jsonl ignored)
│   └── blog_post/      the original April 2026 writeup (revised version: casella.dev)
├── data/               stratified problem slices (SAMPLE.jsonl + schema.json only)
├── RECIPE.md           concise reproduction guide
├── LICENSE             MIT
└── archive/            pre-pivot history (local only, gitignored)
```

## Quickstart

```bash
# 1. set up
uv venv .venv && source .venv/bin/activate
uv pip install -e .   # or install tinker_cookbook deps directly
cp .env.example .env  # fill in TINKER_API_KEY + HF_TOKEN

# 2. stage 1: GSM8K warmup (80 RL steps by default, ~2 h on Tinker;
#    the published adapter's run was resumed to 128 steps, per the run logs)
bash scripts/rl_multipersona_gsm8k.sh

# Export the resulting checkpoint URI (printed by the run) before stage 2:
export PANEL_GSM8K_CHECKPOINT=tinker://<your-session>:train:0/weights/final

# 3. stage 2: MATH continuation (128 RL steps)
bash scripts/rl_multipersona_math.sh

# Export the post-MATH URIs (used by olympiad RL + post-MATH eval sweeps):
export PANEL_MATH_CHECKPOINT=tinker://<your-session>:train:0/weights/final
export PANEL_MATH_CHECKPOINT_SAMPLER=tinker://<your-session>:train:0/sampler_weights/final

# 4. evaluate
python scripts/eval_math500_vibecheck.py tag=panel_l5_full   variant=panel
python scripts/eval_aime_vibecheck.py    tag=aime_panel_n16  n_samples=16

# 5. analyze
python scripts/analyze_diversity.py
python scripts/pass_at_k_aime.py
```

For the olympiad RLVR experiment (variance-band filter + per-arm RL +
hill-climbing eval), see [§6 of `RECIPE.md`](RECIPE.md#6-olympiad-rlvr-hill-climbing-experiment).
For a per-script index, see [`scripts/README.md`](scripts/README.md).

## Key numbers at a glance

| | panel (this adapter) | Qwen3-30B-A3B, thinking on |
|---|---:|---:|
| MATH-500 L5 pass@1 (134 problems) | 0.584 | 0.909 |
| MATH-500 L5 pass@4 | 0.78 | 0.96 |
| AIME 24 + 25 pass@1 (20 problems) | 0.225 | 0.731 |
| AIME 24 + 25 pass@16 | 0.55 | 0.90 |
| Token limit (MATH-500 / AIME) | 4,096 / 8,192 | 16,384 / 16,384 |
| MATH-500 L5 tokens per correct sampled answer | 1,652 | 8,376 *(5.07×)* |
| AIME 24 + 25 tokens per correct sampled answer | 6,257 | 15,491 *(2.48×)* |
| Median length ratio, both-correct pairs (MATH / AIME) | — | 6.91× / 5.89× |
| Mean pairwise cos dist, first ~384 word pieces (MATH-500 50-slice / AIME) | 0.095 / 0.119 | 0.053 / 0.068 |

Sources: `reports/eval_math500_vibecheck/{panel,thinking}_l5_full`,
`reports/eval_aime_vibecheck/aime_{panel_postmath,thinking}_n16`,
`reports/pass_at_k_crossover`, `reports/pass_at_k_aime`,
`reports/token_efficiency`, `reports/diversity_analysis{,_aime}`.
The length difference on both-correct pairs has a Wilcoxon signed-rank
*p* = 6×10⁻⁵² (MATH-500 L5, n = 306) and 8×10⁻¹³ (AIME, n = 68); this tests
completion length only, and pairs from the same problem are not independent.
Run `python scripts/analyze_token_efficiency.py` to reproduce.

## Bounds on the claim

- The panel is less accurate than the thinking model at pass@1 and at every
  pass@k measured. The pass@k gap narrows with k, as it does whenever the model
  that is behind has room to rise; it is not evidence of wider search.
- Format and post-training are confounded: one side is a base-model LoRA, the
  other Qwen's production model. Nothing here isolates the panel format. A
  matched-format control (a `<think>` arm trained from the same base with the
  same recipe) has not been run.
- Token limits and sample counts differ between the two models.
- The sets are small (20 AIME problems; a 50-problem MATH-500 slice and 134
  level-5 problems), and each training stage is a single run with one seed.
- The dispersion measure sees only the opening of each trace and includes
  persona names.

## Reproducing this work

| stage | wall-time on Tinker | output |
|---|---|---|
| Stage 1: GSM8K warmup (80 steps by default; the published adapter ran 128) | ~2 h | `PANEL_GSM8K_CHECKPOINT` |
| Stage 2: MATH continuation (128 steps) | ~6 h | `PANEL_MATH_CHECKPOINT` (the panel adapter cited in the blog as eval session 44722365) |
| Dispersion + pass@k evals | ~3 h | the dispersion and pass@k tables |
| Olympiad pool build + variance-band filter (G=8, both arms, parallelizable) | ~7 h | `data/olympiad_pool/{panel,thinking}_train.jsonl` |
| Panel olympiad RL (100 steps) | ~3 h | the 14% → 29% hill-climbing curve |
| Thinking olympiad RL (100 steps, matched) | ~37 h (estimate) | not yet run — the open follow-up |

The first five rows add up to about 21 hours of wall time as listed. The last row's ~37 h is an estimate for a run that has not happened.
Row six is the highest-cost open item; we hit a billing wall before it
finished and never restarted. Stage URIs are read from `.env` (see
`.env.example`). The only Tinker session URI hard-coded as a default in
the scripts is the publicly-cited `44722365-…` panel-MATH checkpoint
(`build_case_study_transcripts.py`, `chat_panel.py`); both are
overridable via `PANEL_MATH_CHECKPOINT_SAMPLER`.

## Try the model

The post-MATH-RL adapter is published as a standalone PEFT LoRA at
[`scasella91/qwen3-30b-a3b-multipersona-debate-lora`](https://huggingface.co/scasella91/qwen3-30b-a3b-multipersona-debate-lora)
(MIT, 3.4 GB bf16). It loads on top of `Qwen/Qwen3-30B-A3B-Base` via
`transformers + peft` — no Tinker account required. You'll want ~60 GB of
GPU memory for the base model; the adapter itself adds negligible runtime
overhead.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id = "Qwen/Qwen3-30B-A3B-Base"
adapter_id = "scasella91/qwen3-30b-a3b-multipersona-debate-lora"

tok = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_id, torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(model, adapter_id)
model.eval()

PROMPT = (
    "A conversation between User and Multi-Persona Panel of Experts. "
    "The user asks a question, and the Multi-Persona Panel of Experts solves it. "
    "The Multi-Persona Panel of Experts first deliberates and debates the reasoning "
    "process with each other and then provides the user with the answer. "
    "The deliberation process and answer are enclosed within "
    "<mutipersonaDebate>...</mutipersonaDebate> and <answer>...</answer> tags, "
    "respectively, i.e., <mutipersonaDebate> deliberation process here "
    "</mutipersonaDebate> <answer>answer here </answer>. "
    "User: {problem}. Assistant: "
)

inputs = tok(PROMPT.format(problem="If 2x + 3 = 11, what is x?"),
             return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=1024, temperature=1.0, do_sample=True)
print(tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

The model card on Hugging Face has the full caveat list, including the known
gap on per-sample accuracy and the experimental status of MoE expert LoRA
serving in vLLM/SGLang.

## License

MIT — see [`LICENSE`](LICENSE).
