# Panel-of-experts scaffold vs `<think>` on Qwen3-30B-A3B

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Backbone: Qwen3-30B-A3B](https://img.shields.io/badge/backbone-Qwen3--30B--A3B-ff6b6b.svg)](https://huggingface.co/Qwen/Qwen3-30B-A3B-Base)
[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Model-multipersona--debate--lora-FFD21E.svg)](https://huggingface.co/scasella91/qwen3-30b-a3b-multipersona-debate-lora)
[![RL: Tinker](https://img.shields.io/badge/RL-Tinker-blueviolet.svg)](https://tinker-console.thinkingmachines.ai/)

A LoRA rank-32 fine-tune of `Qwen/Qwen3-30B-A3B-Base` that replaces the usual
`<think>…</think>` monologue with a three-persona debate
(`<mutipersonaDebate>…</mutipersonaDebate>`) before answering, scored against
Qwen's native thinking model on the same backbone.

**Headline finding #1 — wider search per sample.** Panel reasoning traces
sit +78.2% further apart on MATH-500 and +75.6% further apart on AIME 24 + 25
(mean pairwise cosine distance, all-mpnet-base-v2 embeddings). On AIME the
paired pass@k gap to Qwen3-thinking closes by +15.6pp from k = 1 to k = 16,
monotone in k.

**Headline finding #2 — and the deployment-cost consequence.** On the **374
paired draws** across MATH-500 L5 and AIME 24+25 where both arms produced a
correct answer, thinking burns **5.9–6.9× more tokens (median)** than the
panel — and is longer than the panel in **99.6–99.7%** of all paired draws,
regardless of correctness. At the cost-per-correct-answer level — total
tokens divided by correct samples — thinking is **2.5–5× more expensive**
than the panel. Per-sample accuracy still favors thinking; per-token
efficiency reverses the comparison once compute is in the denominator.
(Wilcoxon signed-rank: *p* = 6×10⁻⁵² on MATH, *p* = 8×10⁻¹³ on AIME. Run
`python scripts/analyze_token_efficiency.py` to reproduce.)

**Follow-up — Apr 25.** The diversity shows up where it matters for RL. On a
fresh 877-problem olympiad-math pool the panel has **1.83× more variance-band
problems** than Qwen3-thinking (382 vs 209) — the only regime where
group-relative RLVR generates non-zero gradient. 100 RL steps on that band
carry the panel from **14% → 29%** on a shared held-out, with per-source
gains scaling with training-pool representation.

The full pipeline is a LoRA rank-32 adapter on the frozen base. The
post-training compute is roughly four to five orders of magnitude under
Qwen's investment in the thinking baseline.

👉 **Full writeup:** [`reports/blog_post/diversity.html`](reports/blog_post/diversity.html)
🤗 **Adapter on Hugging Face:** [`scasella91/qwen3-30b-a3b-multipersona-debate-lora`](https://huggingface.co/scasella91/qwen3-30b-a3b-multipersona-debate-lora)

---

## Repo layout

```
.
├── envs/               four RL environments: panel + think × math + gsm8k
├── scripts/            training, eval, and analysis drivers (see scripts/README.md)
├── reports/            eval + analysis outputs (summary.json tracked, rollouts.jsonl ignored)
│   └── blog_post/      the writeup
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

# 2. stage 1: GSM8K warmup (80 RL steps, ~2 h on Tinker)
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

| | panel (ours) | thinking (Qwen3-30B-A3B native) |
|---|---:|---:|
| MATH-500 L5 avg hit rate | 0.58 | 0.75 |
| MATH-500 L5 pass@4 | 0.90 | 1.00 |
| AIME 24 + 25 pass@1 | 0.23 | 0.73 |
| AIME 24 + 25 pass@16 | 0.55 | 0.75 |
| Mean pairwise cos dist (MATH) | 0.095 | 0.053 |
| Mean pairwise cos dist (AIME) | 0.119 | 0.068 |
| **MATH-500 L5 tokens / correct** | **1,652** | **8,376** *(5.07× more)* |
| **AIME 24 + 25 tokens / correct** | **6,257** | **15,491** *(2.48× more)* |
| Median token ratio, both-correct (MATH / AIME) | — | **6.91× / 5.89×** |

The panel starts behind on pass@1 and **closes the gap sharply with more samples** —
the pattern you'd expect from wider, less redundant search. Per-sample
accuracy favors thinking; per-token cost-per-correct reverses the comparison.

## Bounds on the claim

The panel does **not** outperform the thinking baseline at pass@1. What the
data actually supports:

1. Per-sample semantic spread is wider on both benchmarks (paired t = 5.91 / 4.72).
2. That spread tightens pass@k against thinking on three independent slices.
3. **At fixed correctness, the panel uses 5.9–6.9× fewer tokens than thinking
   (median, both-correct subset; n = 374 paired draws). At the cost-per-correct
   level, thinking is 2.5–5× more expensive. This is the deployment-cost
   reading of the same diversity mechanism.**
4. On a 877-problem olympiad pool, that same spread yields 1.83× more
   variance-band problems than thinking — and 100 LoRA RL steps on those
   problems lift the panel's shared held-out from 14% to 29%.
5. The compute behind all of this is ~10⁴–10⁵× under Qwen's post-training stack.

See the blog post for the full caveat set: embedding-based diversity metric,
small AIME slice, no token-budget-matched baseline (the cost-per-correct
finding partly addresses this — at fixed compute, panel wins; at fixed sample
count, thinking does), no matched thinking-arm RL run.

## Reproducing this work

| stage | wall-time on Tinker | output |
|---|---|---|
| Stage 1: GSM8K warmup (80 steps) | ~2 h | `PANEL_GSM8K_CHECKPOINT` |
| Stage 2: MATH continuation (128 steps) | ~6 h | `PANEL_MATH_CHECKPOINT` (the panel adapter cited in the blog as eval session 44722365) |
| Diversity + pass@k evals | ~3 h | the +78% / +75.6% / +15.6pp numbers |
| Olympiad pool build + variance-band filter (G=8, both arms, parallelizable) | ~7 h | `data/olympiad_pool/{panel,thinking}_train.jsonl` |
| Panel olympiad RL (100 steps) | ~3 h | the 14% → 29% hill-climbing curve |
| Thinking olympiad RL (100 steps, matched) | ~37 h | not yet run — the open follow-up |

The first five rows fit comfortably in a single afternoon of Tinker spend.
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
