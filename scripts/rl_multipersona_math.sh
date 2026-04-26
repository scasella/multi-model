#!/usr/bin/env bash
# Full MATH train split RL run — continuation from the GSM8K-trained
# batch-128 checkpoint produced by rl_multipersona_gsm8k.sh.
#
# Prereqs:
#   - Run rl_multipersona_gsm8k.sh first (or have its weights/final URI on hand).
#   - Set PANEL_GSM8K_CHECKPOINT to that session's weights/final URI:
#       export PANEL_GSM8K_CHECKPOINT=tinker://<your-session>:train:0/weights/final
#
# Design:
#   - init: weights/final from your GSM8K run (fresh optimizer state)
#   - data: EleutherAI/hendrycks_math, 7 subjects, round-robin category-
#           balanced sampling (every batch = 16 groups, each rotation
#           cycle serves one from each of 7 subjects + 2 extras that
#           rotate across batches)
#   - in-training eval: 256-problem MATH-test slice, MATH-500 excluded
#   - budget: 128 steps, groups_per_batch=16, group_size=8 → 128 × 128 =
#           16,384 rollouts total
#   - rollout length: max_tokens=8192 (was 4096 on GSM8K — MATH chains
#           with the panel scaffold can easily exceed 4k)
#
# Save every 10 steps; eval every 20 steps.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
set -a; source .env; set +a

: "${PANEL_GSM8K_CHECKPOINT:?Set PANEL_GSM8K_CHECKPOINT to your GSM8K run's weights/final URI, e.g. tinker://<session>:train:0/weights/final}"

RUN_DIR="/tmp/tinker-examples/rl/multipersona_math/$(date +%Y-%m-%d-%H-%M)"

python scripts/rl_multipersona_math.py \
    model_name=Qwen/Qwen3-30B-A3B-Base \
    lora_rank=32 \
    load_checkpoint_path="${PANEL_GSM8K_CHECKPOINT}" \
    seed=20260423 \
    group_size=8 \
    groups_per_batch=16 \
    learning_rate=5e-6 \
    temperature=1.0 \
    max_tokens=8192 \
    tag_coef=0.2 \
    max_steps=128 \
    save_every=10 \
    eval_every=20 \
    n_eval=256 \
    log_path="$RUN_DIR" \
    behavior_if_log_dir_exists=delete
