#!/usr/bin/env bash
# Extend the multipersona_gsm8k run from step 80 to step 160.
#
# The monotone acceleration across all seven 10-step windows (window means:
# 0.396 → 0.426 → 0.462 → 0.544 → 0.585 → 0.665 → 0.741 on correct)
# with no plateau at step 80 argues for more steps at the same
# hyperparameters. This launcher adds another 80 training steps.
#
# Resume mechanism is identical to the step-20-billing-halt resume:
#   log_path points at the original dir; behavior_if_log_dir_exists=resume
#   causes tinker_cookbook.rl.train.main() to read checkpoints.jsonl,
#   pick the last entry (step 80's `final` alias, batch=80), and reload
#   weights from that state_path.  max_steps is absolute, so 160 means
#   80 more steps.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RL_LOG="/tmp/tinker-examples/rl/multipersona_gsm8k/2026-04-23-08-08"

cd "$ROOT"
source .venv/bin/activate
set -a; source .env; set +a

python scripts/rl_multipersona_gsm8k.py \
    model_name=Qwen/Qwen3-30B-A3B-Base \
    lora_rank=32 \
    seed=20260423 \
    n_train=2048 \
    n_test=128 \
    group_size=8 \
    groups_per_batch=16 \
    learning_rate=5e-6 \
    temperature=1.0 \
    max_tokens=4096 \
    tag_coef=0.2 \
    max_steps=160 \
    save_every=10 \
    eval_every=0 \
    log_path="$RL_LOG" \
    behavior_if_log_dir_exists=resume
