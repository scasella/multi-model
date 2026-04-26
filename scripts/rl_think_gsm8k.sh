#!/usr/bin/env bash
# Stage 1 of the matched <think> control: pure RL from Qwen3-30B-A3B-Base
# on GSM8K with a <think>...</think><answer>...</answer> scaffold.
#
# Hyperparameters are byte-for-byte identical to the panel GSM8K run
# (see scripts/rl_multipersona_gsm8k_extend_160.sh for the panel
# counterpart). Only the scaffold differs.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
set -a; source .env; set +a

RUN_DIR="/tmp/tinker-examples/rl/think_gsm8k/$(date +%Y-%m-%d-%H-%M)"

python scripts/rl_think_gsm8k.py \
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
    max_steps=128 \
    save_every=10 \
    eval_every=0 \
    log_path="$RUN_DIR" \
    behavior_if_log_dir_exists=delete
