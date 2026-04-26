#!/usr/bin/env bash
# Resume the multipersona_gsm8k run from step 20 after the 402 billing halt.
#
# This re-targets the ORIGINAL log directory
#   /tmp/tinker-examples/rl/multipersona_gsm8k/2026-04-23-08-08
# and sets behavior_if_log_dir_exists=resume so tinker_cookbook.rl.train.main()
# reads checkpoints.jsonl, finds the step-20 record, and calls
# initialize_training_client_from_checkpoint(...state_path=.../weights/000020...),
# setting start_batch=20.
#
# With max_steps=80 (absolute ceiling, unchanged from the original run), this
# will train steps 21..80 — 60 new steps — saving at 30, 40, 50, 60, 70, 80.
#
# Dataset determinism: same seed=20260423 and same dataset builder → the
# per-step batch indexing resumes at batch 20 and sees the same examples it
# would have seen on an uninterrupted run. No data skipped or re-used.

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
    max_steps=80 \
    save_every=10 \
    eval_every=0 \
    log_path="$RL_LOG" \
    behavior_if_log_dir_exists=resume
