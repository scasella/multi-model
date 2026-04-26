#!/usr/bin/env bash
# Smoke-test the <think> GSM8K env (stage 1 of the matched control) before
# launching the full 128-step run.
#
# Deliberately tiny: 2 steps, batch=4, group=4, max_tokens=2048.
# Checks template renders, grader fires, tag_valid tracks <think>, no
# exceptions, checkpoint saves.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
set -a; source .env; set +a

RUN_DIR="/tmp/tinker-examples/rl/think_gsm8k_smoke/$(date +%Y-%m-%d-%H-%M)"

python scripts/rl_think_gsm8k.py \
    model_name=Qwen/Qwen3-30B-A3B-Base \
    lora_rank=32 \
    seed=20260423 \
    n_train=64 \
    n_test=32 \
    group_size=4 \
    groups_per_batch=4 \
    learning_rate=5e-6 \
    temperature=1.0 \
    max_tokens=2048 \
    tag_coef=0.2 \
    max_steps=2 \
    save_every=1 \
    eval_every=0 \
    log_path="$RUN_DIR" \
    behavior_if_log_dir_exists=delete
