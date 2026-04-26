#!/usr/bin/env bash
# Smoke-test the MATH train env before launching the full run.
#
# Goals (cheap, fast):
#   - Dataset loads (7 subjects, gold extractable, MATH-500 exclusion works)
#   - Round-robin batch composition has exactly-one-per-subject rotation
#   - Prompt renders, tokenizer works, tinker sampling client talks to the service
#   - math_verify grader fires on completions
#   - Checkpoint init from .../weights/final resolves the renderer
#
# Deliberately tiny: 2 steps, groups_per_batch=4 (below 7-subject rotation
# so you can visually verify subject rotation order), 4 per subject in train,
# 16-problem eval, max_tokens=2048.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
set -a; source .env; set +a

: "${PANEL_GSM8K_CHECKPOINT:?Set PANEL_GSM8K_CHECKPOINT to your GSM8K run's weights/final URI, e.g. tinker://<session>:train:0/weights/final}"

RUN_DIR="/tmp/tinker-examples/rl/multipersona_math_smoke/$(date +%Y-%m-%d-%H-%M)"

python scripts/rl_multipersona_math.py \
    model_name=Qwen/Qwen3-30B-A3B-Base \
    lora_rank=32 \
    load_checkpoint_path="${PANEL_GSM8K_CHECKPOINT}" \
    seed=20260423 \
    group_size=4 \
    groups_per_batch=4 \
    learning_rate=5e-6 \
    temperature=1.0 \
    max_tokens=2048 \
    tag_coef=0.2 \
    max_steps=2 \
    save_every=1 \
    eval_every=0 \
    n_eval=16 \
    n_train_per_subject=4 \
    virtual_train_len=32 \
    log_path="$RUN_DIR" \
    behavior_if_log_dir_exists=delete
