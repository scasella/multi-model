#!/usr/bin/env bash
# Stage 2 of the matched <think> control: continue from the stage-1
# think-GSM8K final checkpoint and run RL on MATH train, round-robin
# category-balanced. Mirrors scripts/rl_multipersona_math.sh exactly,
# with the <think> scaffold substituted.
#
# Usage:
#   env THINK_GSM8K_STATE=tinker://<run-id>:train:0/weights/final \
#       scripts/rl_think_math.sh
#
# If THINK_GSM8K_STATE is not set, the script reads the final state_path
# from the most-recent /tmp/tinker-examples/rl/think_gsm8k/<RUN>/checkpoints.jsonl.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
set -a; source .env; set +a

# Resolve stage-1 final checkpoint.
if [[ -n "${THINK_GSM8K_STATE:-}" ]]; then
    INIT_PATH="${THINK_GSM8K_STATE}"
else
    LATEST_RUN_DIR=$(ls -1dt /tmp/tinker-examples/rl/think_gsm8k/*/ 2>/dev/null | head -1 || true)
    if [[ -z "${LATEST_RUN_DIR}" ]]; then
        echo "No stage-1 run dirs found under /tmp/tinker-examples/rl/think_gsm8k/" >&2
        exit 1
    fi
    INIT_PATH=$(python3 -c "
import json, sys
path = '${LATEST_RUN_DIR}checkpoints.jsonl'
last = None
with open(path) as f:
    for line in f:
        if line.strip():
            last = json.loads(line)
if last is None or last.get('name') != 'final':
    print('Stage-1 run did not complete (no final checkpoint):', path, file=sys.stderr)
    sys.exit(1)
print(last['state_path'])
")
    echo "Auto-resolved stage-1 final: ${INIT_PATH}" >&2
fi

RUN_DIR="/tmp/tinker-examples/rl/think_math/$(date +%Y-%m-%d-%H-%M)"

python scripts/rl_think_math.py \
    model_name=Qwen/Qwen3-30B-A3B-Base \
    lora_rank=32 \
    load_checkpoint_path="${INIT_PATH}" \
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
