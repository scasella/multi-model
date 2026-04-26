#!/usr/bin/env bash
# Thinking-arm RL on the olympiad union pool. MATCHED to rl_panel_olympiad.sh —
# same hyperparameters, different scaffold (Qwen3 native thinking chat
# template, starting from Qwen/Qwen3-30B-A3B published weights). Trains on
# the THINKING-arm variance band; scores on the shared held-out eval.
#
# Prereqs: same as rl_panel_olympiad.sh — variance-band filter must be run
# for BOTH arms before scripts/build_per_arm_splits.py.

set -euo pipefail
cd "$(dirname "$0")/.."

RUN_TAG="thinking_olympiad_$(date +%Y%m%d_%H%M)"
LOG="logs/${RUN_TAG}.log"
mkdir -p logs

python scripts/rl_thinking_olympiad.py \
    group_size=16 \
    groups_per_batch=8 \
    learning_rate=5e-6 \
    temperature=1.0 \
    max_tokens=12288 \
    max_steps=100 \
    save_every=10 \
    eval_every=10 \
    seed=20260424 \
    2>&1 | tee "$LOG"
