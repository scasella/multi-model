#!/usr/bin/env bash
# Panel RL on the olympiad union pool (panel-arm variance band).
# Pair this with scripts/rl_thinking_olympiad.sh — matched hyperparameters.
# Each arm trains on its own variance band; both arms score on the same
# shared held-out eval (data/olympiad_pool/heldout_eval.jsonl).
#
# Prereqs (run once, in order):
#   1) python scripts/build_olympiad_pool.py
#   2) python scripts/filter_variance_band.py --arm panel    --tag panel_g8
#   3) python scripts/filter_variance_band.py --arm thinking --tag thinking_g8
#   4) python scripts/build_per_arm_splits.py \
#         --panel    reports/variance_band/panel_g8/per_problem.jsonl \
#         --thinking reports/variance_band/thinking_g8/per_problem.jsonl
#   5) this script.

set -euo pipefail
cd "$(dirname "$0")/.."

RUN_TAG="panel_olympiad_$(date +%Y%m%d_%H%M)"
LOG="logs/${RUN_TAG}.log"
mkdir -p logs

python scripts/rl_panel_olympiad.py \
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
