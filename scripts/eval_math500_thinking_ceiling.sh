#!/usr/bin/env bash
# Thinking-model ceiling: Qwen3-30B-A3B (hybrid) with enable_thinking=True
# on the same 50-problem MATH500 stratified slice + n=4 that all other
# eval points in reports/eval_math500_vibecheck/ use.
#
# This is the right reference for our <multipersonaDebate> panel
# scaffold — a production-grade post-trained Qwen that natively emits
# <think>. Orders of magnitude more post-training than any matched
# RL-from-Base <think> control we could run with our compute budget.
#
# Model choice: Tinker does not support sampling from -Thinking-2507,
# QwQ, or the R1-distills (BadRequestError). It DOES support the
# original hybrid Qwen3-30B-A3B, which is architecturally identical to
# our panel adapter's base (Qwen3-30B-A3B-Base) and toggles thinking
# via `enable_thinking=True` in apply_chat_template. See the script
# docstring for full rationale.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
set -a; source .env; set +a

mkdir -p logs
LOG="logs/eval_math500_thinking_ceiling.log"

echo "Logging to $LOG"
python scripts/eval_math500_thinking_ceiling.py tag=thinking_native 2>&1 | tee "$LOG"
