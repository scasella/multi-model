#!/usr/bin/env bash
# Multi-Persona Panel of Experts — pure RL from Qwen3-30B-A3B-Base on GSM8K.
#
# Design (see envs/multipersona_gsm8k.py and the new preregistration):
#   • Pure RL from base. No SFT warmup, no prior checkpoint.
#   • Template-only setup shaping: the prompt casts the generator as a
#     "Multi-Persona Panel of Experts" that deliberates inside
#     <mutipersonaDebate> tags, then answers inside <answer> tags.
#   • Outcome-only reward: (1) numeric correctness on GSM8K gold,
#     (2) tag enforcement. No process reward on interior content.
#   • 30B-A3B MoE base (~3B active) — closer to DeepSeek's viability
#     threshold (≥32B dense) than the prior 8B experiment.
#
# Hyperparameters are a first pass; tune after a 10-step sanity run.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date +%Y-%m-%d-%H-%M)"
RL_LOG="/tmp/tinker-examples/rl/multipersona_gsm8k/${STAMP}"

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
    behavior_if_log_dir_exists=delete
