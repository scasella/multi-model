#!/usr/bin/env bash
# MATH500 vibe-check against your post-MATH-training panel checkpoint.
#
# Same stratified-50 slice (10 per level × 5 levels), n=4 samples, in
# both prompt variants, so the result is directly comparable to
# reports/eval_math500_vibecheck/final_{panel,no_panel}/ from the
# GSM8K-only checkpoint.
#
# Prereqs:
#   export PANEL_MATH_CHECKPOINT_SAMPLER=tinker://<your-session>:train:0/sampler_weights/final
#
# Output tags:
#   math_final_panel     — panel scaffold (training prompt)
#   math_final_no_panel  — no-scaffold prompt

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
set -a; source .env; set +a

: "${PANEL_MATH_CHECKPOINT_SAMPLER:?Set PANEL_MATH_CHECKPOINT_SAMPLER to your panel-MATH run's sampler_weights/final URI}"
SAMPLER="${PANEL_MATH_CHECKPOINT_SAMPLER}"

for VARIANT in panel no_panel; do
    TAG="math_final_${VARIANT}"
    echo ""
    echo "================================================================"
    echo "MATH500 vibe-check  variant=${VARIANT}  tag=${TAG}"
    echo "sampler=${SAMPLER}"
    echo "================================================================"
    python scripts/eval_math500_vibecheck.py \
        sampler_path="${SAMPLER}" \
        variant="${VARIANT}" \
        tag="${TAG}" \
        n_per_level=10 \
        n_samples=4 \
        max_tokens=4096 \
        temperature=1.0 \
        concurrent_problems=8
done
