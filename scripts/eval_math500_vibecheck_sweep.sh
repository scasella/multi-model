#!/usr/bin/env bash
# MATH500 vibe-check: 50 stratified problems (10 per level × 5 levels),
# n=4 samples each, in both prompt variants, against the extended-run
# final checkpoint (batch=128). 200 samples per variant → narrower
# Wilson CI widths than the AIME sweep (~±3pp at 0.30 rate).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
set -a; source .env; set +a

: "${PANEL_GSM8K_CHECKPOINT_SAMPLER:?Set PANEL_GSM8K_CHECKPOINT_SAMPLER to your panel-GSM8K run's sampler_weights/final URI, e.g. tinker://<session>:train:0/sampler_weights/final}"
SAMPLER="${PANEL_GSM8K_CHECKPOINT_SAMPLER}"

for VARIANT in panel no_panel; do
    TAG="final_${VARIANT}"
    echo ""
    echo "================================================================"
    echo "MATH500 vibe-check  variant=${VARIANT}  tag=${TAG}"
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
