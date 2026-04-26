#!/usr/bin/env bash
# Sequentially run the AIME vibe-check eval in both prompt variants
# against the extended-run final checkpoint (batch=128).
#
# Each variant: 20 problems (10 AIME24 + 10 AIME25), n=8 samples,
# T=1.0, max_tokens=8192. Produces:
#   reports/eval_aime_vibecheck/<tag>/summary.json
#   reports/eval_aime_vibecheck/<tag>/rollouts.jsonl

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
set -a; source .env; set +a

: "${PANEL_GSM8K_CHECKPOINT_SAMPLER:?Set PANEL_GSM8K_CHECKPOINT_SAMPLER to your panel-GSM8K run's sampler_weights/final URI}"
SAMPLER="${PANEL_GSM8K_CHECKPOINT_SAMPLER}"

for VARIANT in panel no_panel; do
    TAG="final_${VARIANT}"
    echo ""
    echo "================================================================"
    echo "AIME vibe-check  variant=${VARIANT}  tag=${TAG}"
    echo "================================================================"
    python scripts/eval_aime_vibecheck.py \
        sampler_path="${SAMPLER}" \
        variant="${VARIANT}" \
        tag="${TAG}" \
        n_per_year=10 \
        n_samples=8 \
        max_tokens=8192 \
        temperature=1.0 \
        concurrent_problems=5
done
