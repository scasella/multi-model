#!/usr/bin/env bash
# Sequentially evaluate the three most-promising training checkpoints
# (step 60 / 70 / 80) on a fixed 500-example GSM8K test slice at the
# training temperature (1.0) and max_tokens (4096). Produces
#   reports/eval_gsm8k/<tag>/summary.json
#   reports/eval_gsm8k/<tag>/rollouts.jsonl
# for each tag.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
set -a; source .env; set +a

SESS="f32e0238-cffd-5198-a497-c76f04b4cf4d"

for STEP in 000060 000070 000080; do
    TAG="step${STEP#00000}"           # step60 / step70 / step80
    TAG="${TAG#0}"
    echo ""
    echo "================================================================"
    echo "Evaluating ${TAG} (sampler_weights/${STEP})"
    echo "================================================================"
    python scripts/eval_gsm8k.py \
        sampler_path="tinker://${SESS}:train:0/sampler_weights/${STEP}" \
        tag="${TAG}" \
        n_test=500 \
        batch_size=16 \
        max_tokens=4096 \
        temperature=1.0 \
        seed=20260423
done
