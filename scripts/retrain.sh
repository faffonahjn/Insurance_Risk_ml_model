#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# retrain.sh
# Retrains model, validates AUC threshold, pushes new image if pass
# Usage: bash scripts/retrain.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

MIN_AUC=0.80
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "==> [${TIMESTAMP}] Starting retraining..."

# Step 1: Train
python pipelines/train_pipeline.py --config configs/config.yaml

# Step 2: Extract AUC from metrics
AUC=$(python3 -c "
import json
with open('artifacts/metrics/evaluation_metrics.json') as f:
    m = json.load(f)
print(m['test_roc_auc'])
")

echo "==> Test AUC: $AUC | Minimum required: $MIN_AUC"

# Step 3: Gate on AUC threshold
PASS=$(python3 -c "print('yes' if float('$AUC') >= float('$MIN_AUC') else 'no')")

if [ "$PASS" = "yes" ]; then
    echo "==> AUC gate PASSED. Rebuilding Docker image..."
    docker build -f docker/Dockerfile -t insurance-risk-api:latest .
    echo "==> New model image built. Run 'make azure-deploy' to push."
else
    echo "==> AUC gate FAILED ($AUC < $MIN_AUC). Aborting deployment."
    exit 1
fi
