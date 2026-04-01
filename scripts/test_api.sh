#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# test_api.sh
# Smoke-tests the running FastAPI service
# Usage: bash scripts/test_api.sh [BASE_URL]
# Default: http://localhost:8000
# ─────────────────────────────────────────────────────────────────────────────
BASE_URL="${1:-http://localhost:8000}"

echo "==> Testing API at: $BASE_URL"

# Health check
echo ""
echo "--- /health ---"
curl -sf "$BASE_URL/health" | python3 -m json.tool

# Single prediction
echo ""
echo "--- /predict (High Risk expected) ---"
curl -sf -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55, "sex": "male", "bmi": 39.5, "children": 1,
    "smoker": "yes", "region": "southeast",
    "age_group": "Middle-Aged (46-55)", "bmi_category": "Obese Class II",
    "bmi_age_interaction": 2172.5,
    "sex_female": 0, "smoker_flag": 1,
    "region_northeast": 0, "region_northwest": 0,
    "region_southeast": 1, "region_southwest": 0
  }' | python3 -m json.tool

# Low risk prediction
echo ""
echo "--- /predict (Low Risk expected) ---"
curl -sf -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 25, "sex": "female", "bmi": 22.0, "children": 0,
    "smoker": "no", "region": "northwest",
    "age_group": "Young Adult (18-25)", "bmi_category": "Normal Weight",
    "bmi_age_interaction": 550.0,
    "sex_female": 1, "smoker_flag": 0,
    "region_northeast": 0, "region_northwest": 1,
    "region_southeast": 0, "region_southwest": 0
  }' | python3 -m json.tool

echo ""
echo "==> Smoke tests complete."
