"""Integration tests — FastAPI endpoints (v2 schema)."""
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.serving.api import app

client = TestClient(app)

SAMPLE_HIGH_RISK = {
    "age": 55, "sex": "male", "bmi": 38.0, "children": 1,
    "smoker": "yes", "region": "southeast",
    "bmi_age_interaction": 2090.0,
}
SAMPLE_LOW_RISK = {
    "age": 25, "sex": "female", "bmi": 22.0, "children": 0,
    "smoker": "no", "region": "northwest",
    "bmi_age_interaction": 550.0,
}

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"

def test_predict_schema_valid():
    r = client.post("/predict", json=SAMPLE_HIGH_RISK)
    assert r.status_code == 200
    body = r.json()
    assert body["is_high_risk"] in [0, 1]
    assert 0.0 <= body["risk_probability"] <= 1.0
    assert body["risk_label"] in ["High Risk", "Low Risk"]
    assert body["latency_ms"] > 0

def test_high_risk_patient_classified():
    r = client.post("/predict", json=SAMPLE_HIGH_RISK)
    assert r.status_code == 200
    assert r.json()["is_high_risk"] == 1

def test_low_risk_patient_classified():
    r = client.post("/predict", json=SAMPLE_LOW_RISK)
    assert r.status_code == 200
    assert r.json()["is_high_risk"] == 0

def test_predict_batch():
    r = client.post("/predict/batch", json=[SAMPLE_HIGH_RISK, SAMPLE_LOW_RISK])
    assert r.status_code == 200
    body = r.json()
    assert body["total_records"] == 2
    assert len(body["predictions"]) == 2
    assert body["high_risk_count"] == 1

def test_invalid_region_rejected():
    r = client.post("/predict", json={**SAMPLE_HIGH_RISK, "region": "midwest"})
    assert r.status_code == 422

def test_age_out_of_range_rejected():
    r = client.post("/predict", json={**SAMPLE_HIGH_RISK, "age": 200})
    assert r.status_code == 422

def test_batch_size_limit():
    r = client.post("/predict/batch", json=[SAMPLE_HIGH_RISK] * 501)
    assert r.status_code == 400
