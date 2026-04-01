# System Architecture

## Overview

The Medical Insurance Risk Classifier is a production ML system that predicts whether an insurance patient is high-risk based on demographic and lifestyle features. It is deployed as a containerized REST API on Azure Container Apps.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
│  data/raw/  →  src/data/loader.py  →  src/features/engineer.py │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                       TRAINING LAYER                            │
│  pipelines/train_pipeline.py                                    │
│  src/models/trainer.py (XGBoost + sklearn Pipeline)            │
│  src/evaluation/metrics.py (ROC, PR, SHAP, CM)                 │
│  artifacts/models/risk_classifier_v1.pkl                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                       SERVING LAYER                             │
│  src/serving/api.py (FastAPI)                                   │
│  POST /predict         → single inference                       │
│  POST /predict/batch   → up to 500 records                     │
│  GET  /health          → liveness probe                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                     INFRASTRUCTURE LAYER                        │
│  docker/Dockerfile (multi-stage Python 3.11 slim)              │
│  docker/docker-compose.yml (local dev)                         │
│  Azure Container Registry → Azure Container Apps               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Feature Pipeline

| Stage | Module | Description |
|---|---|---|
| Load | `src/data/loader.py` | Read CSV, validate schema, split X/y |
| Encode | `src/features/engineer.py` | OneHotEncode categoricals, StandardScale numerics |
| Drop leakage | `configs/config.yaml` | `risk_score`, `charges`, `monthly_premium_est`, `insurance_tier` |
| Train | `src/models/trainer.py` | XGBoost with 5-fold CV |
| Evaluate | `src/evaluation/metrics.py` | AUC, AP, F1, confusion matrix, SHAP |
| Serve | `src/serving/api.py` | FastAPI with Pydantic validation |

---

## Data Flow

```
Raw CSV (24 cols)
    → Drop leakage (5 cols)
    → Drop target (1 col)
    → 18 feature columns remain
        ├── 5 categorical  → OneHotEncoder(drop='first')
        ├── 4 numeric      → StandardScaler
        └── 6 binary flags → passthrough
    → XGBoost classifier
    → is_high_risk (0 / 1) + probability
```

---

## Security

- Non-root Docker user (`appuser`)
- No secrets in image — injected via environment variables
- Pydantic schema validation on all API inputs
- CORS configured via FastAPI middleware

---

## Scalability

Azure Container Apps auto-scales between 1–3 replicas based on HTTP traffic.
Model is loaded once at startup (lifespan context) and shared across requests.
Batch endpoint handles up to 500 records per request.
