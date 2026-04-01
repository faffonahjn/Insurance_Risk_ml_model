# 🏥 Medical Insurance Risk Classifier

> **End-to-end production ML pipeline** — predicts high-cost insurance patients from demographic and lifestyle features, deployed as a containerized REST API with a clinical Streamlit dashboard.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-orange)](https://xgboost.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45-red?logo=streamlit)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)](https://docker.com)
[![Azure](https://img.shields.io/badge/Azure-Container%20Apps-blue?logo=microsoftazure)](https://azure.microsoft.com)
[![Tests](https://img.shields.io/badge/Tests-21%20passing-brightgreen)](tests/)
[![AUC](https://img.shields.io/badge/Test%20AUC-0.899-blue)](artifacts/metrics/)

---

## 📌 Project Overview

This project demonstrates a **production-grade clinical ML system** built from raw insurance data through to a deployed REST API and interactive dashboard. It was developed with the architectural discipline of a senior ML engineer — including a full **data leakage forensic audit**, clinically motivated target engineering, threshold sensitivity analysis, and SHAP explainability.

**Clinical Problem:** Identify patients likely to incur high insurance costs (charges > 75th percentile / $16,658) from demographic and lifestyle features — enabling proactive risk stratification and care management.

---

## 🏗️ System Architecture

```
+---------------------------------------------------------+
|                      DATA LAYER                         |
|  data/raw/ -> loader.py -> engineer.py (sklearn Pipeline)|
+------------------------+--------------------------------+
                         |
+------------------------v--------------------------------+
|                    TRAINING LAYER                       |
|  train_pipeline.py -> XGBoost -> risk_classifier_v2.pkl |
|  Evaluation: ROC | PR | SHAP | Threshold Sensitivity    |
+------------------------+--------------------------------+
                         |
+------------------------v--------------------------------+
|                    SERVING LAYER                        |
|  FastAPI  ->  POST /predict  |  POST /predict/batch     |
|  Streamlit  ->  Single | Batch | EDA | Model Info       |
+------------------------+--------------------------------+
                         |
+------------------------v--------------------------------+
|                 INFRASTRUCTURE LAYER                    |
|  Docker (multi-stage) -> Compose -> Azure Container Apps|
+---------------------------------------------------------+
```

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| CV AUC (5-fold) | 0.879 +/- 0.036 |
| **Test AUC** | **0.899** |
| Average Precision | 0.872 |
| Sensitivity (Recall) | 76.1% |
| Specificity | 95.5% |
| Decision Threshold | **0.35** (clinical recall) |
| False Negatives | 16 / 268 (6.0%) |
| False Positives | 9 / 268 (3.4%) |

> **Threshold rationale:** Default 0.5 maximizes F1. Clinical deployment uses **0.35** — in risk stratification, missing a high-cost patient is more costly than a false alarm. Threshold sensitivity analysis confirmed 0.35 as the optimal precision-recall operating point.

---

## 🔬 Leakage Audit — The Hard Engineering Decision

The original dataset contained `is_high_risk` as a **100% deterministic rule** (`BMI >= 30 OR smoker = yes`), yielding a perfect AUC of 1.0. This was identified as structural dataset leakage, not model performance.

**Resolution:** Target was rebuilt as `charges > P75 ($16,658)` — the actuarial standard definition of a high-cost patient pool. This produces a genuinely noisy, clinically meaningful label with realistic AUC of 0.899.

**Columns dropped and why:**

| Column | Reason |
|---|---|
| `charges`, `monthly_premium_est`, `charges_per_child` | Direct target source — hard leakage |
| `risk_score`, `insurance_tier` | Derived from charges |
| `bmi_category` | Deterministic bin of `bmi` — structural leakage |
| `smoker_flag`, `sex_female`, `region_*` | Duplicate encodings |
| `age_group` | Binned duplicate of `age` |

---

## 🚀 Quickstart

### Prerequisites
- Python 3.11+
- Docker Desktop
- Anaconda / pip

### 1. Clone and Setup

```bash
git clone https://github.com/faffonahjn/insurance-risk-ml.git
cd insurance-risk-ml

conda create -n ml_env python=3.11 -y
conda activate ml_env
pip install -r requirements.txt
```

### 2. Train

```bash
python pipelines/train_pipeline.py
```

Expected output:
```
CV AUC: 0.8793 +/- 0.0357
Test AUC: 0.8988 | AP: 0.8723
Model saved -> artifacts/models/risk_classifier_v2.pkl
```

### 3. Batch Predict

```bash
python pipelines/predict_pipeline.py \
  --input data/raw/medical_insurance.csv \
  --output data/processed/predictions.csv
```

### 4. Serve Locally

```bash
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload
# API docs -> http://localhost:8000/docs
```

### 5. Launch Full Stack (Docker)

```bash
docker compose -f docker/docker-compose.yml up --build
```

| Service | URL |
|---|---|
| FastAPI | http://localhost:8000/docs |
| Streamlit Dashboard | http://localhost:8501 |

### 6. Run Tests

```bash
pytest tests/ -v
# 21 passed
```

---

## 🌐 API Reference

**Base URL:** `http://localhost:8000`

### GET /health
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "2.0.0",
  "decision_threshold": 0.35
}
```

### POST /predict

**Request:**
```json
{
  "age": 55,
  "sex": "male",
  "bmi": 38.0,
  "children": 1,
  "smoker": "yes",
  "region": "southeast",
  "bmi_age_interaction": 2090.0
}
```

**Response:**
```json
{
  "is_high_risk": 1,
  "risk_probability": 0.9874,
  "risk_label": "High Risk",
  "latency_ms": 3.21
}
```

### POST /predict/batch
Accepts up to 500 records. Returns predictions with aggregate high-risk count.

---

## 📁 Project Structure

```
ML_PROJECT/
├── artifacts/
│   ├── metrics/           evaluation_metrics.json
│   ├── models/            risk_classifier_v2.pkl
│   └── plots/             ROC | PR | confusion matrix | feature importance
├── configs/
│   └── config.yaml        YAML-driven pipeline configuration
├── data/
│   ├── raw/               medical_insurance.csv
│   └── processed/         predictions.csv
├── docker/
│   ├── Dockerfile         Multi-stage API image (Python 3.11-slim)
│   ├── Dockerfile.streamlit
│   └── docker-compose.yml
├── docs/
│   ├── architecture.md
│   └── api_reference.md
├── notebooks/
│   ├── exploratory/       01_eda.ipynb
│   ├── modeling/          02_modeling.ipynb
│   └── evaluation/        03_evaluation.ipynb
├── pipelines/
│   ├── train_pipeline.py
│   └── predict_pipeline.py
├── scripts/
│   ├── setup_azure.sh     Azure Container Apps provisioning
│   ├── retrain.sh         AUC-gated retrain + Docker rebuild
│   └── test_api.sh        API smoke tests
├── src/
│   ├── data/              loader.py
│   ├── features/          engineer.py
│   ├── models/            trainer.py
│   ├── evaluation/        metrics.py
│   ├── serving/           api.py (FastAPI)
│   └── utils/             logger.py | validators.py
├── streamlit_app/
│   └── app.py             4-tab clinical dashboard
├── tests/
│   ├── unit/              test_features | test_trainer | test_validators
│   └── integration/       test_api
├── Makefile
├── requirements.txt
└── README.md
```

---

## 🖥️ Streamlit Dashboard

The clinical dashboard provides four tabs:

| Tab | Description |
|---|---|
| 🔍 Single Prediction | Patient form with live risk score and probability gauge |
| 📋 Batch Prediction | CSV upload, bulk scoring, downloadable results |
| 📊 EDA Dashboard | Interactive charts — distributions, risk by feature, correlations |
| ℹ️ Model Info | Architecture, metrics, leakage audit, threshold rationale |

The sidebar shows live API health status and current decision threshold.

---

## ☁️ Azure Deployment

```bash
# Prerequisites: az CLI installed and logged in
az login
bash scripts/setup_azure.sh
```

Provisions:
- Azure Resource Group
- Azure Container Registry (ACR)
- Azure Container Apps environment
- Deployed API with external ingress (auto-scaling 1-3 replicas)

---

## 🧪 SHAP Explainability

The model is fully explainable via SHAP TreeExplainer. Top clinical drivers:

| Feature | Direction | Clinical Meaning |
|---|---|---|
| `smoker_yes` | Strong positive | Dominant cost driver — 4-6x SHAP impact |
| `bmi_age_interaction` | Moderate positive | Compounding obesity-age risk |
| `bmi` | Moderate positive | Higher BMI increases cost risk |
| `age` | Moderate positive | Older patients incur higher costs |
| `sex_male` | Near zero | Minimal predictive power |

**Model blind spot:** Non-smoking obese patients (mean BMI 31.7, mean age 41) with high charges driven by children and regional factors — 16 false negatives at threshold 0.35.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML | XGBoost 3.2 · scikit-learn 1.8 |
| Pipeline | sklearn Pipeline · ColumnTransformer |
| API | FastAPI · Uvicorn · Pydantic v2 |
| Dashboard | Streamlit · Matplotlib · Seaborn |
| Serialization | Joblib |
| Containerization | Docker multi-stage · Compose |
| Cloud | Azure Container Apps · Azure Container Registry |
| Testing | pytest · FastAPI TestClient |
| Config | YAML-driven · python-dotenv |

---

## 👤 Author

**Francis Affonah**
Clinical Data Scientist · ML Engineer · Registered Nurse

[![GitHub](https://img.shields.io/badge/GitHub-faffonahjn-black?logo=github)](https://github.com/faffonahjn)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Francis%20Affonah-blue?logo=linkedin)](https://linkedin.com/in/francis-affonah-23745a205/)

> *"So built we the wall... for the people had a mind to work."* — Nehemiah 4:6

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
