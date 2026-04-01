# Medical Insurance Risk Classifier
**Production ML pipeline** — predicts high-risk insurance patients from demographic and lifestyle features.  
Deployed via FastAPI + Docker → Azure Container Apps.

---

## Architecture

```
data/raw/ → loader.py → engineer.py → trainer.py → risk_classifier_v1.pkl
                                                          ↓
                                               FastAPI (api.py)
                                                          ↓
                                            Docker → Azure Container Apps
```

## Stack
| Layer | Technology |
|---|---|
| Model | XGBoost (scikit-learn Pipeline) |
| API | FastAPI + Uvicorn |
| Containerization | Docker multi-stage build |
| Deployment | Azure Container Apps |
| Config | YAML-driven (configs/config.yaml) |

---

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Train
make train

# 3. Batch predict
make predict

# 4. Serve locally
make serve
# → http://localhost:8000/docs

# 5. Docker
make docker-build && make docker-up

# 6. Deploy to Azure
make azure-deploy
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| POST | `/predict` | Single-record inference |
| POST | `/predict/batch` | Batch inference (max 500) |

### Example Request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45, "sex": "male", "bmi": 34.5, "children": 2,
    "smoker": "yes", "region": "southeast",
    "age_group": "Adult (36-45)", "bmi_category": "Obese Class I",
    "bmi_age_interaction": 1552.5,
    "sex_female": 0, "smoker_flag": 1,
    "region_northeast": 0, "region_northwest": 0,
    "region_southeast": 1, "region_southwest": 0
  }'
```

### Example Response
```json
{
  "is_high_risk": 1,
  "risk_probability": 0.9874,
  "risk_label": "High Risk",
  "latency_ms": 3.21
}
```

---

## Model Performance
| Metric | Score |
|---|---|
| CV AUC (5-fold) | 1.000 ± 0.000 |
| Test AUC | 1.000 |
| Test Avg Precision | 1.000 |

> Note: Perfect AUC reflects that `is_high_risk` is a deterministic rule-based label derived from BMI + smoking + age in this dataset. In a real clinical setting, target labels from chart review or outcomes data will produce realistic AUC in the 0.75–0.90 range.

---

## Project Structure
```
ML_PROJECT/
├── artifacts/          # Models, metrics, plots
├── configs/            # config.yaml
├── data/               # raw → interim → processed
├── docker/             # Dockerfile, docker-compose.yml
├── pipelines/          # train_pipeline.py, predict_pipeline.py
├── src/
│   ├── data/           # loader.py
│   ├── features/       # engineer.py
│   ├── models/         # trainer.py
│   ├── evaluation/     # metrics.py
│   └── serving/        # api.py (FastAPI)
├── tests/              # unit + integration
├── Makefile
└── requirements.txt
```
