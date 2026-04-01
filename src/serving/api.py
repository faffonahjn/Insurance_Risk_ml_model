"""
FastAPI serving layer — Medical Insurance Risk Classifier v2.
Features: age, bmi, children, sex, smoker, region, bmi_age_interaction
Target: is_high_risk = charges > P75 ($16,658)
"""
import logging
import joblib
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List
import warnings
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = "configs/config.yaml"
_model = None


def _load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _init_model():
    global _model
    if _model is None:
        config = _load_config()
        model_path = Path(config["paths"]["model_dir"]) / config["training"]["model_filename"]
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _model = joblib.load(model_path)
        logger.info(f"Model loaded: {model_path}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_model()
    logger.info("API ready.")
    yield


app = FastAPI(
    title="Medical Insurance Risk Classifier",
    description="Predicts high-cost insurance patients (charges > P75). Features: age, bmi, children, sex, smoker, region.",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class InsuranceRecord(BaseModel):
    age: int = Field(..., ge=18, le=100)
    sex: str
    bmi: float = Field(..., ge=10.0, le=70.0)
    children: int = Field(..., ge=0, le=10)
    smoker: str
    region: str
    bmi_age_interaction: float

    @field_validator("sex")
    @classmethod
    def validate_sex(cls, v):
        if v not in {"male", "female"}:
            raise ValueError("sex must be 'male' or 'female'")
        return v

    @field_validator("smoker")
    @classmethod
    def validate_smoker(cls, v):
        if v not in {"yes", "no"}:
            raise ValueError("smoker must be 'yes' or 'no'")
        return v

    @field_validator("region")
    @classmethod
    def validate_region(cls, v):
        if v not in {"northeast", "northwest", "southeast", "southwest"}:
            raise ValueError("region must be one of: northeast, northwest, southeast, southwest")
        return v


class PredictionResponse(BaseModel):
    is_high_risk: int
    risk_probability: float
    risk_label: str
    latency_ms: float


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_records: int
    high_risk_count: int

DECISION_THRESHOLD = 0.40  # tuned for clinical recall

def _predict_single(record: InsuranceRecord) -> PredictionResponse:
    _init_model()
    t0 = time.perf_counter()
    df = pd.DataFrame([record.model_dump()])
    prob = float(_model.predict_proba(df)[0][1])
    label = int(prob >= DECISION_THRESHOLD)  # changed from 0.5 to 0.35 for better recall   
    return PredictionResponse(
        is_high_risk=label,
        risk_probability=round(prob, 4),
        risk_label="High Risk" if label else "Low Risk",
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
    )


@app.get("/health", tags=["System"])
def health():
    return {"status": "healthy", "model_loaded": _model is not None, "version": "2.0.0", "decision_threshold": DECISION_THRESHOLD}


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(record: InsuranceRecord):
    return _predict_single(record)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(records: List[InsuranceRecord]):
    if len(records) > 500:
        raise HTTPException(status_code=400, detail="Batch size limit: 500 records")
    predictions = [_predict_single(r) for r in records]
    return BatchPredictionResponse(
        predictions=predictions,
        total_records=len(predictions),
        high_risk_count=sum(p.is_high_risk for p in predictions),
    )
