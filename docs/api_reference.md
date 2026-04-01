# API Reference

Base URL: `http://localhost:8000` (local) | `https://<app>.azurecontainerapps.io` (Azure)

Interactive docs: `GET /docs` (Swagger UI)

---

## Endpoints

### GET /health
Liveness check. Returns model load status.

**Response 200:**
```json
{ "status": "healthy", "model_loaded": true }
```

---

### POST /predict
Single-record inference.

**Request Body:**
| Field | Type | Constraints | Example |
|---|---|---|---|
| age | int | 18–100 | 45 |
| sex | str | male / female | "male" |
| bmi | float | 10.0–70.0 | 34.5 |
| children | int | 0–10 | 2 |
| smoker | str | yes / no | "yes" |
| region | str | northeast/northwest/southeast/southwest | "southeast" |
| age_group | str | see valid values | "Adult (36-45)" |
| bmi_category | str | see valid values | "Obese Class I" |
| bmi_age_interaction | float | age × bmi | 1552.5 |
| sex_female | int | 0 / 1 | 0 |
| smoker_flag | int | 0 / 1 | 1 |
| region_northeast | int | 0 / 1 | 0 |
| region_northwest | int | 0 / 1 | 0 |
| region_southeast | int | 0 / 1 | 1 |
| region_southwest | int | 0 / 1 | 0 |

**Valid age_group values:**
`Young Adult (18-25)`, `Adult (26-35)`, `Adult (36-45)`, `Middle-Aged (46-55)`, `Senior (56+)`

**Valid bmi_category values:**
`Underweight`, `Normal Weight`, `Overweight`, `Obese Class I`, `Obese Class II`, `Obese Class III`

**Response 200:**
```json
{
  "is_high_risk": 1,
  "risk_probability": 0.9874,
  "risk_label": "High Risk",
  "latency_ms": 3.21
}
```

**Error 422:** Invalid input schema (Pydantic validation error)
**Error 503:** Model not loaded

---

### POST /predict/batch
Batch inference — up to 500 records.

**Request Body:** Array of InsuranceRecord objects (same schema as /predict)

**Response 200:**
```json
{
  "predictions": [...],
  "total_records": 50,
  "high_risk_count": 32
}
```

**Error 400:** Batch size exceeds 500

---

## Example cURL

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

---

## Python Client Example

```python
import httpx

record = {
    "age": 45, "sex": "male", "bmi": 34.5, "children": 2,
    "smoker": "yes", "region": "southeast",
    "age_group": "Adult (36-45)", "bmi_category": "Obese Class I",
    "bmi_age_interaction": 1552.5,
    "sex_female": 0, "smoker_flag": 1,
    "region_northeast": 0, "region_northwest": 0,
    "region_southeast": 1, "region_southwest": 0,
}

response = httpx.post("http://localhost:8000/predict", json=record)
print(response.json())
# {"is_high_risk": 1, "risk_probability": 0.9874, "risk_label": "High Risk", "latency_ms": 3.21}
```
