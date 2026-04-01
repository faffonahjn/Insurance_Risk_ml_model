"""Input validators — v2 schema (7 clean features)."""
from typing import Any, Dict, List

VALID_SEX = {"male", "female"}
VALID_SMOKER = {"yes", "no"}
VALID_REGIONS = {"northeast", "northwest", "southeast", "southwest"}
REQUIRED_FIELDS = ["age", "sex", "bmi", "children", "smoker", "region", "bmi_age_interaction"]


def validate_record(record: Dict[str, Any]) -> List[str]:
    errors = []
    for field in REQUIRED_FIELDS:
        if field not in record:
            errors.append(f"Missing required field: {field}")
    if errors:
        return errors

    if not isinstance(record["age"], int) or not (18 <= record["age"] <= 100):
        errors.append("age must be an integer between 18 and 100")
    if not isinstance(record["bmi"], (int, float)) or not (10.0 <= record["bmi"] <= 70.0):
        errors.append("bmi must be a float between 10.0 and 70.0")
    if not isinstance(record["children"], int) or not (0 <= record["children"] <= 10):
        errors.append("children must be an integer between 0 and 10")
    if record["sex"] not in VALID_SEX:
        errors.append(f"sex must be one of {VALID_SEX}")
    if record["smoker"] not in VALID_SMOKER:
        errors.append(f"smoker must be one of {VALID_SMOKER}")
    if record["region"] not in VALID_REGIONS:
        errors.append(f"region must be one of {VALID_REGIONS}")
    return errors
