"""Unit tests — input validators (v2 schema)."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.validators import validate_record

VALID_RECORD = {
    "age": 35, "sex": "male", "bmi": 28.5, "children": 2,
    "smoker": "no", "region": "southeast",
    "bmi_age_interaction": 997.5,
}

def test_valid_record_passes():
    assert validate_record(VALID_RECORD) == []

def test_missing_field_caught():
    bad = {k: v for k, v in VALID_RECORD.items() if k != "age"}
    assert any("age" in e for e in validate_record(bad))

def test_invalid_sex_caught():
    assert any("sex" in e for e in validate_record({**VALID_RECORD, "sex": "unknown"}))

def test_invalid_region_caught():
    assert any("region" in e for e in validate_record({**VALID_RECORD, "region": "midwest"}))

def test_age_out_of_range():
    assert any("age" in e for e in validate_record({**VALID_RECORD, "age": 150}))

def test_bmi_out_of_range():
    assert any("bmi" in e for e in validate_record({**VALID_RECORD, "bmi": 5.0}))
