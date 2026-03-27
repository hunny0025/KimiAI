"""
backend/tests/test_plfs_validation.py – Unit Tests: PLFS Schema Validation & Mapping
SkillGenome X

All tests run without a live Postgres connection:
  - Stages 1-3 (validate, map, impute) use only in-memory Pandas DataFrames.
  - Stage 4 (load) is tested against the SQLite fixture from conftest.py.
"""

import pytest
import pandas as pd

from ingestion.plfs_pipeline import PLFSPipeline, PLFSValidationError
from tests.conftest import make_profile, make_skill_score


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def pipeline():
    return PLFSPipeline()


@pytest.fixture
def valid_data():
    return pd.DataFrame({
        "sector":                 [1, 2],
        "state_code":             [27, 29],
        "nic_code":               ["01111", "62013"],
        "usual_activity_status":  [21, 11],
        "education_level":        [10, 13],
        "age":                    [45, 28],
    })


# ════════════════════════════════════════════════════════════════════════════
# Stage 1 – Validate
# ════════════════════════════════════════════════════════════════════════════

def test_valid_schema_passes(pipeline, valid_data):
    """A complete, valid PLFS DataFrame must pass validation without raising."""
    pipeline.stage_1_validate(valid_data)  # no exception = pass


def test_missing_required_col_raises(pipeline, valid_data):
    """Removing a single required column must raise PLFSValidationError naming it."""
    bad = valid_data.drop(columns=["state_code"])
    with pytest.raises(PLFSValidationError) as exc_info:
        pipeline.stage_1_validate(bad)
    assert "state_code" in str(exc_info.value)


def test_multiple_missing_cols_all_listed(pipeline, valid_data):
    """Validation error must list EVERY missing column, not just the first."""
    bad = valid_data.drop(columns=["state_code", "education_level"])
    with pytest.raises(PLFSValidationError) as exc_info:
        pipeline.stage_1_validate(bad)
    msg = str(exc_info.value)
    assert "state_code" in msg
    assert "education_level" in msg


def test_empty_dataframe_raises(pipeline):
    """An empty DataFrame must raise PLFSValidationError."""
    with pytest.raises(PLFSValidationError):
        pipeline.stage_1_validate(pd.DataFrame())


def test_extra_columns_are_accepted(pipeline, valid_data):
    """PLFS exports may carry extra columns; they must not cause validation to fail."""
    df_extra = valid_data.copy()
    df_extra["some_extra_column"] = 42
    pipeline.stage_1_validate(df_extra)  # no exception = pass


# ════════════════════════════════════════════════════════════════════════════
# Stage 2 – Map
# ════════════════════════════════════════════════════════════════════════════

def test_state_code_27_to_maharashtra(pipeline, valid_data):
    mapped = pipeline.stage_2_map(valid_data)
    assert mapped.at[0, "state"] == "Maharashtra"


def test_state_code_29_to_karnataka(pipeline, valid_data):
    mapped = pipeline.stage_2_map(valid_data)
    assert mapped.at[1, "state"] == "Karnataka"


def test_sector_1_to_rural(pipeline, valid_data):
    mapped = pipeline.stage_2_map(valid_data)
    assert mapped.at[0, "area_type"] == "Rural"


def test_sector_2_to_urban(pipeline, valid_data):
    mapped = pipeline.stage_2_map(valid_data)
    assert mapped.at[1, "area_type"] == "Urban"


def test_nic_01_to_agriculture(pipeline, valid_data):
    mapped = pipeline.stage_2_map(valid_data)
    assert mapped.at[0, "domain"] == "Agriculture"


def test_nic_62_to_technology(pipeline, valid_data):
    mapped = pipeline.stage_2_map(valid_data)
    assert mapped.at[1, "domain"] == "Technology"


def test_education_10_learning_score_70(pipeline, valid_data):
    mapped = pipeline.stage_2_map(valid_data)
    assert mapped.at[0, "learning_behavior"] == 70


def test_education_13_learning_score_95(pipeline, valid_data):
    mapped = pipeline.stage_2_map(valid_data)
    assert mapped.at[1, "learning_behavior"] == 95


def test_unknown_state_code_row_skipped(pipeline):
    """Rows with unmapped state_code 99 must be excluded from the output."""
    bad_state = pd.DataFrame({
        "sector": [1], "state_code": [99], "nic_code": ["01111"],
        "usual_activity_status": [21], "education_level": [10], "age": [45],
    })
    mapped = pipeline.stage_2_map(bad_state)
    assert len(mapped) == 0


def test_unmapped_nic_defaults_to_general(pipeline):
    """NIC codes with no match must produce domain='General'."""
    df = pd.DataFrame({
        "sector": [1], "state_code": [27], "nic_code": ["99999"],
        "usual_activity_status": [21], "education_level": [10], "age": [45],
    })
    mapped = pipeline.stage_2_map(df)
    assert mapped.at[0, "domain"] == "General"


def test_weekly_wage_maps_to_economic_activity(pipeline):
    """weekly_wage should be normalised into economic_activity (capped at 100)."""
    df = pd.DataFrame({
        "sector": [2], "state_code": [27], "nic_code": ["62013"],
        "usual_activity_status": [11], "education_level": [13], "age": [30],
        "weekly_wage": [5000],
    })
    mapped = pipeline.stage_2_map(df)
    assert 0 <= mapped.at[0, "economic_activity"] <= 100


# ════════════════════════════════════════════════════════════════════════════
# Stage 3 – Impute
# ════════════════════════════════════════════════════════════════════════════

def test_impute_fills_behavioral_cols(pipeline, valid_data):
    """After imputation all behavioural int cols should be non-null."""
    mapped = pipeline.stage_2_map(valid_data)
    final = pipeline.stage_3_impute(mapped)
    for col in ["creation_output", "experience_consistency", "offline_capability"]:
        assert final[col].notna().all(), f"{col} still has NaN after imputation"


def test_impute_default_creation_output_is_50(pipeline, valid_data):
    """creation_output not provided by PLFS should default to 50."""
    mapped = pipeline.stage_2_map(valid_data)
    final = pipeline.stage_3_impute(mapped)
    assert (final["creation_output"] == 50).all()
