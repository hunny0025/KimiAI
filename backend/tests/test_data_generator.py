"""
backend/tests/test_data_generator.py – Fix 9: Unit Tests for Data Generator
SkillGenome X

Run with: pytest tests/ -v
No live DB connection required.
"""
import os
import sys
import pytest
import pandas as pd

# Ensure backend/ is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data_generator import generate, WHEEBOX_SKILL_MEAN, WHEEBOX_SKILL_STD

# Generate once for all tests (50K records takes ~30s — use 5K for speed)
@pytest.fixture(scope="module")
def df():
    return generate(n=5_000, seed=99)


# ── Column tests ────────────────────────────────────────────────────────────

REQUIRED_ORIGINAL_COLS = [
    "domain", "state", "area_type", "opportunity_level",
    "infrastructure_score", "digital_access",
    "creation_output", "github_repos", "projects",
    "learning_behavior", "learning_hours", "experience_consistency",
    "experience_years", "economic_activity", "innovation_problem_solving",
    "hackathons", "collaboration_community", "offline_capability",
    "digital_presence", "skill_score", "skill_history", "is_hidden_talent",
]

NEW_COLS = [
    "gender", "digital_access_level", "eshram_registered",
    "nsqf_level", "prior_experience_years", "community_learning_score",
    "proof_of_work_links", "hidden_talent_flag",
]


def test_output_columns(df):
    """All 30 columns (22 original + 8 new) must be present."""
    missing_orig = [c for c in REQUIRED_ORIGINAL_COLS if c not in df.columns]
    missing_new = [c for c in NEW_COLS if c not in df.columns]
    assert not missing_orig, f"Missing original columns: {missing_orig}"
    assert not missing_new, f"Missing new columns: {missing_new}"


def test_no_nulls_in_key_columns(df):
    """Key numeric + categorical columns must have no NaN."""
    key_cols = ["skill_score", "state", "area_type", "digital_access_level", "gender"]
    for col in key_cols:
        assert df[col].isna().sum() == 0, f"NaN values found in column: {col}"


# ── Skill score calibration ─────────────────────────────────────────────────

def test_skill_score_mean_wheebox(df):
    """skill_score mean must be within ±5 of Wheebox target (42.6)."""
    actual_mean = df["skill_score"].mean()
    tolerance = 5.0
    assert abs(actual_mean - WHEEBOX_SKILL_MEAN) <= tolerance, (
        f"skill_score mean {actual_mean:.2f} deviates more than {tolerance} "
        f"from Wheebox target {WHEEBOX_SKILL_MEAN}"
    )


def test_skill_score_std_wheebox(df):
    """skill_score std must be in a reasonable range (Wheebox target ~18; state-weighted
    sampling may compress it toward 12-22)."""
    actual_std = df["skill_score"].std()
    assert 10 <= actual_std <= 24, (
        f"skill_score std {actual_std:.2f} is implausibly narrow or wide."
    )


def test_skill_score_range(df):
    """All skill_score values must be in [0, 100]."""
    assert df["skill_score"].between(0, 100).all(), "skill_score out of [0, 100]"


# ── Rural/Urban split ───────────────────────────────────────────────────────

def test_rural_urban_split(df):
    """Rural share must be 58-72% (PLFS calibration target: ~65%). 
    Note: state-weighted sampling naturally pushes this to ~70%."""
    rural_pct = (df["area_type"] == "Rural").mean() * 100
    assert 58 <= rural_pct <= 72, (
        f"Rural share {rural_pct:.1f}% outside expected 58-72% range."
    )


# ── Digital access level ────────────────────────────────────────────────────

def test_digital_access_level_range(df):
    """digital_access_level must be in [0, 10]."""
    assert df["digital_access_level"].between(0, 10).all()


def test_rural_digital_access_lower(df):
    """Rural digital_access_level mean must be < Urban mean."""
    rural_mean = df[df["area_type"] == "Rural"]["digital_access_level"].mean()
    urban_mean = df[df["area_type"] == "Urban"]["digital_access_level"].mean()
    assert rural_mean < urban_mean, (
        f"Rural mean {rural_mean:.2f} is not less than Urban mean {urban_mean:.2f}"
    )


# ── Gender ──────────────────────────────────────────────────────────────────

def test_rural_female_participation(df):
    """Rural female participation must be 18-28% (MoSPI LFPR target: 23%)."""
    rural = df[df["area_type"] == "Rural"]
    female_pct = (rural["gender"] == "Female").mean() * 100
    assert 18 <= female_pct <= 28, (
        f"Rural female participation {female_pct:.1f}% outside expected 18-28% range."
    )


# ── New columns ──────────────────────────────────────────────────────────────

def test_nsqf_level_range(df):
    """nsqf_level must only contain values 1-5."""
    assert set(df["nsqf_level"].unique()).issubset({1, 2, 3, 4, 5})


def test_eshram_rural_rate(df):
    """eshram_registered rural rate should be 20-40% (target: 30%)."""
    rural = df[df["area_type"] == "Rural"]
    rate = rural["eshram_registered"].mean() * 100
    assert 20 <= rate <= 40, f"eshram rural rate {rate:.1f}% outside 20-40%"


def test_proof_of_work_links_range(df):
    """proof_of_work_links must be 0-3."""
    assert df["proof_of_work_links"].between(0, 3).all()


def test_hidden_talent_flag_logic(df):
    """hidden_talent_flag = skill_score > 65 AND digital_access_level < 4."""
    # All records with flag=True must satisfy both conditions
    flagged = df[df["hidden_talent_flag"]]
    assert (flagged["skill_score"] > 65).all()
    assert (flagged["digital_access_level"] < 4).all()
