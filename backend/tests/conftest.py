"""
backend/tests/conftest.py – Shared Pytest Fixtures
SkillGenome X

Provides reusable fixtures for the test suite.
DB-dependent tests are isolated via an in-memory SQLite database
(no PostgreSQL connection required).
"""

import json
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.models import Base, TalentProfile, SkillScore


# ── In-Memory SQLite Engine (no Postgres needed) ──────────────────────────────

@pytest.fixture(scope="session")
def engine():
    """Session-scoped in-memory SQLite engine. Created once per test run."""
    _engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(_engine)
    yield _engine
    _engine.dispose()


@pytest.fixture
def db(engine):
    """
    Function-scoped database session. Each test gets its own transaction that
    is rolled back at teardown, keeping tests fully isolated.
    """
    connection = engine.connect()
    transaction = connection.begin()
    Session = sessionmaker(bind=connection)
    session = Session()

    yield session

    session.close()
    transaction.rollback()
    connection.close()


# ── Sample PLFS-style DataFrames ───────────────────────────────────────────────

@pytest.fixture
def valid_plfs_df():
    """A minimal, fully-valid PLFS DataFrame with 4 rows."""
    return pd.DataFrame({
        "sector":                 [1, 2, 1, 2],
        "state_code":             [27, 29, 33, 19],
        "nic_code":               ["01111", "62013", "85110", "86100"],
        "usual_activity_status":  [21,   11,   31,    31],
        "education_level":        [10,   13,   12,    12],
        "age":                    [45,   28,   35,    33],
        "weekly_wage":            [1200, 8500, 4500,  4000],
        "land_possessed":         [1.5,  0,    0,     0],
    })


@pytest.fixture
def plfs_df_missing_cols(valid_plfs_df):
    """Returns a DataFrame missing the 'state_code' column."""
    return valid_plfs_df.drop(columns=["state_code"])


@pytest.fixture
def plfs_df_unknown_state():
    """Contains an unmapped state_code (99)."""
    return pd.DataFrame({
        "sector":                 [1],
        "state_code":             [99],
        "nic_code":               ["01111"],
        "usual_activity_status":  [21],
        "education_level":        [10],
        "age":                    [45],
    })


# ── Seed Helpers ───────────────────────────────────────────────────────────────

def make_profile(**kwargs) -> TalentProfile:
    """Build a minimal TalentProfile ORM object for seeding tests."""
    defaults = dict(
        state="Maharashtra", domain="Technology", area_type="Urban",
        digital_access="Regular", opportunity_level="Moderate",
        creation_output=60, learning_behavior=65, experience_consistency=55,
        economic_activity=70, innovation_problem_solving=60,
        collaboration_community=50, offline_capability=40,
        digital_presence=75, learning_hours=20, projects=5,
        skill_history=json.dumps([50] * 24),
    )
    defaults.update(kwargs)
    return TalentProfile(**defaults)


def make_skill_score(profile: TalentProfile, score: float = 72.5) -> SkillScore:
    """Build a minimal SkillScore linked to a TalentProfile."""
    return SkillScore(
        talent_profile_id=profile.id,
        score=score,
        level="Advanced" if score > 60 else "Intermediate",
        confidence=85.0,
        is_anomaly=False,
        is_hidden_talent=score > 70 and profile.opportunity_level == "Low",
        migration_risk="Low",
    )
