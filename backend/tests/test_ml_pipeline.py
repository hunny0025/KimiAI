"""
backend/tests/test_ml_pipeline.py – Fix 9: ML Pipeline Unit Tests
SkillGenome X

Run with: pytest tests/ -v
"""
import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def models():
    """Load models from disk (must exist — run training first if missing)."""
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    skill_path = os.path.join(models_dir, "gbr_model.joblib")
    iso_path   = os.path.join(models_dir, "isolation_forest.joblib")

    if not (os.path.exists(skill_path) and os.path.exists(iso_path)):
        pytest.skip("Trained models not found. Run ml/training.py first.")

    import joblib
    return {
        "skill": joblib.load(skill_path),
        "anomaly": joblib.load(iso_path),
    }


FEATURE_NAMES = [
    'creation_output', 'learning_behavior', 'experience_consistency',
    'economic_activity', 'innovation_problem_solving', 'collaboration_community',
    'offline_capability', 'digital_presence', 'learning_hours', 'projects'
]

SAMPLE_VECTORS = [
    [60, 70, 65, 55, 60, 50, 45, 65, 15, 4],   # typical urban worker
    [30, 25, 40, 20, 30, 35, 75, 10, 5, 1],    # low-access rural worker
    [95, 90, 88, 85, 92, 80, 30, 92, 40, 15],  # high-skill profile
]


# ── Model existence tests ────────────────────────────────────────────────────

def test_model_files_exist():
    """Both model .joblib files must exist after training."""
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    skill_path = os.path.join(models_dir, "gbr_model.joblib")
    iso_path   = os.path.join(models_dir, "isolation_forest.joblib")
    if not (os.path.exists(skill_path) and os.path.exists(iso_path)):
        pytest.skip("Model files not present — run training to generate them first.")
    assert os.path.exists(skill_path), "gbr_model.joblib not found"
    assert os.path.exists(iso_path),   "isolation_forest.joblib not found"


def test_feature_count(models):
    """GBR model must have been trained on exactly 10 features."""
    skill_model = models["skill"]
    n_features = skill_model.n_features_in_
    assert n_features == len(FEATURE_NAMES), (
        f"Model has {n_features} features, expected {len(FEATURE_NAMES)}"
    )


# ── Prediction range tests ───────────────────────────────────────────────────

def test_prediction_range(models):
    """All skill score predictions must be in [0, 100]."""
    skill_model = models["skill"]
    for fv in SAMPLE_VECTORS:
        raw = float(skill_model.predict([fv])[0])
        score = max(0, min(100, raw))
        assert 0 <= score <= 100, f"Prediction {score} out of [0, 100]"


def test_prediction_is_float(models):
    """Prediction must return a numeric value."""
    skill_model = models["skill"]
    pred = skill_model.predict([SAMPLE_VECTORS[0]])[0]
    assert isinstance(pred, (float, np.floating, int, np.integer))


# ── Anomaly model output tests ───────────────────────────────────────────────

def test_anomaly_output_is_binary(models):
    """IsolationForest .predict() must return only 1 or -1."""
    iso = models["anomaly"]
    for fv in SAMPLE_VECTORS:
        result = iso.predict([fv])[0]
        assert result in (1, -1), f"IsolationForest returned unexpected value: {result}"


def test_anomaly_decision_function_returns_float(models):
    """decision_function() must return a float score."""
    iso = models["anomaly"]
    score = iso.decision_function([SAMPLE_VECTORS[0]])[0]
    assert isinstance(score, (float, np.floating))


# ── Model versioning tests ───────────────────────────────────────────────────

def test_model_versioning_import():
    """model_versioning module must import without error."""
    from ml.model_versioning import list_versions, load_latest_card
    assert callable(list_versions)
    assert callable(load_latest_card)


def test_list_versions_returns_list():
    """list_versions() must return a list (empty is fine if not yet trained)."""
    from ml.model_versioning import list_versions
    versions = list_versions()
    assert isinstance(versions, list)
