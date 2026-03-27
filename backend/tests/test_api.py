"""
backend/tests/test_api.py – Fix 9: Flask API Integration Tests
SkillGenome X

Run with: pytest tests/ -v
Uses Flask test client — no live server needed.
"""
import os
import sys
import json
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Minimal env to allow api.py to load without crashing
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")


@pytest.fixture(scope="module")
def client():
    """Provide a Flask test client for all api tests."""
    import api as app_module
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c


# ── Health endpoint ──────────────────────────────────────────────────────────

def test_health_returns_200(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "ok"
    assert "timestamp" in data


# ── National stats ───────────────────────────────────────────────────────────

def test_national_stats_keys(client):
    """"/api/national-distribution must return the 6 expected top-level keys."""
    res = client.get("/api/national-distribution")
    assert res.status_code == 200
    data = res.get_json()
    for key in ["national_skill_index", "total_profiles", "hidden_talents",
                "risk_zones", "skill_velocity", "states_covered"]:
        assert key in data, f"Missing key: {key}"


# ── Predict endpoint ─────────────────────────────────────────────────────────

VALID_PAYLOAD = {
    "signals": {
        "creation_output": 60, "learning_behavior": 70,
        "experience_consistency": 65, "economic_activity": 55,
        "innovation_problem_solving": 60, "collaboration_community": 50,
        "offline_capability": 45, "digital_presence": 65,
        "learning_hours": 15, "projects": 4, "github_repos": 0, "hackathons": 0,
    },
    "context": {
        "state": "Maharashtra", "area_type": "Urban",
        "opportunity_level": "High", "digital_access": "Regular",
    }
}


def test_predict_valid_input(client):
    """Valid prediction payload must return 200 with skill_score key."""
    res = client.post("/api/predict",
                      data=json.dumps(VALID_PAYLOAD),
                      content_type="application/json")
    assert res.status_code == 200
    data = res.get_json()
    assert "core" in data
    assert "score" in data["core"]
    score = data["core"]["score"]
    assert 0 <= score <= 100


def test_predict_invalid_state_returns_422(client):
    """Invalid state name must return 422 with validation error."""
    bad = {**VALID_PAYLOAD, "context": {**VALID_PAYLOAD["context"], "state": "Narnia"}}
    res = client.post("/api/predict",
                      data=json.dumps(bad),
                      content_type="application/json")
    assert res.status_code == 422
    data = res.get_json()
    assert "error" in data or "detail" in data


def test_predict_out_of_range_signal(client):
    """Signal value > 100 must return 422."""
    bad = {**VALID_PAYLOAD,
           "signals": {**VALID_PAYLOAD["signals"], "learning_behavior": 999}}
    res = client.post("/api/predict",
                      data=json.dumps(bad),
                      content_type="application/json")
    assert res.status_code == 422


def test_predict_response_has_anomaly_score(client):
    """Response must include anomaly_score and anomaly_confidence."""
    res = client.post("/api/predict",
                      data=json.dumps(VALID_PAYLOAD),
                      content_type="application/json")
    assert res.status_code == 200
    data = res.get_json()
    intel = data.get("intelligence", {})
    assert "anomaly_score" in intel
    assert "anomaly_confidence" in intel


# ── Model versions ───────────────────────────────────────────────────────────

def test_model_versions_endpoint(client):
    """/api/model-versions must return 200 with a list."""
    res = client.get("/api/model-versions")
    assert res.status_code == 200
    data = res.get_json()
    assert isinstance(data.get("versions"), list)


# ── AI status ────────────────────────────────────────────────────────────────

def test_ai_status_returns_active_flag(client):
    res = client.get("/api/ai-status")
    assert res.status_code == 200
    data = res.get_json()
    assert "active" in data
