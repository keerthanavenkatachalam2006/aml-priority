"""
Backend API tests.
Run with: python -m pytest aml-priority/tests/backend/test_api.py -v
"""
import sys, pytest
from pathlib import Path
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "ml" / "src"))

from app.main import app
from app.db.database import engine, SessionLocal, Base
from app.models.orm_models import Alert, Case, AppSettings

# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def client():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        if not db.query(AppSettings).first():
            db.add(AppSettings())
            db.commit()
    finally:
        db.close()
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def seeded_alert(client):
    """Insert one alert and return its id."""
    db = SessionLocal()
    try:
        a = Alert(
            tx_id=999999, sender_account_id=1, receiver_account_id=2,
            tx_amount=500.0, timestamp=10, risk_score=0.85,
            priority="HIGH", is_fraud_label=True, alert_type="cycle",
        )
        db.add(a); db.commit(); db.refresh(a)
        return a.id
    finally:
        db.close()


# ── Health ────────────────────────────────────────────────────────────────────
def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ── Dashboard ─────────────────────────────────────────────────────────────────
def test_dashboard_summary(client):
    r = client.get("/api/dashboard/summary")
    assert r.status_code == 200
    d = r.json()
    assert "total_alerts" in d
    assert "high_priority_count" in d


def test_dashboard_timeseries(client):
    r = client.get("/api/dashboard/timeseries")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


# ── Alerts ────────────────────────────────────────────────────────────────────
def test_list_alerts(client, seeded_alert):
    r = client.get("/api/alerts?page=1&size=10")
    assert r.status_code == 200
    d = r.json()
    assert "total" in d and "items" in d


def test_list_alerts_priority_filter(client):
    r = client.get("/api/alerts?priority=HIGH&size=10")
    assert r.status_code == 200
    for item in r.json()["items"]:
        assert item["priority"] == "HIGH"


def test_list_alerts_pagination(client):
    r1 = client.get("/api/alerts?page=1&size=1")
    r2 = client.get("/api/alerts?page=2&size=1")
    assert r1.status_code == 200
    assert r2.status_code == 200
    if r1.json()["total"] >= 2:
        assert r1.json()["items"][0]["id"] != r2.json()["items"][0]["id"]


def test_get_alert_by_id(client, seeded_alert):
    r = client.get(f"/api/alerts/{seeded_alert}")
    assert r.status_code == 200
    assert r.json()["id"] == seeded_alert


def test_get_alert_not_found(client):
    r = client.get("/api/alerts/99999999")
    assert r.status_code == 404


# ── Cases ─────────────────────────────────────────────────────────────────────
def test_create_case_from_alert(client, seeded_alert):
    r = client.post(f"/api/cases/from-alert/{seeded_alert}")
    assert r.status_code == 200
    assert r.json()["alert_id"] == seeded_alert


def test_get_case(client, seeded_alert):
    # Create or get case
    rc = client.post(f"/api/cases/from-alert/{seeded_alert}")
    case_id = rc.json()["id"]
    r = client.get(f"/api/cases/{case_id}")
    assert r.status_code == 200
    assert r.json()["id"] == case_id


def test_update_case_status(client, seeded_alert):
    rc = client.post(f"/api/cases/from-alert/{seeded_alert}")
    case_id = rc.json()["id"]
    r = client.patch(f"/api/cases/{case_id}/status", json={"status": "UNDER_REVIEW"})
    assert r.status_code == 200
    assert r.json()["status"] == "UNDER_REVIEW"


def test_update_case_invalid_status(client, seeded_alert):
    rc = client.post(f"/api/cases/from-alert/{seeded_alert}")
    case_id = rc.json()["id"]
    r = client.patch(f"/api/cases/{case_id}/status", json={"status": "INVALID"})
    assert r.status_code == 422


def test_add_case_note(client, seeded_alert):
    rc = client.post(f"/api/cases/from-alert/{seeded_alert}")
    case_id = rc.json()["id"]
    r = client.post(f"/api/cases/{case_id}/notes", json={"content": "Test note", "author": "TestUser"})
    assert r.status_code == 200
    assert r.json()["content"] == "Test note"


def test_note_persists(client, seeded_alert):
    rc = client.post(f"/api/cases/from-alert/{seeded_alert}")
    case_id = rc.json()["id"]
    client.post(f"/api/cases/{case_id}/notes", json={"content": "Persistent note"})
    r = client.get(f"/api/cases/{case_id}")
    assert any(n["content"] == "Persistent note" for n in r.json()["notes"])


# ── Settings ──────────────────────────────────────────────────────────────────
def test_get_settings(client):
    r = client.get("/api/settings")
    assert r.status_code == 200
    d = r.json()
    assert "high_threshold" in d and "medium_threshold" in d


def test_update_settings(client):
    r = client.patch("/api/settings", json={"high_threshold": 0.75, "default_page_size": 50})
    assert r.status_code == 200
    assert r.json()["high_threshold"] == 0.75
    assert r.json()["default_page_size"] == 50


def test_settings_invalid_threshold(client):
    r = client.patch("/api/settings", json={"high_threshold": 1.5})
    assert r.status_code == 422


# ── Model ─────────────────────────────────────────────────────────────────────
def test_model_status(client):
    r = client.get("/api/model/status")
    assert r.status_code == 200
    d = r.json()
    assert "loaded" in d


def test_score_prediction(client):
    r_status = client.get("/api/model/status")
    if not r_status.json().get("loaded"):
        pytest.skip("Model not loaded")
    features = {k: 0.0 for k in r_status.json().get("feature_cols", [])}
    r = client.post("/api/model/predictions/score", json={"features": features})
    assert r.status_code == 200
    d = r.json()
    if d.get("risk_score") is not None:
        assert 0.0 <= d["risk_score"] <= 1.0
