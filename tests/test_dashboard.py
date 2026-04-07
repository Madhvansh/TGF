"""Tests for Dashboard API endpoints."""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from fastapi.testclient import TestClient
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


@pytest.fixture
def client():
    """Create a test client with mock components."""
    if not FASTAPI_AVAILABLE:
        pytest.skip("FastAPI not installed")

    from tgf_dosing.infrastructure.dashboard_api import create_api
    app = create_api()
    return TestClient(app)


def test_dashboard_html(client):
    """GET / returns HTML dashboard."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "TGF" in resp.text
    assert "Chart.js" in resp.text or "chart.js" in resp.text


def test_status_endpoint(client):
    """GET /api/status returns valid response."""
    resp = client.get("/api/status")
    assert resp.status_code == 200


def test_simulation_stats(client):
    """GET /api/simulation/stats returns progress."""
    resp = client.get("/api/simulation/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "running" in data
    assert "current" in data
    assert "total" in data


def test_charts_timeseries_empty(client):
    """GET /api/charts/timeseries returns empty arrays when no data."""
    resp = client.get("/api/charts/timeseries")
    assert resp.status_code == 200
    data = resp.json()
    assert "labels" in data
    assert "ph" in data


def test_charts_roi_empty(client):
    """GET /api/charts/roi returns zero savings when no data."""
    resp = client.get("/api/charts/roi")
    assert resp.status_code == 200
    data = resp.json()
    assert data["savings_pct"] == 0


def test_charts_risk_empty(client):
    """GET /api/charts/risk returns zero counts when no data."""
    resp = client.get("/api/charts/risk")
    assert resp.status_code == 200
    data = resp.json()
    assert "LOW" in data


def test_charts_anomalies_empty(client):
    """GET /api/charts/anomalies returns empty list when no data."""
    resp = client.get("/api/charts/anomalies")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


def test_charts_dosing_history_empty(client):
    """GET /api/charts/dosing_history returns empty when no data."""
    resp = client.get("/api/charts/dosing_history")
    assert resp.status_code == 200
    data = resp.json()
    assert "labels" in data


def test_alerts_empty(client):
    """GET /api/alerts returns empty list when no alerts."""
    resp = client.get("/api/alerts")
    assert resp.status_code == 200


def test_chemicals_empty(client):
    """GET /api/chemicals returns empty when no controller data."""
    resp = client.get("/api/chemicals")
    assert resp.status_code == 200


def test_control_pause_resume(client):
    """POST /api/control/pause and /resume toggle state."""
    resp = client.post("/api/control/pause")
    assert resp.status_code == 200
    assert resp.json()["status"] == "paused"

    resp = client.get("/api/control/state")
    assert resp.json()["paused"] is True

    resp = client.post("/api/control/resume")
    assert resp.status_code == 200
    assert resp.json()["status"] == "running"

    resp = client.get("/api/control/state")
    assert resp.json()["paused"] is False


def test_report_summary_empty(client):
    """GET /api/report/summary returns empty when no data store."""
    resp = client.get("/api/report/summary")
    assert resp.status_code == 200
