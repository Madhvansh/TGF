"""Tests for DataStore."""
import os
import sys
import time
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tgf_dosing.infrastructure.data_store import DataStore


@pytest.fixture
def store():
    """Create a temporary DataStore."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    ds = DataStore(db_path=path)
    yield ds
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def populated_store(store):
    """DataStore with some test data."""
    ts_base = 1700000000.0
    for i in range(20):
        store.save_sensor_reading(
            timestamp=ts_base + i * 300,
            cycle_index=i + 1,
            ph=7.5 + (i % 5) * 0.1,
            conductivity=2500 + i * 10,
            temperature=32.0 + i * 0.1,
            orp=350 + i,
        )
        store.save_control_decision(
            timestamp=ts_base + i * 300,
            cycle_index=i + 1,
            lsi=0.5 + i * 0.05,
            rsi=7.0 - i * 0.05,
            risk_level="LOW" if i < 15 else "MODERATE",
            risk_score=0.2 + i * 0.01,
            primary_risk="scaling" if i > 10 else "none",
            continuous_doses={"AQUATREAT-2501": 0.1 + i * 0.01},
            slug_doses={},
            blowdown=0.1,
            preemptive=i % 3 == 0,
            opt_cost=10.0 + i,
            converged=True,
            safety_overrides=0,
            emergency_stop=False,
            pid_active=False,
            cycle_cost=50.0 + i * 2,
        )
    # Add some anomalies
    for i in [5, 10, 15]:
        store.save_anomaly_event(
            timestamp=ts_base + i * 300,
            cycle_index=i + 1,
            system_score=0.7 + i * 0.01,
            classification="ANOMALOUS",
            suspect_sensor="ph",
        )
    return store


def test_init_creates_tables(store):
    """DataStore creates all required tables."""
    with store._get_conn() as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {r['name'] for r in tables}
    assert 'sensor_readings' in table_names
    assert 'control_decisions' in table_names
    assert 'anomaly_events' in table_names
    assert 'alerts' in table_names
    assert 'chemical_residuals' in table_names
    assert 'hourly_metrics' in table_names
    assert 'lab_calibrations' in table_names


def test_get_chart_data_empty(store):
    """get_chart_data returns correct empty structure."""
    data = store.get_chart_data()
    assert data["labels"] == []
    assert data["ph"] == []
    assert "lsi" in data
    assert "costs" in data


def test_get_chart_data_populated(populated_store):
    """get_chart_data returns correct data shape."""
    data = populated_store.get_chart_data(limit=10)
    assert len(data["labels"]) <= 10
    assert len(data["ph"]) == len(data["labels"])
    assert len(data["lsi"]) == len(data["labels"])
    assert len(data["costs"]) == len(data["labels"])
    # Data should be in chronological order
    assert data["labels"] == sorted(data["labels"])


def test_max_timestamp_empty(store):
    """_max_timestamp returns current time for empty table."""
    ts = store._max_timestamp("sensor_readings")
    assert abs(ts - time.time()) < 5  # within 5 seconds


def test_max_timestamp_populated(populated_store):
    """_max_timestamp returns actual max from data."""
    ts = populated_store._max_timestamp("sensor_readings")
    assert ts > 1700000000.0


def test_get_recent_readings(populated_store):
    """get_recent_readings returns data relative to max timestamp."""
    readings = populated_store.get_recent_readings(hours=1)
    assert len(readings) > 0
    assert 'ph' in readings[0]


def test_get_dashboard_summary(populated_store):
    """get_dashboard_summary aggregates correctly."""
    summary = populated_store.get_dashboard_summary()
    assert summary["total_cycles"] == 20
    assert "risk_distribution" in summary
    assert summary["total_cost_inr"] > 0


def test_get_anomaly_timeline(populated_store):
    """get_anomaly_timeline returns anomaly events."""
    events = populated_store.get_anomaly_timeline()
    assert len(events) == 3
    assert events[0]["system_classification"] == "ANOMALOUS"


def test_get_simulation_summary(populated_store):
    """get_simulation_summary returns comprehensive stats."""
    summary = populated_store.get_simulation_summary()
    assert summary["total_cycles"] == 20
    assert "cost" in summary
    assert summary["cost"]["total_inr"] > 0
    assert "lsi" in summary
    assert summary["anomalies"] == 3
    assert "preemptive_pct" in summary


def test_get_dosing_history_empty(store):
    """get_dosing_history returns empty structure when no data."""
    data = store.get_dosing_history()
    assert data["labels"] == []
    assert data["chemicals"] == {}


def test_save_and_get_alerts(store):
    """Save and retrieve alerts."""
    alert_id = store.save_alert(
        timestamp=time.time(),
        severity="WARNING",
        category="test",
        title="Test Alert",
        message="This is a test",
    )
    assert alert_id > 0
    unacked = store.get_unacknowledged_alerts()
    assert len(unacked) == 1
    assert unacked[0]["title"] == "Test Alert"

    store.acknowledge_alert(alert_id)
    unacked = store.get_unacknowledged_alerts()
    assert len(unacked) == 0
