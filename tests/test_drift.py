"""Tests for Drift Detector."""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tgf_dosing.infrastructure.drift_detector import DriftDetector


def test_init():
    """DriftDetector initializes properly."""
    dd = DriftDetector()
    assert dd.vote_threshold == 3
    assert dd._total_updates == 0
    assert dd._total_drifts == 0


def test_update_returns_tuple():
    """Update returns (bool, list) tuple."""
    dd = DriftDetector()
    result = dd.update({"ph": 7.8, "conductivity": 2500})
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], bool)
    assert isinstance(result[1], list)


def test_no_drift_on_stable_data():
    """No drift detected on stable readings."""
    dd = DriftDetector(vote_threshold=3)
    for _ in range(100):
        drift, params = dd.update({
            "ph": 7.8,
            "conductivity": 2500,
            "temperature": 32.0,
            "orp": 650.0,
        })
    assert drift is False
    assert dd._total_drifts == 0


def test_update_counter():
    """Total updates counter increments."""
    dd = DriftDetector()
    for _ in range(10):
        dd.update({"ph": 7.8})
    assert dd._total_updates == 10


def test_get_status():
    """get_status returns expected structure."""
    dd = DriftDetector()
    status = dd.get_status()
    assert "available" in status
    assert "backend" in status
    assert "total_updates" in status
    assert "total_drift_events" in status
    assert "parameters_monitored" in status
    assert "vote_threshold" in status


def test_always_available():
    """DriftDetector is always available (has fallback)."""
    dd = DriftDetector()
    assert dd.available is True


def test_handles_none_values():
    """Handles None values in readings gracefully."""
    dd = DriftDetector()
    drift, params = dd.update({"ph": None, "conductivity": None})
    assert drift is False
