"""Tests for Online Anomaly Detector."""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tgf_dosing.models.online_detector import OnlineAnomalyDetector


def test_init():
    """OnlineAnomalyDetector initializes."""
    od = OnlineAnomalyDetector()
    # Available depends on whether river is installed
    assert isinstance(od.available, bool)


def test_score_returns_float():
    """score_and_learn always returns a float."""
    od = OnlineAnomalyDetector()
    score = od.score_and_learn({
        "ph": 7.8, "conductivity": 2500,
        "temperature": 32.0, "orp": 650.0,
    })
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_score_empty_reading():
    """Empty reading returns 0.0."""
    od = OnlineAnomalyDetector()
    score = od.score_and_learn({})
    assert score == 0.0


def test_score_none_values():
    """None values are filtered out."""
    od = OnlineAnomalyDetector()
    score = od.score_and_learn({"ph": None, "conductivity": None})
    assert score == 0.0


def test_get_status():
    """get_status returns expected structure."""
    od = OnlineAnomalyDetector()
    status = od.get_status()
    assert "available" in status
    assert "samples_processed" in status
    assert "backend" in status


def test_multiple_scores():
    """Multiple calls increment sample count."""
    od = OnlineAnomalyDetector()
    for i in range(10):
        od.score_and_learn({"ph": 7.8 + i * 0.01, "conductivity": 2500 + i})

    if od.available:
        assert od._sample_count == 10
    else:
        assert od._sample_count == 0
