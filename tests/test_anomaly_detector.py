"""
Unit tests for AnomalyDetector.
"""
import os
import sys
import time
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from tgf_dosing.infrastructure.anomaly_detector import AnomalyDetector


@pytest.fixture
def detector():
    return AnomalyDetector(window_size=100, warmup_cycles=10)


def _normal_reading():
    return dict(ph=7.5, conductivity=2500, temperature=32, orp=350)


def _feed_normal(det, n=20):
    """Feed n normal readings to build up statistics."""
    for i in range(n):
        det.analyze(
            ph=7.5 + (i % 3) * 0.1,
            conductivity=2500 + (i % 5) * 10,
            temperature=32 + (i % 4) * 0.5,
            orp=350 + (i % 3) * 5,
            cycle_index=i,
        )


def test_warmup_returns_normal(detector):
    """During warmup, all readings should be classified NORMAL."""
    for i in range(detector.warmup_cycles - 1):
        report = detector.analyze(
            ph=7.5, conductivity=2500, temperature=32, orp=350,
            cycle_index=i,
        )
        assert report.system_classification == "NORMAL"


def test_normal_readings_stay_normal(detector):
    """Consistent normal readings should not trigger anomalies."""
    _feed_normal(detector, 30)
    report = detector.analyze(
        ph=7.5, conductivity=2510, temperature=32.2, orp=348,
        cycle_index=31,
    )
    assert report.system_classification == "NORMAL"
    assert report.system_score < 0.4


def test_extreme_ph_triggers_anomaly(detector):
    """pH of 12.0 should be detected as anomalous or critical."""
    _feed_normal(detector, 20)
    report = detector.analyze(
        ph=12.0, conductivity=2500, temperature=32, orp=350,
        cycle_index=21,
    )
    assert report.system_classification in ("ANOMALOUS", "CRITICAL")
    assert report.system_score > 0.4


def test_extreme_temperature_triggers_anomaly(detector):
    """Temperature of 80C should be anomalous."""
    _feed_normal(detector, 20)
    report = detector.analyze(
        ph=7.5, conductivity=2500, temperature=80, orp=350,
        cycle_index=21,
    )
    assert report.system_classification in ("ANOMALOUS", "CRITICAL")


def test_suspect_sensor_identified(detector):
    """When one parameter is extreme, it should be identified as suspect."""
    _feed_normal(detector, 20)
    report = detector.analyze(
        ph=12.0, conductivity=2500, temperature=32, orp=350,
        cycle_index=21,
    )
    assert report.suspect_sensor == "pH"


def test_get_stats(detector):
    """Stats should report basic metrics."""
    _feed_normal(detector, 15)
    stats = detector.get_stats()
    assert "total_cycles_analyzed" in stats
    assert "anomaly_rate_pct" in stats
    assert stats["total_cycles_analyzed"] == 15


def test_moment_fallback_without_checkpoint(detector):
    """Without MOMENT checkpoint, statistical-only mode should work fine."""
    assert detector._moment_available is False
    _feed_normal(detector, 20)
    report = detector.analyze(
        ph=7.5, conductivity=2500, temperature=32, orp=350,
        cycle_index=21,
    )
    assert report is not None
    assert report.system_score >= 0


def test_cross_parameter_high_conductivity_low_ph(detector):
    """High conductivity + very low pH should trigger cross-parameter flags."""
    _feed_normal(detector, 20)
    report = detector.analyze(
        ph=4.0, conductivity=8000, temperature=32, orp=350,
        cycle_index=21, tds=6000,
    )
    assert report.system_classification in ("ANOMALOUS", "CRITICAL")
