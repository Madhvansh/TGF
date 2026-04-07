"""Tests for Cascade Detector."""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tgf_dosing.core.cascade_detector import CascadeDetector, STATES, CAUSAL_LINKS


def test_initial_state_healthy():
    """Detector starts in HEALTHY state."""
    cd = CascadeDetector()
    assert cd.state == "HEALTHY"
    assert cd._state_index == 0


def test_states_list():
    """States list has the right entries in order."""
    assert STATES == ["HEALTHY", "CORROSION", "PARTICLES", "BIOFILM", "SCALE"]


def test_update_returns_dict():
    """Update returns a dict with expected keys."""
    cd = CascadeDetector(step=1)
    result = cd.update(iron=0.1, turbidity=5.0, frc=1.0, calcium_hardness=200)
    assert "state" in result
    assert "triggered" in result
    assert "link" in result
    assert "p_value" in result


def test_update_not_triggered_with_little_data():
    """No transition with insufficient data."""
    cd = CascadeDetector(step=1)
    for _ in range(10):
        result = cd.update(iron=0.1, turbidity=5.0, frc=1.0, calcium_hardness=200)
    assert result["triggered"] is False
    assert cd.state == "HEALTHY"


def test_reset():
    """Reset returns to HEALTHY."""
    cd = CascadeDetector()
    cd._state_index = 3
    cd.reset()
    assert cd.state == "HEALTHY"
    assert cd.last_triggered_link is None


def test_get_status():
    """get_status returns expected structure."""
    cd = CascadeDetector()
    status = cd.get_status()
    assert status["state"] == "HEALTHY"
    assert "root_cause_action" in status
    assert "samples" in status


def test_root_cause_actions():
    """Each state has a specific action recommendation."""
    cd = CascadeDetector()
    for i, state in enumerate(STATES):
        cd._state_index = i
        action = cd._root_cause_action()
        assert len(action) > 0
        assert isinstance(action, str)


def test_causal_links_defined():
    """Causal links are properly defined."""
    assert len(CAUSAL_LINKS) == 3
    for cause, effect, transition in CAUSAL_LINKS:
        assert transition in STATES
        assert cause in ["iron", "turbidity", "frc"]
        assert effect in ["turbidity", "frc", "calcium_hardness"]
