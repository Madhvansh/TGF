"""Tests for Virtual Sensor."""
import os
import sys
import tempfile
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tgf_dosing.core.virtual_sensor import VirtualSensor, TARGETS


def test_init_default():
    """VirtualSensor initializes in unavailable state."""
    vs = VirtualSensor()
    assert not vs.available
    assert vs._model is None


def test_predict_fallback_when_untrained():
    """Predict returns physics values with RED confidence when untrained."""
    vs = VirtualSensor()
    preds, confidence = vs.predict(
        ph=7.8, conductivity=2500, temperature=32, orp=650,
        coc=5.0, physics_hardness=400, physics_calcium=300,
        physics_alkalinity=200,
    )
    assert confidence == "RED"
    assert preds["total_hardness"] == 400
    assert preds["calcium_hardness"] == 300
    assert preds["total_alkalinity"] == 200


def test_dsigma_dt_computation():
    """Rate-of-change of conductivity computed correctly."""
    vs = VirtualSensor()
    assert vs._compute_dsigma_dt(2500) == 0.0  # first reading
    assert vs._compute_dsigma_dt(2510) == 10.0
    assert vs._compute_dsigma_dt(2505) == -5.0


def test_save_load_roundtrip():
    """Save and load preserves model state."""
    vs = VirtualSensor()
    # Simulate having a trained model
    vs._model = "fake_model"
    vs._scaler_X = "fake_scaler_X"
    vs._scaler_y = "fake_scaler_y"
    vs._train_std = np.array([1.0, 2.0, 3.0])
    vs._available = True

    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        path = f.name

    try:
        vs.save(path)
        assert os.path.exists(path)

        vs2 = VirtualSensor()
        vs2.load(path)
        assert vs2.available
        assert vs2._model == "fake_model"
        assert np.allclose(vs2._train_std, [1.0, 2.0, 3.0])
    finally:
        os.unlink(path)


def test_load_nonexistent_file():
    """Loading from nonexistent file keeps sensor unavailable."""
    vs = VirtualSensor(model_path="/nonexistent/path.pkl")
    assert not vs.available


def test_targets_defined():
    """TARGETS list has the expected entries."""
    assert "total_hardness" in TARGETS
    assert "calcium_hardness" in TARGETS
    assert "total_alkalinity" in TARGETS
    assert len(TARGETS) == 3
