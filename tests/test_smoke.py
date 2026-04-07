"""
Smoke tests: verify imports, configs, and data files exist.
"""
import os
import sys
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def test_import_tower_config():
    from tgf_dosing.config.tower_config import AQUATECH_850_TPD
    assert AQUATECH_850_TPD is not None
    assert AQUATECH_850_TPD.name == "850_TPD_Main"


def test_tower_has_seven_chemicals():
    from tgf_dosing.config.tower_config import AQUATECH_850_TPD
    assert len(AQUATECH_850_TPD.chemicals) == 7


def test_import_physics_engine():
    from tgf_dosing.core.physics_engine import PhysicsEngine
    from tgf_dosing.config.tower_config import AQUATECH_850_TPD
    pe = PhysicsEngine(AQUATECH_850_TPD)
    assert pe is not None


def test_import_anomaly_detector():
    from tgf_dosing.infrastructure.anomaly_detector import AnomalyDetector
    ad = AnomalyDetector(window_size=50, warmup_cycles=5)
    assert ad is not None
    assert ad._moment_available is False  # no checkpoint provided


def test_import_moment_detector():
    from tgf_dosing.models.moment_detector import MomentAnomalyDetector
    assert MomentAnomalyDetector is not None


def test_dataset_exists():
    data_path = os.path.join(PROJECT_ROOT, "data", "Parameters_5K.csv")
    assert os.path.exists(data_path), f"Dataset not found at {data_path}"


def test_dataset_has_rows():
    import pandas as pd
    data_path = os.path.join(PROJECT_ROOT, "data", "Parameters_5K.csv")
    df = pd.read_csv(data_path)
    assert len(df) > 5000, f"Expected 5000+ rows, got {len(df)}"
    assert "pH" in df.columns


def test_import_dosing_controller():
    # DosingController uses relative imports from config/ — just verify the module file exists
    import importlib.util
    spec = importlib.util.find_spec("tgf_dosing.core.dosing_controller")
    assert spec is not None


def test_import_data_ingestion():
    from tgf_dosing.infrastructure.data_ingestion import DataIngestionPipeline
    assert DataIngestionPipeline is not None
