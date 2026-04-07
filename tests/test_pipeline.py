"""
Integration test: run a short pipeline cycle and verify no crashes.
"""
import os
import sys
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def test_pipeline_10_cycles():
    """Run 10 cycles headless — verify no crashes and data populated."""
    import subprocess
    csv_path = os.path.join(PROJECT_ROOT, "data", "Parameters_5K.csv")
    if not os.path.exists(csv_path):
        pytest.skip("Parameters_5K.csv not found")

    result = subprocess.run(
        [sys.executable, "-m", "tgf_dosing.main",
         "--data", csv_path, "--cycles", "10", "--no-api", "--speed", "0", "--no-forecast"],
        capture_output=True, text=True, timeout=120,
        cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0, f"Pipeline failed:\n{result.stderr[-500:]}"
    assert "Simulation Complete" in result.stderr or "Simulation Complete" in result.stdout


def test_physics_engine_lsi():
    """Verify physics engine LSI calculation produces valid results."""
    from tgf_dosing.core.physics_engine import PhysicsEngine, WaterChemistry
    from tgf_dosing.config.tower_config import AQUATECH_850_TPD

    pe = PhysicsEngine(AQUATECH_850_TPD)
    wc = WaterChemistry(
        ph=7.5,
        temperature_c=32,
        conductivity_us=2500,
        orp_mv=350,
        tds_ppm=1600,
        calcium_hardness_ppm=400,
        total_alkalinity_ppm=200,
        total_hardness_ppm=600,
    )
    lsi = pe.calculate_lsi(wc)
    assert -3.0 < lsi < 3.0, f"LSI {lsi} out of reasonable range"


def test_physics_engine_coc():
    """Verify CoC estimation."""
    from tgf_dosing.core.physics_engine import PhysicsEngine
    from tgf_dosing.config.tower_config import AQUATECH_850_TPD

    pe = PhysicsEngine(AQUATECH_850_TPD)
    coc = pe.estimate_coc(2500)
    assert 1.0 <= coc <= 15.0, f"CoC {coc} out of range"
