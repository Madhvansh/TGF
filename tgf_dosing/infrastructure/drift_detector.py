"""
TGF Drift Detector
====================
ADWIN-based concept drift detection for sensor parameters.

Concept drift from seasonal changes, makeup water source changes,
or contamination silently degrades all ML models. ADWIN detects
both gradual and abrupt drift in O(log n) memory/time.

Uses majority voting: drift alert when >=vote_threshold parameters
simultaneously detect drift, with a cooldown between alerts.
"""
import logging
import time
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Per-feature ADWIN drift detection with majority voting.

    Falls back to a simple rolling-mean detector if river is not installed.
    """

    def __init__(self, vote_threshold: int = 3, cooldown_seconds: float = 3600.0):
        """
        Args:
            vote_threshold: Number of parameters that must drift simultaneously
            cooldown_seconds: Minimum time between drift alerts (default 1 hour)
        """
        self.vote_threshold = vote_threshold
        self.cooldown_seconds = cooldown_seconds
        self._last_alert_time = 0.0
        self._total_updates = 0
        self._total_drifts = 0
        self._available = False

        # ADWIN delta values per parameter (smaller = more sensitive)
        self._deltas = {
            "ph": 0.001,
            "conductivity": 0.002,
            "temperature": 0.005,
            "orp": 0.001,
            "tds": 0.002,
            "hardness": 0.002,
        }

        try:
            from river.drift import ADWIN
            self._detectors = {
                param: ADWIN(delta=delta)
                for param, delta in self._deltas.items()
            }
            self._available = True
            logger.info("ADWIN drift detection active (river installed)")
        except ImportError:
            self._detectors = {}
            self._fallback_windows: Dict[str, List[float]] = {
                p: [] for p in self._deltas
            }
            logger.info("river not installed — using rolling-mean drift fallback")

    @property
    def available(self) -> bool:
        return True  # Always available (has fallback)

    def update(self, readings: Dict[str, Optional[float]]) -> Tuple[bool, List[str]]:
        """
        Feed one set of readings and check for drift.

        Args:
            readings: dict mapping parameter names to values
                      (e.g. {"ph": 7.8, "conductivity": 2500, ...})

        Returns:
            (drift_detected, list_of_drifted_parameters)
            drift_detected is True only when vote_threshold met AND cooldown expired
        """
        self._total_updates += 1
        drifted = []

        if self._available:
            drifted = self._update_adwin(readings)
        else:
            drifted = self._update_fallback(readings)

        # Check vote threshold and cooldown
        if len(drifted) >= self.vote_threshold:
            now = time.time()
            if now - self._last_alert_time >= self.cooldown_seconds:
                self._last_alert_time = now
                self._total_drifts += 1
                logger.warning(f"Drift detected in {len(drifted)} parameters: {drifted}")
                return True, drifted

        return False, drifted

    def _update_adwin(self, readings: Dict[str, Optional[float]]) -> List[str]:
        """Update ADWIN detectors."""
        drifted = []
        for param, detector in self._detectors.items():
            val = readings.get(param)
            if val is not None:
                detector.update(val)
                if detector.drift_detected:
                    drifted.append(param)
        return drifted

    def _update_fallback(self, readings: Dict[str, Optional[float]]) -> List[str]:
        """Rolling-mean fallback: detect when recent mean differs from historical."""
        drifted = []
        window_size = 500
        recent_size = 50

        for param in self._deltas:
            val = readings.get(param)
            if val is None:
                continue

            buf = self._fallback_windows[param]
            buf.append(val)
            if len(buf) > window_size:
                buf.pop(0)

            if len(buf) < recent_size + 50:
                continue

            historical = buf[:-recent_size]
            recent = buf[-recent_size:]

            hist_mean = sum(historical) / len(historical)
            hist_std = max((sum((x - hist_mean) ** 2 for x in historical) / len(historical)) ** 0.5, 1e-8)
            recent_mean = sum(recent) / len(recent)

            # Z-test on means
            z = abs(recent_mean - hist_mean) / (hist_std / len(recent) ** 0.5)
            if z > 3.0:  # 3-sigma shift
                drifted.append(param)

        return drifted

    def get_status(self) -> Dict:
        """Current drift detection status for dashboard."""
        return {
            "available": self._available,
            "backend": "ADWIN" if self._available else "rolling-mean",
            "total_updates": self._total_updates,
            "total_drift_events": self._total_drifts,
            "parameters_monitored": list(self._deltas.keys()),
            "vote_threshold": self.vote_threshold,
        }
