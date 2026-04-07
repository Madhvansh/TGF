"""
TGF Drift Detector
====================
ADWIN-based concept drift detection for sensor parameters.

FIX: Previous version declared 6 ADWIN detectors (ph, conductivity,
temperature, orp, tds, hardness) but main.py only feeds 4 parameters
(ph, conductivity, temperature, orp). With vote_threshold=3 and only
4 active detectors, drift detection was possible but the 2 dead
detectors wasted memory and confused the status report.

Now: only monitors parameters that are actually fed every cycle.
TDS and hardness can be added when those sensors are available.
"""
import logging
import time
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Per-feature ADWIN drift detection with majority voting.
    Falls back to rolling-mean detector if river is not installed.
    """

    def __init__(self, vote_threshold: int = 3,
                 cooldown_seconds: float = 3600.0):
        self.vote_threshold = vote_threshold
        self.cooldown_seconds = cooldown_seconds
        self._last_alert_time = 0.0
        self._total_updates = 0
        self._total_drifts = 0
        self._available = False

        # ONLY parameters that main.py actually feeds every cycle
        # delta: smaller = more sensitive to drift
        self._deltas = {
            "ph": 0.001,           # pH shifts of 0.1 are significant
            "conductivity": 0.002, # standard sensitivity
            "temperature": 0.005,  # higher tolerance (seasonal)
            "orp": 0.001,          # ORP shifts matter for biocide efficacy
        }

        try:
            from river.drift import ADWIN
            self._detectors = {
                param: ADWIN(delta=delta)
                for param, delta in self._deltas.items()
            }
            self._available = True
            logger.info(f"ADWIN drift detection active on "
                        f"{len(self._deltas)} parameters")
        except ImportError:
            self._detectors = {}
            self._fallback_windows: Dict[str, List[float]] = {
                p: [] for p in self._deltas
            }
            logger.info("river not installed — using rolling-mean fallback")

    @property
    def available(self) -> bool:
        return True  # Always available (has fallback)

    def update(self, readings: Dict[str, Optional[float]]
               ) -> Tuple[bool, List[str]]:
        """
        Feed one set of readings and check for drift.

        Args:
            readings: {"ph": float, "conductivity": float, ...}
                      Keys must match self._deltas keys.

        Returns:
            (drift_detected, list_of_drifted_parameters)
        """
        self._total_updates += 1
        drifted = []

        if self._available:
            drifted = self._update_adwin(readings)
        else:
            drifted = self._update_fallback(readings)

        if len(drifted) >= self.vote_threshold:
            now = time.time()
            if now - self._last_alert_time >= self.cooldown_seconds:
                self._last_alert_time = now
                self._total_drifts += 1
                logger.warning(
                    f"Drift detected in {len(drifted)} parameters: "
                    f"{drifted}")
                return True, drifted

        return False, drifted

    def _update_adwin(self, readings: Dict[str, Optional[float]]
                      ) -> List[str]:
        drifted = []
        for param, detector in self._detectors.items():
            val = readings.get(param)
            if val is not None:
                detector.update(val)
                if detector.drift_detected:
                    drifted.append(param)
        return drifted

    def _update_fallback(self, readings: Dict[str, Optional[float]]
                         ) -> List[str]:
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
            hist_std = max(
                (sum((x - hist_mean) ** 2 for x in historical) /
                 len(historical)) ** 0.5, 1e-8)
            recent_mean = sum(recent) / len(recent)
            z = abs(recent_mean - hist_mean) / (
                hist_std / len(recent) ** 0.5)
            if z > 3.0:
                drifted.append(param)

        return drifted

    def get_status(self) -> Dict:
        return {
            "available": self._available,
            "backend": "ADWIN" if self._available else "rolling-mean",
            "total_updates": self._total_updates,
            "total_drift_events": self._total_drifts,
            "parameters_monitored": list(self._deltas.keys()),
            "active_detectors": len(self._deltas),
            "vote_threshold": self.vote_threshold,
        }