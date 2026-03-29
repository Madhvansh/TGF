"""
TGF Cascade Detector
=====================
Detects corrosion→particles→biofilm→scale cascade failures using
Granger causality testing and a 5-state machine.

The cascade is the most dangerous failure mode: one mechanism triggers
the next, creating a positive feedback loop that's hard to stop once
it reaches Stage 3+.

State machine: HEALTHY → CORROSION → PARTICLES → BIOFILM → SCALE
Left-to-right only — can't jump states.
"""
import logging
import numpy as np
from collections import deque
from typing import Optional, Tuple, List, Dict

logger = logging.getLogger(__name__)

STATES = ["HEALTHY", "CORROSION", "PARTICLES", "BIOFILM", "SCALE"]

# Causal links tested via Granger causality
CAUSAL_LINKS = [
    ("iron", "turbidity", "CORROSION"),     # corrosion → particles
    ("turbidity", "frc", "PARTICLES"),       # particles → biofilm indicator
    ("frc", "calcium_hardness", "BIOFILM"),  # biofilm → scaling
]


class CascadeDetector:
    """
    Granger-causality-based cascade failure detector.

    Maintains rolling windows of sensor data and tests causal
    links between iron→turbidity→FRC→calcium_hardness.

    Falls back gracefully if statsmodels is not installed.
    """

    def __init__(self, window_size: int = 288, step: int = 12, p_threshold: float = 0.05):
        """
        Args:
            window_size: Rolling window samples (default 288 = 24h at 5-min)
            step: Re-test every N samples (default 12 = hourly)
            p_threshold: Granger causality significance threshold
        """
        self.window_size = window_size
        self.step = step
        self.p_threshold = p_threshold

        self._state_index = 0  # index into STATES
        self._sample_count = 0

        # Rolling data buffers
        self._buffers: Dict[str, deque] = {
            "iron": deque(maxlen=window_size),
            "turbidity": deque(maxlen=window_size),
            "frc": deque(maxlen=window_size),
            "calcium_hardness": deque(maxlen=window_size),
        }

        # Last detected link info
        self.last_triggered_link: Optional[str] = None
        self.last_p_value: Optional[float] = None

        # Check for statsmodels
        try:
            from statsmodels.tsa.stattools import grangercausalitytests  # noqa: F401
            self._stats_available = True
        except ImportError:
            self._stats_available = False
            logger.info("statsmodels not installed — cascade detector uses correlation fallback")

    @property
    def state(self) -> str:
        return STATES[self._state_index]

    def update(self,
               iron: Optional[float] = None,
               turbidity: Optional[float] = None,
               frc: Optional[float] = None,
               calcium_hardness: Optional[float] = None,
               ) -> Dict:
        """
        Feed one reading and check for cascade transitions.

        Returns:
            dict with keys: state, triggered, link, p_value
        """
        # Fill buffers (use 0.0 for missing)
        self._buffers["iron"].append(iron or 0.0)
        self._buffers["turbidity"].append(turbidity or 0.0)
        self._buffers["frc"].append(frc or 0.0)
        self._buffers["calcium_hardness"].append(calcium_hardness or 0.0)
        self._sample_count += 1

        result = {
            "state": self.state,
            "triggered": False,
            "link": None,
            "p_value": None,
        }

        # Only test periodically and after enough data
        if self._sample_count % self.step != 0:
            return result
        if len(self._buffers["iron"]) < max(50, self.step * 4):
            return result

        # Test causal links in order
        for cause_key, effect_key, transition_state in CAUSAL_LINKS:
            target_idx = STATES.index(transition_state)
            # Only test the NEXT transition (left-to-right)
            if target_idx != self._state_index + 1:
                continue

            p_val = self._test_causality(cause_key, effect_key)
            if p_val is not None and p_val < self.p_threshold:
                self._state_index = target_idx
                self.last_triggered_link = f"{cause_key}→{effect_key}"
                self.last_p_value = p_val
                result["triggered"] = True
                result["link"] = self.last_triggered_link
                result["p_value"] = round(p_val, 4)
                result["state"] = self.state
                logger.warning(
                    f"Cascade transition: {STATES[target_idx - 1]} → {self.state} "
                    f"({cause_key}→{effect_key}, p={p_val:.4f})"
                )
                break  # one transition per update

        return result

    def _test_causality(self, cause_key: str, effect_key: str) -> Optional[float]:
        """Run Granger causality test; return min p-value or None."""
        cause = np.array(self._buffers[cause_key])
        effect = np.array(self._buffers[effect_key])

        # Skip if either is constant
        if np.std(cause) < 1e-8 or np.std(effect) < 1e-8:
            return None

        if self._stats_available:
            return self._granger_test(cause, effect)
        else:
            return self._correlation_fallback(cause, effect)

    def _granger_test(self, cause: np.ndarray, effect: np.ndarray) -> Optional[float]:
        """Statsmodels Granger causality test."""
        try:
            from statsmodels.tsa.stattools import grangercausalitytests
            data = np.column_stack([effect, cause])
            max_lag = min(4, len(cause) // 10)
            if max_lag < 1:
                return None
            results = grangercausalitytests(data, maxlag=max_lag, verbose=False)
            # Get minimum p-value across lags
            p_values = []
            for lag in range(1, max_lag + 1):
                for test_name in results[lag][0]:
                    p_values.append(results[lag][0][test_name][1])
            return min(p_values) if p_values else None
        except Exception:
            return None

    def _correlation_fallback(self, cause: np.ndarray, effect: np.ndarray) -> Optional[float]:
        """Time-lagged cross-correlation fallback (no statsmodels)."""
        try:
            from scipy import signal as sig
            # Normalize
            c = (cause - np.mean(cause)) / max(np.std(cause), 1e-8)
            e = (effect - np.mean(effect)) / max(np.std(effect), 1e-8)
            corr = np.correlate(c, e, mode='full')
            corr /= max(len(cause), 1)
            mid = len(corr) // 2
            # Check if cause leads effect (positive lags)
            max_corr = np.max(np.abs(corr[mid:mid + 20])) if len(corr) > mid + 1 else 0
            # Convert to pseudo p-value (higher correlation → lower p)
            pseudo_p = max(0.001, 1.0 - max_corr * 2)
            return pseudo_p
        except Exception:
            return None

    def reset(self):
        """Reset to HEALTHY state."""
        self._state_index = 0
        self.last_triggered_link = None
        self.last_p_value = None

    def get_status(self) -> Dict:
        """Current cascade status for dashboard."""
        return {
            "state": self.state,
            "state_index": self._state_index,
            "last_link": self.last_triggered_link,
            "last_p_value": self.last_p_value,
            "samples": self._sample_count,
            "root_cause_action": self._root_cause_action(),
        }

    def _root_cause_action(self) -> str:
        """Suggest root-cause treatment based on current state."""
        actions = {
            "HEALTHY": "No action needed",
            "CORROSION": "Increase corrosion inhibitor (AQUATREAT-2150) to prevent downstream cascade",
            "PARTICLES": "Address corrosion source + boost dispersant (AQUATREAT-6625)",
            "BIOFILM": "Immediate biocide slug (AQUATREAT-3331/399) + address upstream corrosion",
            "SCALE": "Emergency: full treatment program review — blowdown + inhibitor + biocide",
        }
        return actions.get(self.state, "Unknown state")
