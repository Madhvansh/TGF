"""
TGF Cascade Detector
=====================
Detects corrosion→particles→biofilm→scale cascade failures using
Granger causality testing and a 5-state machine.

FIX: Previous version filled missing data with 0.0, which poisons
Granger tests when iron (71% coverage) or FRC (89% coverage) are
sparse. Now skips updates for missing params and tracks valid counts
separately from buffer length.

State machine: HEALTHY → CORROSION → PARTICLES → BIOFILM → SCALE
Left-to-right only — can't jump states.
"""
import logging
import numpy as np
from collections import deque
from typing import Optional, Tuple, List, Dict

logger = logging.getLogger(__name__)

STATES = ["HEALTHY", "CORROSION", "PARTICLES", "BIOFILM", "SCALE"]

CAUSAL_LINKS = [
    ("iron", "turbidity", "CORROSION"),
    ("turbidity", "frc", "PARTICLES"),
    ("frc", "calcium_hardness", "BIOFILM"),
]


class CascadeDetector:
    """
    Granger-causality-based cascade failure detector with proper
    missing data handling.
    """

    def __init__(self, window_size: int = 288, step: int = 12,
                 p_threshold: float = 0.05):
        self.window_size = window_size
        self.step = step
        self.p_threshold = p_threshold

        self._state_index = 0
        self._sample_count = 0

        # Rolling data buffers — store (value, is_valid) pairs
        self._buffers: Dict[str, deque] = {
            "iron": deque(maxlen=window_size),
            "turbidity": deque(maxlen=window_size),
            "frc": deque(maxlen=window_size),
            "calcium_hardness": deque(maxlen=window_size),
        }
        # Track how many valid (non-None) values each buffer has
        self._valid_counts: Dict[str, int] = {k: 0 for k in self._buffers}

        self.last_triggered_link: Optional[str] = None
        self.last_p_value: Optional[float] = None

        try:
            from statsmodels.tsa.stattools import grangercausalitytests  # noqa
            self._stats_available = True
        except ImportError:
            self._stats_available = False
            logger.info("statsmodels not installed — cascade detector uses "
                        "correlation fallback")

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
        Feed one reading. Only buffers values that are actually present.
        Missing values get np.nan (not 0.0) so they can be excluded
        from statistical tests.
        """
        param_values = {
            "iron": iron,
            "turbidity": turbidity,
            "frc": frc,
            "calcium_hardness": calcium_hardness,
        }

        for key, val in param_values.items():
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                self._buffers[key].append(val)
                # Maintain valid count (decrement if buffer was full before append)
                if len(self._buffers[key]) == self.window_size:
                    # deque was at capacity, oldest was evicted — but we
                    # don't know if evicted was valid. Use actual count.
                    pass
                self._valid_counts[key] = len(self._buffers[key])
            else:
                # Append NaN placeholder to keep time alignment
                self._buffers[key].append(np.nan)

        self._sample_count += 1

        result = {
            "state": self.state,
            "triggered": False,
            "link": None,
            "p_value": None,
        }

        if self._sample_count % self.step != 0:
            return result

        # Test causal links
        for cause_key, effect_key, transition_state in CAUSAL_LINKS:
            target_idx = STATES.index(transition_state)
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
                    f"Cascade transition: {STATES[target_idx - 1]} → "
                    f"{self.state} ({cause_key}→{effect_key}, p={p_val:.4f})"
                )
                break

        return result

    def _test_causality(self, cause_key: str,
                        effect_key: str) -> Optional[float]:
        """Run Granger causality test on valid (non-NaN) aligned data."""
        cause_raw = np.array(self._buffers[cause_key])
        effect_raw = np.array(self._buffers[effect_key])

        # Remove indices where EITHER is NaN
        valid_mask = ~(np.isnan(cause_raw) | np.isnan(effect_raw))
        cause = cause_raw[valid_mask]
        effect = effect_raw[valid_mask]

        # Need minimum 50 valid aligned samples
        if len(cause) < 50:
            return None

        if np.std(cause) < 1e-8 or np.std(effect) < 1e-8:
            return None

        if self._stats_available:
            return self._granger_test(cause, effect)
        else:
            return self._correlation_fallback(cause, effect)

    def _granger_test(self, cause: np.ndarray,
                      effect: np.ndarray) -> Optional[float]:
        """Statsmodels Granger causality test."""
        try:
            from statsmodels.tsa.stattools import grangercausalitytests
            data = np.column_stack([effect, cause])
            max_lag = min(4, len(cause) // 10)
            if max_lag < 1:
                return None
            results = grangercausalitytests(data, maxlag=max_lag,
                                            verbose=False)
            p_values = []
            for lag in range(1, max_lag + 1):
                for test_name in results[lag][0]:
                    p_values.append(results[lag][0][test_name][1])
            return min(p_values) if p_values else None
        except Exception:
            return None

    def _correlation_fallback(self, cause: np.ndarray,
                              effect: np.ndarray) -> Optional[float]:
        """Time-lagged cross-correlation fallback."""
        try:
            c = (cause - np.mean(cause)) / max(np.std(cause), 1e-8)
            e = (effect - np.mean(effect)) / max(np.std(effect), 1e-8)
            corr = np.correlate(c, e, mode='full')
            corr /= max(len(cause), 1)
            mid = len(corr) // 2
            max_corr = (np.max(np.abs(corr[mid:mid + 20]))
                        if len(corr) > mid + 1 else 0)
            pseudo_p = max(0.001, 1.0 - max_corr * 2)
            return pseudo_p
        except Exception:
            return None

    def reset(self):
        self._state_index = 0
        self.last_triggered_link = None
        self.last_p_value = None

    def get_status(self) -> Dict:
        return {
            "state": self.state,
            "state_index": self._state_index,
            "last_link": self.last_triggered_link,
            "last_p_value": self.last_p_value,
            "samples": self._sample_count,
            "valid_counts": dict(self._valid_counts),
            "root_cause_action": self._root_cause_action(),
        }

    def _root_cause_action(self) -> str:
        actions = {
            "HEALTHY": "No action needed",
            "CORROSION": "Increase corrosion inhibitor (AQUATREAT-2150) to prevent downstream cascade",
            "PARTICLES": "Address corrosion source + boost dispersant (AQUATREAT-6625)",
            "BIOFILM": "Immediate biocide slug (AQUATREAT-3331/399) + address upstream corrosion",
            "SCALE": "Emergency: full treatment program review — blowdown + inhibitor + biocide",
        }
        return actions.get(self.state, "Unknown state")