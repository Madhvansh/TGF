"""
TGF Online Anomaly Detector
==============================
Incremental anomaly detection using River's HalfSpaceTrees.

Unlike MOMENT (batch inference), HalfSpaceTrees adapts per-sample
with no retraining — truly online learning.

Score range: 0.0 (normal) to 1.0 (anomalous).
"""
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class OnlineAnomalyDetector:
    """
    Incremental anomaly detection using HalfSpaceTrees.

    Falls back gracefully if river is not installed — score_and_learn
    returns 0.0 (no anomaly signal).
    """

    def __init__(self, n_trees: int = 10, height: int = 8, window_size: int = 250):
        """
        Args:
            n_trees: Number of half-space trees
            height: Maximum tree height
            window_size: Sliding window for reference distribution
        """
        self._available = False
        self._model = None
        self._sample_count = 0

        try:
            from river import anomaly, compose, preprocessing
            self._model = compose.Pipeline(
                preprocessing.MinMaxScaler(),
                anomaly.HalfSpaceTrees(
                    n_trees=n_trees,
                    height=height,
                    window_size=window_size,
                )
            )
            self._available = True
            logger.info("Online anomaly detector active (river HalfSpaceTrees)")
        except ImportError:
            logger.info("river not installed — online anomaly detector disabled")

    @property
    def available(self) -> bool:
        return self._available

    def score_and_learn(self, reading: Dict[str, float]) -> float:
        """
        Score the reading and update the model in one step.

        Args:
            reading: dict of parameter values
                     e.g. {"ph": 7.8, "conductivity": 2500, "temperature": 32, "orp": 650}

        Returns:
            Anomaly score 0.0 (normal) to 1.0 (anomalous).
            Returns 0.0 if river is not installed.
        """
        if not self._available or self._model is None:
            return 0.0

        try:
            # Filter to numeric values only
            clean = {k: float(v) for k, v in reading.items() if v is not None}
            if not clean:
                return 0.0

            score = self._model.score_one(clean)
            self._model.learn_one(clean)
            self._sample_count += 1
            return float(score)
        except Exception as e:
            logger.debug(f"Online detector error: {e}")
            return 0.0

    def get_status(self) -> Dict:
        """Status for dashboard."""
        return {
            "available": self._available,
            "samples_processed": self._sample_count,
            "backend": "HalfSpaceTrees" if self._available else "disabled",
        }
