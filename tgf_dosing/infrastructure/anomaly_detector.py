"""
TGF Anomaly Detector
======================
Real-time anomaly detection for cooling tower sensor data.

For MVP: Uses statistical anomaly detection (Z-score, IQR, rate-of-change)
that feeds directly into the control loop. This module is the INTEGRATION
POINT where MOMENT foundation model will plug in.

Architecture:
    SensorReading → AnomalyDetector → AnomalyReport
                                          ↓
                              DosingController (adjusts behavior)
                              AlertManager (sends notifications)

Detection methods (layered defense):
1. Range check: Is the value physically possible?
2. Z-score: Is it statistically unusual given recent history?
3. Rate-of-change: Is it changing too fast?
4. Cross-parameter: Are parameter relationships broken?
5. Pattern: Does the multivariate pattern look abnormal?
   (This is where MOMENT reconstruction-based detection will go)

Each detection outputs a score 0-1 and a classification:
    NORMAL → no action
    ANOMALOUS → flag for operator attention, adjust MPC weights
    CRITICAL → trigger safety layer, alert immediately
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class ParameterAnomaly:
    """Anomaly assessment for a single parameter."""
    parameter: str
    value: float
    score: float                    # 0-1, higher = more anomalous
    classification: str             # "NORMAL", "ANOMALOUS", "CRITICAL"
    detections: List[str]           # Which methods flagged it
    details: str                    # Human-readable explanation


@dataclass
class AnomalyReport:
    """Complete anomaly assessment for one sensor reading cycle."""
    timestamp: float
    cycle_index: int
    
    # Per-parameter anomalies
    parameters: Dict[str, ParameterAnomaly]
    
    # Overall system anomaly score (0-1)
    system_score: float
    system_classification: str      # "NORMAL", "ANOMALOUS", "CRITICAL"
    
    # Cross-parameter anomalies
    cross_parameter_flags: List[str]
    
    # Recommendations for the control loop
    should_increase_safety: bool    # Tell safety layer to be more conservative
    should_reduce_dosing: bool      # Tell MPC to reduce aggressive dosing
    suspect_sensor: Optional[str]   # Which sensor might be faulty
    
    @property
    def is_anomalous(self) -> bool:
        return self.system_classification in ("ANOMALOUS", "CRITICAL")
    
    @property
    def is_critical(self) -> bool:
        return self.system_classification == "CRITICAL"


class AnomalyDetector:
    """
    Multi-method anomaly detection for cooling tower sensors.
    
    Usage:
        detector = AnomalyDetector()
        
        for reading in sensor_stream:
            report = detector.analyze(reading)
            if report.is_anomalous:
                alert_manager.process(report)
                controller.adjust_for_anomaly(report)
    """
    
    # Parameters we monitor
    MONITORED_PARAMS = ["pH", "conductivity", "temperature", "orp"]
    
    # Z-score thresholds
    Z_ANOMALOUS = 2.5   # > 2.5 sigma = anomalous
    Z_CRITICAL = 4.0     # > 4.0 sigma = critical
    
    # Rate-of-change thresholds (per 5-minute step)
    MAX_RATES = {
        "pH": 0.5,                # pH units per 5 min
        "conductivity": 500.0,    # µS/cm per 5 min
        "temperature": 3.0,       # °C per 5 min
        "orp": 100.0,             # mV per 5 min
    }
    
    # Cross-parameter expected relationships
    # If conductivity goes up, TDS should go up too
    # If temperature goes up, ORP decay accelerates
    
    def __init__(self,
                 window_size: int = 288,     # 24 hours at 5-min intervals
                 warmup_cycles: int = 50,    # Need this many readings before detecting
                 sensitivity: float = 1.0,   # 0.5=lenient, 1.0=normal, 2.0=strict
                 moment_checkpoint: str = None):  # Path to MOMENT model checkpoint
        """
        Args:
            window_size: Historical window for statistics
            warmup_cycles: Minimum readings before detection activates
            sensitivity: Multiplier for detection thresholds
            moment_checkpoint: Path to MOMENT .pt checkpoint (None = statistical only)
        """
        self.window_size = window_size
        self.warmup_cycles = warmup_cycles
        self.sensitivity = sensitivity
        
        # Rolling history for each parameter
        self.history: Dict[str, deque] = {
            p: deque(maxlen=window_size) for p in self.MONITORED_PARAMS
        }
        
        # Running statistics
        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}
        self.last_values: Dict[str, float] = {}
        
        # Anomaly tracking
        self.cycle_count = 0
        self.consecutive_anomalies: Dict[str, int] = {p: 0 for p in self.MONITORED_PARAMS}
        self.total_anomalies = 0
        self.anomaly_history: deque = deque(maxlen=1000)
        
        # MOMENT model integration
        self._moment_model = None
        self._moment_available = False
        if moment_checkpoint:
            self.load_moment_model(moment_checkpoint)
    
    def analyze(self, 
                ph: float, 
                conductivity: float, 
                temperature: float, 
                orp: float,
                timestamp: float = None,
                cycle_index: int = 0,
                tds: Optional[float] = None
                ) -> AnomalyReport:
        """
        Analyze one sensor reading for anomalies.
        
        Args:
            ph, conductivity, temperature, orp: Current sensor readings
            timestamp: Unix timestamp
            cycle_index: Cycle number
            tds: Optional TDS reading for cross-validation
        
        Returns:
            AnomalyReport with per-parameter and system-level assessments
        """
        self.cycle_count += 1
        timestamp = timestamp or time.time()
        
        values = {
            "pH": ph,
            "conductivity": conductivity,
            "temperature": temperature,
            "orp": orp,
        }
        
        # Update history
        for param, val in values.items():
            self.history[param].append(val)
        
        # Not enough data yet → everything is NORMAL
        if self.cycle_count < self.warmup_cycles:
            self.last_values = values.copy()
            self._update_statistics()
            return AnomalyReport(
                timestamp=timestamp,
                cycle_index=cycle_index,
                parameters={p: ParameterAnomaly(
                    parameter=p, value=values[p], score=0.0,
                    classification="NORMAL", detections=[], 
                    details="Warmup period"
                ) for p in self.MONITORED_PARAMS},
                system_score=0.0,
                system_classification="NORMAL",
                cross_parameter_flags=[],
                should_increase_safety=False,
                should_reduce_dosing=False,
                suspect_sensor=None,
            )
        
        # Run detection methods
        param_anomalies = {}
        for param in self.MONITORED_PARAMS:
            param_anomalies[param] = self._analyze_parameter(param, values[param])
        
        # Cross-parameter analysis
        cross_flags = self._cross_parameter_check(values, tds)
        
        # MOMENT reconstruction-based detection
        moment_score = self._moment_detect(values) if self._moment_available else 0.0

        # System-level score (blend statistical + MOMENT when available)
        param_scores = [a.score for a in param_anomalies.values()]
        if self._moment_available and moment_score > 0.0:
            # Blend: 40% statistical max, 20% statistical mean, 40% MOMENT
            system_score = max(param_scores) * 0.4 + np.mean(param_scores) * 0.2 + moment_score * 0.4
        else:
            # Statistical-only fallback
            system_score = max(param_scores) * 0.6 + np.mean(param_scores) * 0.3
        if cross_flags:
            system_score += 0.1 * len(cross_flags)
        system_score = min(1.0, system_score)
        
        if system_score > 0.7:
            system_class = "CRITICAL"
        elif system_score > 0.4:
            system_class = "ANOMALOUS"
        else:
            system_class = "NORMAL"
        
        # Recommendations
        should_increase_safety = system_class in ("ANOMALOUS", "CRITICAL")
        should_reduce_dosing = system_class == "CRITICAL"
        
        # Identify suspect sensor (the one with highest anomaly score)
        suspect = max(param_anomalies.items(), key=lambda x: x[1].score)
        suspect_sensor = suspect[0] if suspect[1].score > 0.5 else None
        
        # Track consecutive anomalies
        for param, anomaly in param_anomalies.items():
            if anomaly.classification != "NORMAL":
                self.consecutive_anomalies[param] += 1
            else:
                self.consecutive_anomalies[param] = 0
        
        if system_class != "NORMAL":
            self.total_anomalies += 1
        
        # Update state
        self.last_values = values.copy()
        self._update_statistics()
        
        report = AnomalyReport(
            timestamp=timestamp,
            cycle_index=cycle_index,
            parameters=param_anomalies,
            system_score=system_score,
            system_classification=system_class,
            cross_parameter_flags=cross_flags,
            should_increase_safety=should_increase_safety,
            should_reduce_dosing=should_reduce_dosing,
            suspect_sensor=suspect_sensor,
        )
        
        self.anomaly_history.append(report)
        
        if report.is_anomalous:
            logger.warning(
                f"Anomaly detected at cycle {cycle_index}: "
                f"score={system_score:.2f} class={system_class} "
                f"suspect={suspect_sensor} flags={cross_flags}"
            )
        
        return report
    
    def _analyze_parameter(self, param: str, value: float) -> ParameterAnomaly:
        """Run all detection methods on a single parameter."""
        detections = []
        scores = []
        details = []
        
        # Method 1: Z-score
        z_score, z_detail = self._zscore_check(param, value)
        if abs(z_score) > self.Z_ANOMALOUS / self.sensitivity:
            detections.append("z_score")
            details.append(z_detail)
        scores.append(min(1.0, abs(z_score) / self.Z_CRITICAL))
        
        # Method 2: Rate of change
        rate_score, rate_detail = self._rate_check(param, value)
        if rate_score > 0.5:
            detections.append("rate_of_change")
            details.append(rate_detail)
        scores.append(rate_score)
        
        # Method 3: Consecutive anomalies (persistence)
        if self.consecutive_anomalies.get(param, 0) >= 3:
            detections.append("persistent_anomaly")
            details.append(
                f"{param} has been anomalous for "
                f"{self.consecutive_anomalies[param]} consecutive cycles"
            )
            scores.append(0.6)
        
        # Combined score
        combined = max(scores) if scores else 0.0
        
        if combined > 0.7:
            classification = "CRITICAL"
        elif combined > 0.4:
            classification = "ANOMALOUS"
        else:
            classification = "NORMAL"
        
        return ParameterAnomaly(
            parameter=param,
            value=value,
            score=combined,
            classification=classification,
            detections=detections,
            details="; ".join(details) if details else f"{param}={value:.2f} within normal range",
        )
    
    def _zscore_check(self, param: str, value: float) -> Tuple[float, str]:
        """Z-score based anomaly detection."""
        mean = self.means.get(param, value)
        std = self.stds.get(param, 1.0)
        
        if std < 1e-6:
            return 0.0, f"{param} has no variance"
        
        z = (value - mean) / std
        detail = f"{param}={value:.2f}, z-score={z:.2f} (mean={mean:.2f}, σ={std:.2f})"
        
        return z, detail
    
    def _rate_check(self, param: str, value: float) -> Tuple[float, str]:
        """Rate-of-change anomaly detection."""
        last = self.last_values.get(param)
        if last is None:
            return 0.0, "No previous value"
        
        rate = abs(value - last)
        max_rate = self.MAX_RATES.get(param, float('inf')) / self.sensitivity
        
        if max_rate <= 0:
            return 0.0, "No rate limit defined"
        
        score = rate / max_rate
        detail = f"{param} changed by {rate:.2f} (max expected: {max_rate:.2f})"
        
        return min(1.0, score), detail
    
    def _cross_parameter_check(self, values: Dict[str, float], 
                                tds: Optional[float] = None) -> List[str]:
        """Check for broken relationships between parameters."""
        flags = []
        
        # pH vs Conductivity: if conductivity is very high but pH is low,
        # might indicate acid contamination
        if values["conductivity"] > 4000 and values["pH"] < 6.5:
            flags.append("High conductivity + low pH: possible acid contamination")
        
        # ORP vs pH: very high ORP with very high pH is unusual
        if values["orp"] > 750 and values["pH"] > 9.0:
            flags.append("High ORP + high pH: unusual combination, check sensors")
        
        # Temperature vs ORP: if temp is high but ORP is also high,
        # biocide should be decaying faster
        if values["temperature"] > 40 and values["orp"] > 700:
            flags.append("High temperature but ORP not declining: check ORP sensor")
        
        # Conductivity vs TDS consistency (if both available)
        if tds is not None and tds > 0:
            expected_tds = values["conductivity"] * 0.65
            ratio = tds / expected_tds if expected_tds > 0 else 1.0
            if ratio < 0.3 or ratio > 3.0:
                flags.append(
                    f"TDS/Conductivity mismatch: TDS={tds:.0f}, "
                    f"expected ~{expected_tds:.0f} from conductivity"
                )
        
        return flags
    
    def _update_statistics(self):
        """Update running mean and std for each parameter."""
        for param in self.MONITORED_PARAMS:
            data = list(self.history[param])
            if len(data) >= 10:
                self.means[param] = np.mean(data)
                self.stds[param] = max(np.std(data), 1e-6)
    
    # ========================================================================
    # MOMENT INTEGRATION POINT (Future)
    # ========================================================================
    
    def load_moment_model(self, model_path: str = None):
        """
        Load MOMENT foundation model for reconstruction-based anomaly detection.

        MOMENT catches complex multivariate patterns that statistical
        methods miss (e.g., subtle drift, novel failure modes).
        """
        try:
            from tgf_dosing.models.moment_detector import MomentAnomalyDetector
            self._moment_model = MomentAnomalyDetector(checkpoint_path=model_path)
            self._moment_available = True
            logger.info(f"MOMENT model loaded successfully (checkpoint: {model_path})")
        except ImportError:
            logger.warning("MOMENT model package not available (missing tgf_dosing.models.moment_detector)")
            self._moment_available = False
        except Exception as e:
            logger.warning(f"MOMENT model failed to load: {e}")
            self._moment_available = False

    def _moment_detect(self, values: Dict[str, float]) -> float:
        """
        MOMENT reconstruction-based anomaly score.
        Returns 0-1 score (higher = more anomalous).
        """
        if not self._moment_available or self._moment_model is None:
            return 0.0

        try:
            self._moment_model.add_reading(values)
            if not self._moment_model.is_ready():
                return 0.0
            global_score, _ = self._moment_model.anomaly_score(self.history)
            return float(np.clip(global_score, 0.0, 1.0))
        except Exception as e:
            logger.debug(f"MOMENT detection error: {e}")
            return 0.0
    
    # ========================================================================
    # STATISTICS & REPORTING
    # ========================================================================
    
    def get_stats(self) -> dict:
        """Get anomaly detection statistics."""
        recent = list(self.anomaly_history)[-100:] if self.anomaly_history else []
        recent_anomalous = sum(1 for r in recent if r.is_anomalous)
        
        return {
            "total_cycles_analyzed": self.cycle_count,
            "total_anomalies_detected": self.total_anomalies,
            "anomaly_rate_pct": round(self.total_anomalies / max(self.cycle_count, 1) * 100, 2),
            "recent_100_anomaly_rate_pct": round(recent_anomalous / max(len(recent), 1) * 100, 2),
            "consecutive_anomalies": dict(self.consecutive_anomalies),
            "current_means": {k: round(v, 2) for k, v in self.means.items()},
            "current_stds": {k: round(v, 2) for k, v in self.stds.items()},
            "moment_available": self._moment_available,
        }
