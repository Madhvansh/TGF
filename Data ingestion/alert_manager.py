"""
TGF Alert Manager
===================
Manages alerts, notifications, and escalation for autonomous operation.

Alert flow:
    AnomalyDetector/SafetyLayer/Controller → AlertManager → DataStore
                                                          → Console log
                                                          → (Future: SMS/Email/Webhook)

Features:
1. Deduplication: don't spam same alert repeatedly
2. Escalation: WARNING → CRITICAL if unresolved
3. Auto-resolve: clear alerts when condition resolves
4. Rate limiting: max N alerts per hour per category
5. Severity levels: INFO, WARNING, CRITICAL, EMERGENCY

Alert categories:
- anomaly: Anomaly detection triggered
- safety: Safety layer override or e-stop
- chemical: Chemical residual out of range
- sensor: Sensor fault/degradation
- system: System health (MPC failure, forecast unavailable)
- water_chemistry: LSI/RSI/CoC out of bounds
"""
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


class AlertCategory(Enum):
    ANOMALY = "anomaly"
    SAFETY = "safety"
    CHEMICAL = "chemical"
    SENSOR = "sensor"
    SYSTEM = "system"
    WATER_CHEMISTRY = "water_chemistry"


@dataclass
class Alert:
    """Single alert instance."""
    id: int
    timestamp: float
    severity: str
    category: str
    title: str
    message: str
    metadata: Dict = field(default_factory=dict)
    acknowledged: bool = False
    auto_resolved: bool = False
    resolved_at: Optional[float] = None
    
    # Dedup tracking
    dedup_key: str = ""
    occurrence_count: int = 1
    first_seen: float = 0
    last_seen: float = 0


class AlertManager:
    """
    Centralized alert management for TGF autonomous control.
    
    Usage:
        alerts = AlertManager(data_store)
        
        # From anomaly detector:
        alerts.check_anomaly(anomaly_report)
        
        # From safety layer:
        alerts.check_safety(safety_report, chemistry)
        
        # From control loop:
        alerts.check_chemistry(risk_assessment, coc)
        alerts.check_chemical_levels(tracker_snapshot)
    """
    
    # Rate limits: max alerts per category per hour
    RATE_LIMITS = {
        "anomaly": 10,
        "safety": 20,
        "chemical": 10,
        "sensor": 5,
        "system": 5,
        "water_chemistry": 10,
    }
    
    # Escalation: how long before WARNING becomes CRITICAL
    ESCALATION_MINUTES = {
        "anomaly": 30,
        "safety": 15,
        "chemical": 60,
        "sensor": 10,
        "water_chemistry": 45,
    }
    
    # Dedup window: don't repeat same alert within this many seconds
    DEDUP_WINDOW_SECONDS = 300  # 5 minutes
    
    def __init__(self, data_store=None):
        """
        Args:
            data_store: DataStore instance for persistence (optional for testing)
        """
        self.data_store = data_store
        
        # Alert counter
        self._next_id = 1
        
        # Active (unresolved) alerts
        self.active_alerts: Dict[str, Alert] = {}  # dedup_key → Alert
        
        # Rate limiting: category → list of timestamps
        self._rate_tracker: Dict[str, list] = defaultdict(list)
        
        # Alert history (in-memory, last 1000)
        self.alert_history: List[Alert] = []
        self._max_history = 1000
        
        # Callbacks for external notifications
        self._callbacks: List[Callable] = []
        
        # Stats
        self.total_alerts_created = 0
        self.total_alerts_deduplicated = 0
        self.total_alerts_auto_resolved = 0
    
    def register_callback(self, callback: Callable):
        """Register a callback for new alerts. callback(alert: Alert)"""
        self._callbacks.append(callback)
    
    # ========================================================================
    # ALERT CREATION (from various sources)
    # ========================================================================
    
    def check_anomaly(self, anomaly_report) -> Optional[Alert]:
        """Process an anomaly report and create alert if needed."""
        if not anomaly_report.is_anomalous:
            # Auto-resolve any active anomaly alerts
            self._auto_resolve("anomaly")
            return None
        
        severity = ("CRITICAL" if anomaly_report.is_critical else "WARNING")
        
        suspect = anomaly_report.suspect_sensor or "multiple"
        title = f"Anomaly detected: {anomaly_report.system_classification}"
        
        details = []
        for param, pa in anomaly_report.parameters.items():
            if pa.classification != "NORMAL":
                details.append(f"{param}={pa.value:.2f} ({pa.classification}, score={pa.score:.2f})")
        
        message = (
            f"System anomaly score: {anomaly_report.system_score:.2f}. "
            f"Suspect sensor: {suspect}. "
            f"Details: {'; '.join(details)}"
        )
        
        if anomaly_report.cross_parameter_flags:
            message += f" Cross-parameter: {'; '.join(anomaly_report.cross_parameter_flags)}"
        
        return self._create_alert(
            severity=severity,
            category="anomaly",
            title=title,
            message=message,
            dedup_key=f"anomaly_{suspect}",
            metadata={
                "system_score": anomaly_report.system_score,
                "suspect_sensor": suspect,
                "cycle_index": anomaly_report.cycle_index,
            }
        )
    
    def check_safety(self, safety_report, chemistry=None) -> List[Alert]:
        """Process safety layer report and create alerts."""
        alerts = []
        
        if safety_report.emergency_stop:
            alert = self._create_alert(
                severity="EMERGENCY",
                category="safety",
                title="EMERGENCY STOP ACTIVATED",
                message="Safety layer triggered emergency stop. All dosing halted. Manual intervention required.",
                dedup_key="emergency_stop",
                metadata={"overrides": safety_report.overrides}
            )
            if alert:
                alerts.append(alert)
        
        elif safety_report.sensor_fault:
            alert = self._create_alert(
                severity="CRITICAL",
                category="sensor",
                title="Sensor fault detected",
                message=f"Sensor readings outside physical limits. Overrides: {safety_report.overrides}",
                dedup_key="sensor_fault",
            )
            if alert:
                alerts.append(alert)
        
        elif len(safety_report.overrides) > 3:
            alert = self._create_alert(
                severity="WARNING",
                category="safety",
                title=f"Multiple safety overrides ({len(safety_report.overrides)})",
                message=f"Safety layer made {len(safety_report.overrides)} modifications to MPC output.",
                dedup_key="multi_override",
            )
            if alert:
                alerts.append(alert)
        else:
            self._auto_resolve("safety")
            self._auto_resolve("sensor")
        
        return alerts
    
    def check_chemistry(self, risk_assessment, coc: float = None) -> Optional[Alert]:
        """Check water chemistry indices and alert if out of range."""
        lsi = risk_assessment.lsi
        risk_level = risk_assessment.risk_level
        
        if risk_level == "CRITICAL":
            return self._create_alert(
                severity="CRITICAL",
                category="water_chemistry",
                title=f"CRITICAL water chemistry: {risk_assessment.details.get('cascade', '')}",
                message=(
                    f"LSI={lsi:.2f}, RSI={risk_assessment.rsi:.2f}. "
                    f"Scaling risk={risk_assessment.scaling_risk:.2f}, "
                    f"Corrosion risk={risk_assessment.corrosion_risk:.2f}, "
                    f"Biofouling risk={risk_assessment.biofouling_risk:.2f}."
                ),
                dedup_key="chemistry_critical",
                metadata={"lsi": lsi, "rsi": risk_assessment.rsi, "risk_level": risk_level}
            )
        
        elif risk_level == "HIGH":
            return self._create_alert(
                severity="WARNING",
                category="water_chemistry",
                title=f"Elevated water chemistry risk: LSI={lsi:.2f}",
                message=(
                    f"Risk level: {risk_level}. "
                    f"Primary concern: {risk_assessment.details.get('scaling', '')}."
                ),
                dedup_key="chemistry_high",
                metadata={"lsi": lsi, "risk_level": risk_level}
            )
        else:
            self._auto_resolve("water_chemistry")
            return None
    
    def check_chemical_levels(self, tracker_snapshot) -> List[Alert]:
        """Check chemical residual levels and alert on critical deficits."""
        alerts = []
        
        for name, state in tracker_snapshot.chemicals.items():
            if state.status == "CRITICAL":
                alert = self._create_alert(
                    severity="WARNING",
                    category="chemical",
                    title=f"Chemical CRITICAL: {name}",
                    message=(
                        f"{name} at {state.estimated_ppm:.1f} ppm "
                        f"(target: {state.target_ppm:.1f}, confidence: {state.confidence:.0%})"
                    ),
                    dedup_key=f"chemical_{name}",
                    metadata={"chemical": name, "ppm": state.estimated_ppm}
                )
                if alert:
                    alerts.append(alert)
            elif state.status == "OVERDOSED":
                alert = self._create_alert(
                    severity="WARNING",
                    category="chemical",
                    title=f"Chemical OVERDOSED: {name}",
                    message=f"{name} at {state.estimated_ppm:.1f} ppm (max safe level exceeded)",
                    dedup_key=f"chemical_over_{name}",
                )
                if alert:
                    alerts.append(alert)
            else:
                # Auto-resolve if chemical is back to normal
                self._auto_resolve_key(f"chemical_{name}")
        
        return alerts
    
    def system_alert(self, title: str, message: str, 
                     severity: str = "INFO") -> Optional[Alert]:
        """Create a system-level alert (MPC failure, startup, etc)."""
        return self._create_alert(
            severity=severity,
            category="system",
            title=title,
            message=message,
            dedup_key=f"system_{title[:30]}",
        )
    
    # ========================================================================
    # CORE ALERT ENGINE
    # ========================================================================
    
    def _create_alert(self,
                      severity: str,
                      category: str,
                      title: str,
                      message: str,
                      dedup_key: str = "",
                      metadata: dict = None) -> Optional[Alert]:
        """
        Create an alert with deduplication and rate limiting.
        Returns the Alert if created, None if deduplicated/rate-limited.
        """
        now = time.time()
        
        # Deduplication: if same alert is active, just increment count
        if dedup_key and dedup_key in self.active_alerts:
            existing = self.active_alerts[dedup_key]
            if now - existing.last_seen < self.DEDUP_WINDOW_SECONDS:
                existing.occurrence_count += 1
                existing.last_seen = now
                self.total_alerts_deduplicated += 1
                
                # Escalation check
                minutes_active = (now - existing.first_seen) / 60
                escalation_limit = self.ESCALATION_MINUTES.get(category, 60)
                if (minutes_active > escalation_limit and 
                    existing.severity == "WARNING"):
                    existing.severity = "CRITICAL"
                    logger.warning(f"Alert escalated to CRITICAL: {title}")
                
                return None
        
        # Rate limiting
        rate_limit = self.RATE_LIMITS.get(category, 10)
        recent = [t for t in self._rate_tracker[category] if now - t < 3600]
        self._rate_tracker[category] = recent
        
        if len(recent) >= rate_limit:
            logger.debug(f"Alert rate-limited: {category} ({len(recent)}/{rate_limit}/hr)")
            return None
        
        # Create alert
        alert = Alert(
            id=self._next_id,
            timestamp=now,
            severity=severity,
            category=category,
            title=title,
            message=message,
            metadata=metadata or {},
            dedup_key=dedup_key,
            first_seen=now,
            last_seen=now,
        )
        
        self._next_id += 1
        self.total_alerts_created += 1
        
        # Track
        if dedup_key:
            self.active_alerts[dedup_key] = alert
        self._rate_tracker[category].append(now)
        
        # History
        self.alert_history.append(alert)
        if len(self.alert_history) > self._max_history:
            self.alert_history.pop(0)
        
        # Persist
        if self.data_store:
            try:
                self.data_store.save_alert(
                    timestamp=now, severity=severity, category=category,
                    title=title, message=message, metadata=metadata
                )
            except Exception as e:
                logger.error(f"Failed to persist alert: {e}")
        
        # Notify callbacks
        for cb in self._callbacks:
            try:
                cb(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
        
        # Log
        log_fn = logger.critical if severity == "EMERGENCY" else (
            logger.warning if severity in ("WARNING", "CRITICAL") else logger.info)
        log_fn(f"[ALERT {severity}] {title}: {message}")
        
        return alert
    
    def _auto_resolve(self, category: str):
        """Auto-resolve all active alerts of a category."""
        now = time.time()
        to_remove = [
            key for key, alert in self.active_alerts.items()
            if alert.category == category
        ]
        for key in to_remove:
            alert = self.active_alerts.pop(key)
            alert.auto_resolved = True
            alert.resolved_at = now
            self.total_alerts_auto_resolved += 1
    
    def _auto_resolve_key(self, dedup_key: str):
        """Auto-resolve a specific alert by key."""
        if dedup_key in self.active_alerts:
            alert = self.active_alerts.pop(dedup_key)
            alert.auto_resolved = True
            alert.resolved_at = time.time()
            self.total_alerts_auto_resolved += 1
    
    # ========================================================================
    # QUERY INTERFACE
    # ========================================================================
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all currently active (unresolved) alerts."""
        return sorted(
            self.active_alerts.values(),
            key=lambda a: {"EMERGENCY": 0, "CRITICAL": 1, "WARNING": 2, "INFO": 3}.get(a.severity, 4)
        )
    
    def get_recent_alerts(self, count: int = 50) -> List[Alert]:
        """Get most recent alerts."""
        return self.alert_history[-count:]
    
    def acknowledge(self, alert_id: int):
        """Acknowledge an alert."""
        for alert in self.active_alerts.values():
            if alert.id == alert_id:
                alert.acknowledged = True
                if self.data_store:
                    self.data_store.acknowledge_alert(alert_id)
                return True
        return False
    
    def get_stats(self) -> dict:
        """Get alert system statistics."""
        active = self.get_active_alerts()
        return {
            "total_created": self.total_alerts_created,
            "total_deduplicated": self.total_alerts_deduplicated,
            "total_auto_resolved": self.total_alerts_auto_resolved,
            "active_count": len(active),
            "active_by_severity": {
                s: sum(1 for a in active if a.severity == s)
                for s in ["EMERGENCY", "CRITICAL", "WARNING", "INFO"]
            },
            "active_by_category": {
                c: sum(1 for a in active if a.category == c)
                for c in ["anomaly", "safety", "chemical", "sensor", "system", "water_chemistry"]
            },
        }
