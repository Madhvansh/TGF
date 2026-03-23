"""
TGF Data Persistence Layer
============================
SQLite-based storage for all operational data.

Tables:
1. sensor_readings: Raw sensor inputs every 5 minutes
2. control_decisions: What MPC decided + what safety approved
3. anomaly_events: Anomaly detections
4. alerts: Alert history
5. chemical_usage: Chemical dosing log
6. system_metrics: Aggregated hourly/daily metrics
7. lab_calibrations: Lab test results and calibration events

Why SQLite:
- Zero-config, single file, perfect for MVP
- Handles 100K+ rows easily
- Can migrate to PostgreSQL/TimescaleDB for production
"""
import sqlite3
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)

DB_SCHEMA = """
-- Raw sensor readings from every control cycle
CREATE TABLE IF NOT EXISTS sensor_readings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    cycle_index INTEGER NOT NULL,
    ph REAL,
    conductivity REAL,
    temperature REAL,
    orp REAL,
    tds REAL,
    calcium_hardness REAL,
    total_hardness REAL,
    total_alkalinity REAL,
    has_lab_data INTEGER DEFAULT 0,
    sensor_quality TEXT DEFAULT 'GOOD',
    created_at TEXT DEFAULT (datetime('now'))
);

-- Control decisions from each cycle
CREATE TABLE IF NOT EXISTS control_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    cycle_index INTEGER NOT NULL,
    
    -- Water chemistry indices
    lsi REAL,
    rsi REAL,
    risk_level TEXT,
    risk_score REAL,
    primary_risk TEXT,
    
    -- Dosing decisions (JSON for flexibility)
    continuous_doses_json TEXT,     -- {"AQUATREAT-2501": 0.5, ...}
    slug_doses_json TEXT,
    blowdown_command REAL,
    
    -- MPC info
    preemptive INTEGER DEFAULT 0,
    optimization_cost REAL,
    optimization_converged INTEGER DEFAULT 1,
    
    -- Safety layer
    safety_overrides INTEGER DEFAULT 0,
    emergency_stop INTEGER DEFAULT 0,
    pid_active INTEGER DEFAULT 0,
    
    -- Cost
    cycle_cost_inr REAL,
    
    created_at TEXT DEFAULT (datetime('now'))
);

-- Chemical residual tracking
CREATE TABLE IF NOT EXISTS chemical_residuals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    cycle_index INTEGER NOT NULL,
    chemical_name TEXT NOT NULL,
    estimated_ppm REAL,
    target_ppm REAL,
    status TEXT,
    confidence REAL,
    dose_kg REAL DEFAULT 0
);

-- Anomaly events
CREATE TABLE IF NOT EXISTS anomaly_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    cycle_index INTEGER NOT NULL,
    system_score REAL,
    system_classification TEXT,
    suspect_sensor TEXT,
    parameters_json TEXT,           -- Detailed per-parameter scores
    cross_parameter_flags_json TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Alerts sent to operators
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    severity TEXT NOT NULL,         -- 'INFO', 'WARNING', 'CRITICAL', 'EMERGENCY'
    category TEXT NOT NULL,         -- 'anomaly', 'safety', 'chemical', 'sensor', 'system'
    title TEXT NOT NULL,
    message TEXT NOT NULL,
    metadata_json TEXT,
    acknowledged INTEGER DEFAULT 0,
    acknowledged_at TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Aggregated metrics (hourly)
CREATE TABLE IF NOT EXISTS hourly_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hour_start REAL NOT NULL,
    hour_end REAL NOT NULL,
    cycles_count INTEGER,
    
    -- Averages
    avg_ph REAL,
    avg_conductivity REAL,
    avg_temperature REAL,
    avg_orp REAL,
    avg_lsi REAL,
    avg_rsi REAL,
    
    -- Risk distribution
    low_risk_pct REAL,
    moderate_risk_pct REAL,
    high_risk_pct REAL,
    critical_risk_pct REAL,
    
    -- Dosing
    total_cost_inr REAL,
    preemptive_pct REAL,
    anomaly_count INTEGER DEFAULT 0,
    
    created_at TEXT DEFAULT (datetime('now'))
);

-- Lab calibration events
CREATE TABLE IF NOT EXISTS lab_calibrations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    cycle_index INTEGER,
    lab_results_json TEXT,          -- {"calcium_hardness": 450, "alkalinity": 120}
    conductivity_at_calibration REAL,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Indexes for fast querying
CREATE INDEX IF NOT EXISTS idx_sensor_timestamp ON sensor_readings(timestamp);
CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON control_decisions(timestamp);
CREATE INDEX IF NOT EXISTS idx_anomaly_timestamp ON anomaly_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity, timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_unacked ON alerts(acknowledged, timestamp);
CREATE INDEX IF NOT EXISTS idx_hourly_start ON hourly_metrics(hour_start);
CREATE INDEX IF NOT EXISTS idx_residuals_cycle ON chemical_residuals(cycle_index);
"""


class DataStore:
    """
    SQLite persistence for all TGF operational data.
    
    Usage:
        store = DataStore("tgf_data.db")
        store.save_sensor_reading(reading)
        store.save_control_decision(result)
        recent = store.get_recent_readings(hours=24)
    """
    
    def __init__(self, db_path: str = "tgf_data.db"):
        self.db_path = Path(db_path)
        self._init_db()
        logger.info(f"DataStore initialized: {self.db_path}")
    
    @contextmanager
    def _get_conn(self):
        """Get a database connection with proper error handling."""
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent read performance
        conn.execute("PRAGMA synchronous=NORMAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _init_db(self):
        """Create tables if they don't exist."""
        with self._get_conn() as conn:
            conn.executescript(DB_SCHEMA)
        logger.info("Database schema initialized")
    
    # ========================================================================
    # WRITE OPERATIONS
    # ========================================================================
    
    def save_sensor_reading(self, 
                            timestamp: float, cycle_index: int,
                            ph: float, conductivity: float,
                            temperature: float, orp: float,
                            tds: float = None,
                            calcium_hardness: float = None,
                            total_hardness: float = None,
                            total_alkalinity: float = None,
                            has_lab_data: bool = False,
                            sensor_quality: str = "GOOD"):
        """Save one sensor reading."""
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO sensor_readings 
                   (timestamp, cycle_index, ph, conductivity, temperature, orp,
                    tds, calcium_hardness, total_hardness, total_alkalinity,
                    has_lab_data, sensor_quality)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (timestamp, cycle_index, ph, conductivity, temperature, orp,
                 tds, calcium_hardness, total_hardness, total_alkalinity,
                 1 if has_lab_data else 0, sensor_quality)
            )
    
    def save_control_decision(self,
                               timestamp: float, cycle_index: int,
                               lsi: float, rsi: float,
                               risk_level: str, risk_score: float,
                               primary_risk: str,
                               continuous_doses: Dict[str, float],
                               slug_doses: Dict[str, float],
                               blowdown: float,
                               preemptive: bool,
                               opt_cost: float, converged: bool,
                               safety_overrides: int, emergency_stop: bool,
                               pid_active: bool, cycle_cost: float):
        """Save one control decision."""
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO control_decisions
                   (timestamp, cycle_index, lsi, rsi, risk_level, risk_score,
                    primary_risk, continuous_doses_json, slug_doses_json,
                    blowdown_command, preemptive, optimization_cost,
                    optimization_converged, safety_overrides, emergency_stop,
                    pid_active, cycle_cost_inr)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (timestamp, cycle_index, lsi, rsi, risk_level, risk_score,
                 primary_risk, json.dumps(continuous_doses), json.dumps(slug_doses),
                 blowdown, 1 if preemptive else 0, opt_cost,
                 1 if converged else 0, safety_overrides,
                 1 if emergency_stop else 0, 1 if pid_active else 0, cycle_cost)
            )
    
    def save_chemical_residuals(self, 
                                 timestamp: float, cycle_index: int,
                                 residuals: Dict[str, dict]):
        """Save chemical residual state for all chemicals."""
        with self._get_conn() as conn:
            for name, state in residuals.items():
                conn.execute(
                    """INSERT INTO chemical_residuals
                       (timestamp, cycle_index, chemical_name, estimated_ppm,
                        target_ppm, status, confidence, dose_kg)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (timestamp, cycle_index, name,
                     state.get('estimated_ppm', 0),
                     state.get('target_ppm', 0),
                     state.get('status', 'UNKNOWN'),
                     state.get('confidence', 0),
                     state.get('dose_kg', 0))
                )
    
    def save_anomaly_event(self,
                            timestamp: float, cycle_index: int,
                            system_score: float, classification: str,
                            suspect_sensor: str = None,
                            parameters: dict = None,
                            cross_flags: list = None):
        """Save an anomaly detection event."""
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO anomaly_events
                   (timestamp, cycle_index, system_score, system_classification,
                    suspect_sensor, parameters_json, cross_parameter_flags_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (timestamp, cycle_index, system_score, classification,
                 suspect_sensor, json.dumps(parameters or {}),
                 json.dumps(cross_flags or []))
            )
    
    def save_alert(self, 
                   timestamp: float, severity: str, category: str,
                   title: str, message: str, metadata: dict = None) -> int:
        """Save an alert. Returns the alert ID."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                """INSERT INTO alerts 
                   (timestamp, severity, category, title, message, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (timestamp, severity, category, title, message,
                 json.dumps(metadata or {}))
            )
            return cursor.lastrowid
    
    def save_lab_calibration(self,
                              timestamp: float, cycle_index: int,
                              lab_results: dict, conductivity: float):
        """Save a lab calibration event."""
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO lab_calibrations
                   (timestamp, cycle_index, lab_results_json, conductivity_at_calibration)
                   VALUES (?, ?, ?, ?)""",
                (timestamp, cycle_index, json.dumps(lab_results), conductivity)
            )
    
    def save_hourly_metrics(self, metrics: dict):
        """Save aggregated hourly metrics."""
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO hourly_metrics
                   (hour_start, hour_end, cycles_count,
                    avg_ph, avg_conductivity, avg_temperature, avg_orp,
                    avg_lsi, avg_rsi,
                    low_risk_pct, moderate_risk_pct, high_risk_pct, critical_risk_pct,
                    total_cost_inr, preemptive_pct, anomaly_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (metrics.get('hour_start', 0), metrics.get('hour_end', 0),
                 metrics.get('cycles_count', 0),
                 metrics.get('avg_ph'), metrics.get('avg_conductivity'),
                 metrics.get('avg_temperature'), metrics.get('avg_orp'),
                 metrics.get('avg_lsi'), metrics.get('avg_rsi'),
                 metrics.get('low_risk_pct', 0), metrics.get('moderate_risk_pct', 0),
                 metrics.get('high_risk_pct', 0), metrics.get('critical_risk_pct', 0),
                 metrics.get('total_cost_inr', 0), metrics.get('preemptive_pct', 0),
                 metrics.get('anomaly_count', 0))
            )
    
    # ========================================================================
    # READ OPERATIONS (for Dashboard & Analytics)
    # ========================================================================
    
    def get_recent_readings(self, hours: float = 24, limit: int = 500) -> List[dict]:
        """Get recent sensor readings."""
        cutoff = time.time() - hours * 3600
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT * FROM sensor_readings 
                   WHERE timestamp > ? ORDER BY timestamp DESC LIMIT ?""",
                (cutoff, limit)
            ).fetchall()
            return [dict(r) for r in rows]
    
    def get_recent_decisions(self, hours: float = 24, limit: int = 500) -> List[dict]:
        """Get recent control decisions."""
        cutoff = time.time() - hours * 3600
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT * FROM control_decisions
                   WHERE timestamp > ? ORDER BY timestamp DESC LIMIT ?""",
                (cutoff, limit)
            ).fetchall()
            results = []
            for r in rows:
                d = dict(r)
                d['continuous_doses'] = json.loads(d.pop('continuous_doses_json', '{}'))
                d['slug_doses'] = json.loads(d.pop('slug_doses_json', '{}'))
                results.append(d)
            return results
    
    def get_unacknowledged_alerts(self) -> List[dict]:
        """Get all unacknowledged alerts."""
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT * FROM alerts WHERE acknowledged = 0 
                   ORDER BY timestamp DESC"""
            ).fetchall()
            return [dict(r) for r in rows]
    
    def acknowledge_alert(self, alert_id: int):
        """Mark an alert as acknowledged."""
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE alerts SET acknowledged = 1, acknowledged_at = datetime('now') WHERE id = ?",
                (alert_id,))
    
    def get_hourly_metrics(self, hours: float = 168) -> List[dict]:
        """Get hourly metrics (default: last 7 days)."""
        cutoff = time.time() - hours * 3600
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT * FROM hourly_metrics
                   WHERE hour_start > ? ORDER BY hour_start""",
                (cutoff,)
            ).fetchall()
            return [dict(r) for r in rows]
    
    def get_dashboard_summary(self) -> dict:
        """Get summary stats for the dashboard."""
        with self._get_conn() as conn:
            # Total cycles
            total_cycles = conn.execute(
                "SELECT COUNT(*) as cnt FROM control_decisions"
            ).fetchone()['cnt']
            
            # Recent risk distribution
            risk_dist = {}
            rows = conn.execute(
                """SELECT risk_level, COUNT(*) as cnt FROM control_decisions 
                   GROUP BY risk_level"""
            ).fetchall()
            for r in rows:
                risk_dist[r['risk_level']] = r['cnt']
            
            # Total cost
            cost = conn.execute(
                "SELECT SUM(cycle_cost_inr) as total FROM control_decisions"
            ).fetchone()['total'] or 0
            
            # Anomaly count
            anomalies = conn.execute(
                "SELECT COUNT(*) as cnt FROM anomaly_events WHERE system_classification != 'NORMAL'"
            ).fetchone()['cnt']
            
            # Unacked alerts
            unacked = conn.execute(
                "SELECT COUNT(*) as cnt FROM alerts WHERE acknowledged = 0"
            ).fetchone()['cnt']
            
            # Latest reading
            latest = conn.execute(
                "SELECT * FROM sensor_readings ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            
            return {
                "total_cycles": total_cycles,
                "risk_distribution": risk_dist,
                "total_cost_inr": round(cost, 2),
                "total_anomalies": anomalies,
                "unacknowledged_alerts": unacked,
                "latest_reading": dict(latest) if latest else None,
            }
    
    def get_chemical_usage_summary(self) -> Dict[str, dict]:
        """Get total chemical usage across all cycles."""
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT chemical_name, 
                          SUM(dose_kg) as total_kg,
                          AVG(estimated_ppm) as avg_ppm,
                          AVG(confidence) as avg_confidence
                   FROM chemical_residuals 
                   GROUP BY chemical_name"""
            ).fetchall()
            return {r['chemical_name']: dict(r) for r in rows}
