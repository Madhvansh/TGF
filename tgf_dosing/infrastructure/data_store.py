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

-- Virtual sensor predictions
CREATE TABLE IF NOT EXISTS virtual_sensor_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_index INTEGER,
    predicted_hardness REAL,
    predicted_calcium REAL,
    predicted_alkalinity REAL,
    physics_hardness REAL,
    physics_calcium REAL,
    physics_alkalinity REAL,
    confidence TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Cascade failure events
CREATE TABLE IF NOT EXISTS cascade_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_index INTEGER,
    cascade_state TEXT,
    triggered_link TEXT,
    p_value REAL,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Drift detection events
CREATE TABLE IF NOT EXISTS drift_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_index INTEGER,
    drifted_params TEXT,
    vote_count INTEGER,
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
    
    def save_virtual_sensor_log(self, cycle_index: int,
                                 predicted: dict, physics: dict,
                                 confidence: str):
        """Save virtual sensor prediction vs physics baseline."""
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO virtual_sensor_log
                   (cycle_index, predicted_hardness, predicted_calcium,
                    predicted_alkalinity, physics_hardness, physics_calcium,
                    physics_alkalinity, confidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (cycle_index,
                 predicted.get("total_hardness"), predicted.get("calcium_hardness"),
                 predicted.get("total_alkalinity"),
                 physics.get("total_hardness"), physics.get("calcium_hardness"),
                 physics.get("total_alkalinity"), confidence)
            )

    def save_cascade_event(self, cycle_index: int, state: str,
                            link: str = None, p_value: float = None):
        """Save a cascade state transition event."""
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO cascade_events
                   (cycle_index, cascade_state, triggered_link, p_value)
                   VALUES (?, ?, ?, ?)""",
                (cycle_index, state, link, p_value)
            )

    def save_drift_event(self, cycle_index: int, drifted_params: list,
                          vote_count: int):
        """Save a drift detection event."""
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO drift_events
                   (cycle_index, drifted_params, vote_count)
                   VALUES (?, ?, ?)""",
                (cycle_index, json.dumps(drifted_params), vote_count)
            )

    # ========================================================================
    # READ OPERATIONS (for Dashboard & Analytics)
    # ========================================================================
    
    def _max_timestamp(self, table: str) -> float:
        """Get the max timestamp in a table (handles historical data)."""
        with self._get_conn() as conn:
            row = conn.execute(f"SELECT MAX(timestamp) as mt FROM {table}").fetchone()
            return row['mt'] if row and row['mt'] else time.time()

    def get_recent_readings(self, hours: float = 24, limit: int = 500) -> List[dict]:
        """Get recent sensor readings (relative to data's own time range)."""
        max_ts = self._max_timestamp("sensor_readings")
        cutoff = max_ts - hours * 3600
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT * FROM sensor_readings
                   WHERE timestamp > ? ORDER BY timestamp DESC LIMIT ?""",
                (cutoff, limit)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_recent_decisions(self, hours: float = 24, limit: int = 500) -> List[dict]:
        """Get recent control decisions (relative to data's own time range)."""
        max_ts = self._max_timestamp("control_decisions")
        cutoff = max_ts - hours * 3600
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

    def get_chart_data(self, limit: int = 200) -> dict:
        """Return chart-optimized arrays for the dashboard."""
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT s.cycle_index, s.ph, s.conductivity, s.temperature, s.orp,
                          d.lsi, d.rsi, d.risk_level, d.cycle_cost_inr
                   FROM sensor_readings s
                   LEFT JOIN control_decisions d ON s.cycle_index = d.cycle_index
                   ORDER BY s.cycle_index DESC LIMIT ?""",
                (limit,)
            ).fetchall()

        if not rows:
            return {"labels": [], "ph": [], "conductivity": [], "temperature": [],
                    "orp": [], "lsi": [], "rsi": [], "risk_levels": [], "costs": []}

        rows = list(reversed(rows))  # chronological order
        return {
            "labels": [r["cycle_index"] for r in rows],
            "ph": [round(r["ph"], 2) if r["ph"] else None for r in rows],
            "conductivity": [round(r["conductivity"], 0) if r["conductivity"] else None for r in rows],
            "temperature": [round(r["temperature"], 1) if r["temperature"] else None for r in rows],
            "orp": [round(r["orp"], 0) if r["orp"] else None for r in rows],
            "lsi": [round(r["lsi"], 2) if r["lsi"] else None for r in rows],
            "rsi": [round(r["rsi"], 2) if r["rsi"] else None for r in rows],
            "risk_levels": [r["risk_level"] for r in rows],
            "costs": [round(r["cycle_cost_inr"], 0) if r["cycle_cost_inr"] else 0 for r in rows],
        }
    
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
        """Get hourly metrics (default: last 7 days, relative to data time range)."""
        max_ts = self._max_timestamp("hourly_metrics")
        cutoff = max_ts - hours * 3600
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
    
    def get_dosing_history(self, limit: int = 200) -> dict:
        """Return per-chemical dosing amounts over cycles for stacked area chart."""
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT cycle_index, chemical_name, dose_kg
                   FROM chemical_residuals
                   ORDER BY cycle_index DESC LIMIT ?""",
                (limit * 7,)  # 7 chemicals per cycle snapshot
            ).fetchall()

        if not rows:
            return {"labels": [], "chemicals": {}}

        # Group by cycle
        from collections import defaultdict
        cycles = defaultdict(dict)
        for r in rows:
            cycles[r["cycle_index"]][r["chemical_name"]] = round(r["dose_kg"] or 0, 4)

        sorted_cycles = sorted(cycles.keys())
        all_chems = set()
        for chem_dict in cycles.values():
            all_chems.update(chem_dict.keys())

        result = {"labels": sorted_cycles, "chemicals": {}}
        for chem in sorted(all_chems):
            result["chemicals"][chem] = [cycles[c].get(chem, 0) for c in sorted_cycles]
        return result

    def get_anomaly_timeline(self, limit: int = 200) -> list:
        """Return anomaly events for timeline scatter chart."""
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT cycle_index, system_score, system_classification, suspect_sensor
                   FROM anomaly_events
                   ORDER BY cycle_index DESC LIMIT ?""",
                (limit,)
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def get_simulation_summary(self) -> dict:
        """Aggregate key metrics across all tables for end-of-run summary."""
        with self._get_conn() as conn:
            # Total cycles
            total = conn.execute("SELECT COUNT(*) as cnt FROM control_decisions").fetchone()['cnt']

            # Risk distribution
            risk_rows = conn.execute(
                "SELECT risk_level, COUNT(*) as cnt FROM control_decisions GROUP BY risk_level"
            ).fetchall()
            risk_dist = {r['risk_level']: r['cnt'] for r in risk_rows}

            # Cost
            cost_row = conn.execute(
                "SELECT SUM(cycle_cost_inr) as total, MIN(cycle_cost_inr) as min_cost, "
                "MAX(cycle_cost_inr) as max_cost, AVG(cycle_cost_inr) as avg_cost "
                "FROM control_decisions"
            ).fetchone()

            # LSI stats
            lsi_row = conn.execute(
                "SELECT AVG(lsi) as avg_lsi, MIN(lsi) as min_lsi, MAX(lsi) as max_lsi "
                "FROM control_decisions"
            ).fetchone()

            # Anomalies
            anomaly_count = conn.execute(
                "SELECT COUNT(*) as cnt FROM anomaly_events"
            ).fetchone()['cnt']

            # Preemptive count
            preemptive = conn.execute(
                "SELECT COUNT(*) as cnt FROM control_decisions WHERE preemptive = 1"
            ).fetchone()['cnt']

            # Safety overrides
            safety = conn.execute(
                "SELECT SUM(safety_overrides) as cnt FROM control_decisions"
            ).fetchone()['cnt'] or 0

            # Alerts
            alert_count = conn.execute("SELECT COUNT(*) as cnt FROM alerts").fetchone()['cnt']

            return {
                "total_cycles": total,
                "risk_distribution": risk_dist,
                "cost": {
                    "total_inr": round(cost_row['total'] or 0, 0),
                    "min_inr": round(cost_row['min_cost'] or 0, 2),
                    "max_inr": round(cost_row['max_cost'] or 0, 2),
                    "avg_inr": round(cost_row['avg_cost'] or 0, 2),
                },
                "lsi": {
                    "avg": round(lsi_row['avg_lsi'] or 0, 3),
                    "min": round(lsi_row['min_lsi'] or 0, 3),
                    "max": round(lsi_row['max_lsi'] or 0, 3),
                },
                "anomalies": anomaly_count,
                "preemptive_count": preemptive,
                "preemptive_pct": round(preemptive / max(total, 1) * 100, 1),
                "safety_overrides": safety,
                "alerts": alert_count,
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
