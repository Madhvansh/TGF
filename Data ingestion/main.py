"""
TGF Main Application
======================
Unified entry point for the complete TGF Autonomous Cooling Tower Control system.

This ties together ALL components:
    Data Ingestion → Anomaly Detection → Physics Engine → Chemical Tracker →
    Chronos Forecaster → MPC Optimizer → Safety Layer → Persistence →
    Alert Manager → Dashboard API

Usage:
    # Full simulation with dashboard
    python main.py
    
    # Headless simulation (no API server)
    python main.py --no-api
    
    # Limit to N cycles
    python main.py --cycles 500
    
    # Faster simulation
    python main.py --speed 0  (as fast as possible)
    
    # With sensor noise and dropout simulation
    python main.py --noise --dropout 0.02
"""
import sys
import os
import argparse
import time
import json
import logging
import signal
from pathlib import Path
from datetime import datetime

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.tower_config import AQUATECH_850_TPD, DEFAULT_LIMITS
from core.dosing_controller import DosingController
from infrastructure.data_ingestion import DataIngestionPipeline, SensorReading
from infrastructure.anomaly_detector import AnomalyDetector
from infrastructure.data_store import DataStore
from infrastructure.alert_manager import AlertManager
from infrastructure.dashboard_api import create_api, start_api_server, FASTAPI_AVAILABLE

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("TGF_Main")

# Graceful shutdown
_shutdown = False
def _signal_handler(sig, frame):
    global _shutdown
    logger.info("Shutdown signal received...")
    _shutdown = True
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


class TGFApplication:
    """
    The complete TGF autonomous control application.
    
    This is the production-ready orchestrator that runs the full
    control loop: sensors → AI → dosing → monitoring.
    """
    
    def __init__(self, 
                 csv_path: str,
                 db_path: str = "tgf_data.db",
                 tower_config=None,
                 enable_api: bool = True,
                 api_port: int = 8000,
                 enable_noise: bool = False,
                 sensor_dropout_rate: float = 0.0,
                 enable_forecasting: bool = True):
        
        self.tower = tower_config or AQUATECH_850_TPD
        self.enable_api = enable_api
        self.api_port = api_port
        
        logger.info("=" * 60)
        logger.info("  TGF AUTONOMOUS COOLING TOWER CONTROL")
        logger.info("  AI-Driven Predictive Dosing System")
        logger.info("=" * 60)
        logger.info(f"  Tower: {self.tower.name}")
        logger.info(f"  Volume: {self.tower.holding_volume_m3} m³")
        logger.info(f"  Chemicals: {len(self.tower.chemicals)} products")
        logger.info(f"  Data source: {csv_path}")
        
        # ============================================================
        # COMPONENT INITIALIZATION
        # ============================================================
        
        # 1. Data Ingestion
        logger.info("\n[1/6] Initializing Data Ingestion Pipeline...")
        self.ingestion = DataIngestionPipeline(
            csv_path=csv_path,
            add_sensor_noise=enable_noise,
            sensor_dropout_rate=sensor_dropout_rate,
        )
        logger.info(f"  → {self.ingestion.total_readings} readings available "
                     f"({self.ingestion.estimated_duration_days:.1f} simulated days)")
        
        # 2. Anomaly Detector
        logger.info("[2/6] Initializing Anomaly Detector...")
        self.anomaly_detector = AnomalyDetector(
            window_size=288,    # 24 hours
            warmup_cycles=50,
        )
        logger.info("  → Statistical detection active (MOMENT integration point ready)")
        
        # 3. Dosing Controller (includes Physics, Chemical Tracker, Forecaster, MPC, Safety)
        logger.info("[3/6] Initializing Dosing Controller...")
        self.controller = DosingController(
            tower_config=self.tower,
            limits=DEFAULT_LIMITS,
            chronos_model_size="base",
            enable_forecasting=enable_forecasting,
        )
        logger.info("  → MPC + Safety + Forecaster initialized")
        
        # 4. Data Store
        logger.info("[4/6] Initializing Data Store...")
        self.store = DataStore(db_path=db_path)
        logger.info(f"  → SQLite database: {db_path}")
        
        # 5. Alert Manager
        logger.info("[5/6] Initializing Alert Manager...")
        self.alert_manager = AlertManager(data_store=self.store)
        
        # Register console alert callback
        self.alert_manager.register_callback(self._alert_callback)
        logger.info("  → Alert system active with dedup + escalation")
        
        # 6. Dashboard API
        self.api_app = None
        self.api_thread = None
        if enable_api and FASTAPI_AVAILABLE:
            logger.info("[6/6] Initializing Dashboard API...")
            self.api_app = create_api(
                controller=self.controller,
                data_store=self.store,
                alert_manager=self.alert_manager,
                anomaly_detector=self.anomaly_detector,
                ingestion=self.ingestion,
            )
            logger.info(f"  → API ready on port {api_port}")
        else:
            reason = "disabled" if not enable_api else "FastAPI not installed"
            logger.info(f"[6/6] Dashboard API: {reason}")
        
        # Tracking
        self.total_cycles = 0
        self.total_cost = 0.0
        self.start_time = None
        self.hourly_buffer = []
        
        logger.info("\n" + "=" * 60)
        logger.info("  ALL SYSTEMS INITIALIZED. READY TO RUN.")
        logger.info("=" * 60 + "\n")
    
    def _alert_callback(self, alert):
        """Console callback for alerts."""
        severity_icons = {
            "EMERGENCY": "🚨",
            "CRITICAL": "🔴",
            "WARNING": "🟡",
            "INFO": "ℹ️",
        }
        icon = severity_icons.get(alert.severity, "❓")
        # Already logged by AlertManager, but we could add external notification here
        # Future: SMS, email, webhook, Slack integration
    
    def run(self, 
            max_cycles: int = None,
            speed_multiplier: float = 0.0) -> dict:
        """
        Run the complete autonomous control loop.
        
        This is THE main function. It:
        1. Streams sensor data from the CSV
        2. Detects anomalies
        3. Runs the dosing control cycle
        4. Persists everything to the database
        5. Checks for alerts
        6. Updates the dashboard
        
        Args:
            max_cycles: Stop after this many cycles (None = all data)
            speed_multiplier: 0=batch, 100=100x realtime, 1=realtime
        
        Returns:
            Final simulation report
        """
        global _shutdown
        
        # Start API server
        if self.api_app is not None:
            self.api_thread = start_api_server(
                self.api_app, port=self.api_port)
            logger.info(f"Dashboard: http://localhost:{self.api_port}")
        
        # System startup alert
        self.alert_manager.system_alert(
            "System Starting",
            f"TGF autonomous control starting. "
            f"Tower: {self.tower.name}, "
            f"Readings: {self.ingestion.total_readings}",
            severity="INFO"
        )
        
        n_total = min(
            self.ingestion.total_readings,
            max_cycles or self.ingestion.total_readings
        )
        
        if self.api_app:
            self.api_app.state.simulation_running = True
            self.api_app.state.simulation_progress = {
                "current": 0, "total": n_total, "pct": 0, "rate": 0
            }
        
        self.start_time = time.time()
        last_progress_time = self.start_time
        
        # Tracking for final report
        lsi_values = []
        rsi_values = []
        risk_levels = []
        costs = []
        preemptive_count = 0
        safety_override_count = 0
        anomaly_count = 0
        chemical_status_counts = {
            name: {"ADEQUATE": 0, "LOW": 0, "CRITICAL": 0, "OVERDOSED": 0}
            for name in self.tower.chemicals
        }
        chemical_doses_total = {name: 0.0 for name in self.tower.chemicals}
        
        logger.info(f"Starting control loop: {n_total} cycles")
        
        # ============================================================
        # MAIN CONTROL LOOP
        # ============================================================
        
        for reading in self.ingestion.stream(
            speed_multiplier=speed_multiplier,
            max_readings=max_cycles
        ):
            if _shutdown:
                logger.info("Shutdown requested. Stopping gracefully...")
                break
            
            self.total_cycles += 1
            cycle = self.total_cycles
            
            # ────────────────────────────────────────────
            # STEP 1: Anomaly Detection
            # ────────────────────────────────────────────
            anomaly_report = self.anomaly_detector.analyze(
                ph=reading.ph,
                conductivity=reading.conductivity,
                temperature=reading.temperature,
                orp=reading.orp,
                timestamp=reading.timestamp,
                cycle_index=reading.cycle_index,
                tds=reading.tds,
            )
            
            if anomaly_report.is_anomalous:
                anomaly_count += 1
                self.alert_manager.check_anomaly(anomaly_report)
                
                # Persist anomaly event
                self.store.save_anomaly_event(
                    timestamp=reading.timestamp,
                    cycle_index=reading.cycle_index,
                    system_score=anomaly_report.system_score,
                    classification=anomaly_report.system_classification,
                    suspect_sensor=anomaly_report.suspect_sensor,
                )
            
            # ────────────────────────────────────────────
            # STEP 2: Run Control Cycle
            # ────────────────────────────────────────────
            try:
                result = self.controller.run_cycle(
                    ph=reading.ph,
                    conductivity=reading.conductivity,
                    temperature=reading.temperature,
                    orp=reading.orp,
                    timestamp=reading.timestamp,
                    tds=reading.tds,
                    calcium_hardness=reading.calcium_hardness,
                    total_alkalinity=reading.total_alkalinity,
                    total_hardness=reading.total_hardness,
                )
            except Exception as e:
                logger.error(f"Control cycle {cycle} failed: {e}")
                if cycle < 5:
                    raise
                continue
            
            # ────────────────────────────────────────────
            # STEP 3: Persist to Database
            # ────────────────────────────────────────────
            # Save sensor reading
            self.store.save_sensor_reading(
                timestamp=reading.timestamp,
                cycle_index=reading.cycle_index,
                ph=reading.ph,
                conductivity=reading.conductivity,
                temperature=reading.temperature,
                orp=reading.orp,
                tds=reading.tds,
                calcium_hardness=reading.calcium_hardness,
                total_hardness=reading.total_hardness,
                total_alkalinity=reading.total_alkalinity,
                has_lab_data=reading.has_lab_data,
                sensor_quality=reading.sensor_quality,
            )
            
            # Save control decision
            cycle_cost = result.total_chemical_cost_inr
            self.store.save_control_decision(
                timestamp=reading.timestamp,
                cycle_index=reading.cycle_index,
                lsi=result.risk_assessment.lsi,
                rsi=result.risk_assessment.rsi,
                risk_level=result.risk_assessment.risk_level,
                risk_score=result.risk_assessment.overall_risk,
                primary_risk=result.safe_decision.primary_risk,
                continuous_doses=result.safe_decision.continuous_doses_kg,
                slug_doses=result.safe_decision.slug_doses,
                blowdown=result.safe_decision.blowdown_command,
                preemptive=result.safe_decision.preemptive,
                opt_cost=result.mpc_decision.optimization_cost,
                converged=result.mpc_decision.optimization_converged,
                safety_overrides=len(result.safety_report.overrides),
                emergency_stop=result.safety_report.emergency_stop,
                pid_active=result.safety_report.pid_active,
                cycle_cost=cycle_cost,
            )
            
            # Save chemical residuals (every 12 cycles = hourly to save DB space)
            if cycle % 12 == 0:
                chem_states = {}
                for name, state in result.tracker_snapshot.chemicals.items():
                    dose_kg = result.safe_decision.continuous_doses_kg.get(name, 0)
                    chem_states[name] = {
                        "estimated_ppm": state.estimated_ppm,
                        "target_ppm": state.target_ppm,
                        "status": state.status,
                        "confidence": state.confidence,
                        "dose_kg": dose_kg,
                    }
                self.store.save_chemical_residuals(
                    reading.timestamp, reading.cycle_index, chem_states)
            
            # ────────────────────────────────────────────
            # STEP 4: Alert Checks
            # ────────────────────────────────────────────
            self.alert_manager.check_safety(result.safety_report)
            self.alert_manager.check_chemistry(result.risk_assessment)
            self.alert_manager.check_chemical_levels(result.tracker_snapshot)
            
            # ────────────────────────────────────────────
            # STEP 5: Lab Calibration (from dataset)
            # ────────────────────────────────────────────
            lab_data = self.ingestion.get_lab_calibration_data(reading)
            if lab_data and reading.has_lab_data:
                # Only calibrate every ~7 days worth of data
                if cycle % self.ingestion.lab_interval == 0 or cycle <= 5:
                    self.controller.calibrate_from_lab(
                        lab_results=lab_data,
                        current_conductivity=reading.conductivity,
                        timestamp=reading.timestamp,
                    )
                    self.store.save_lab_calibration(
                        reading.timestamp, reading.cycle_index,
                        lab_data, reading.conductivity,
                    )
                    logger.info(f"  Lab calibration at cycle {cycle}: {lab_data}")
            
            # ────────────────────────────────────────────
            # STEP 6: Track Metrics
            # ────────────────────────────────────────────
            lsi_values.append(result.risk_assessment.lsi)
            rsi_values.append(result.risk_assessment.rsi)
            risk_levels.append(result.risk_assessment.risk_level)
            costs.append(cycle_cost)
            self.total_cost += cycle_cost
            
            if result.safe_decision.preemptive:
                preemptive_count += 1
            if len(result.safety_report.overrides) > 0:
                safety_override_count += 1
            
            for name, state in result.tracker_snapshot.chemicals.items():
                if name in chemical_status_counts:
                    chemical_status_counts[name][state.status] = (
                        chemical_status_counts[name].get(state.status, 0) + 1)
            
            for name, kg in result.safe_decision.continuous_doses_kg.items():
                chemical_doses_total[name] = chemical_doses_total.get(name, 0) + kg
            
            # Hourly metric aggregation
            self.hourly_buffer.append({
                "ph": reading.ph, "conductivity": reading.conductivity,
                "temperature": reading.temperature, "orp": reading.orp,
                "lsi": result.risk_assessment.lsi,
                "rsi": result.risk_assessment.rsi,
                "risk_level": result.risk_assessment.risk_level,
                "cost": cycle_cost,
                "preemptive": result.safe_decision.preemptive,
                "anomaly": anomaly_report.is_anomalous,
                "timestamp": reading.timestamp,
            })
            
            if len(self.hourly_buffer) >= 12:  # Every hour
                self._flush_hourly_metrics()
            
            # ────────────────────────────────────────────
            # STEP 7: Progress Reporting
            # ────────────────────────────────────────────
            now = time.time()
            if cycle % 500 == 0 or now - last_progress_time > 5:
                elapsed = now - self.start_time
                rate = cycle / max(elapsed, 0.01)
                eta = (n_total - cycle) / max(rate, 0.1)
                pct = cycle / n_total * 100
                
                logger.info(
                    f"  [{cycle}/{n_total}] {pct:.0f}% | "
                    f"{rate:.0f} cycles/sec | ETA: {eta:.0f}s | "
                    f"Risk: {result.risk_assessment.risk_level} | "
                    f"LSI: {result.risk_assessment.lsi:.2f} | "
                    f"Cost: ₹{self.total_cost:,.0f} | "
                    f"Anomalies: {anomaly_count}"
                )
                
                if self.api_app:
                    self.api_app.state.simulation_progress = {
                        "current": cycle, "total": n_total,
                        "pct": round(pct, 1), "rate": round(rate, 0)
                    }
                
                last_progress_time = now
        
        # ============================================================
        # SIMULATION COMPLETE
        # ============================================================
        
        if self.api_app:
            self.api_app.state.simulation_running = False
        
        elapsed = time.time() - self.start_time
        
        # Flush remaining hourly metrics
        if self.hourly_buffer:
            self._flush_hourly_metrics()
        
        # Build final report
        import numpy as np
        lsi_arr = np.array(lsi_values) if lsi_values else np.array([0])
        
        report = {
            "simulation": {
                "tower": self.tower.name,
                "total_cycles": self.total_cycles,
                "elapsed_seconds": round(elapsed, 1),
                "cycles_per_second": round(self.total_cycles / max(elapsed, 0.01), 1),
                "simulated_days": round(self.total_cycles * 5 / 60 / 24, 1),
            },
            "water_chemistry": {
                "LSI": {
                    "mean": round(float(np.mean(lsi_arr)), 3),
                    "std": round(float(np.std(lsi_arr)), 3),
                    "pct_scaling": round(float(np.mean(lsi_arr > 1.5) * 100), 1),
                    "pct_corrosive": round(float(np.mean(lsi_arr < -1.0) * 100), 1),
                    "pct_optimal": round(float(np.mean((lsi_arr >= -0.5) & (lsi_arr <= 1.5)) * 100), 1),
                },
            },
            "risk_distribution": {
                level: round(risk_levels.count(level) / max(len(risk_levels), 1) * 100, 1)
                for level in ["LOW", "MODERATE", "HIGH", "CRITICAL"]
            },
            "dosing": {
                "total_chemical_cost_inr": round(self.total_cost, 0),
                "daily_avg_cost_inr": round(self.total_cost / max(self.total_cycles * 5 / 60 / 24, 0.01), 0),
                "preemptive_pct": round(preemptive_count / max(self.total_cycles, 1) * 100, 1),
                "safety_override_pct": round(safety_override_count / max(self.total_cycles, 1) * 100, 1),
                "per_chemical_kg": {
                    name: round(kg, 2) for name, kg in chemical_doses_total.items()
                },
            },
            "chemical_adequacy": {
                name: {
                    status: round(count / max(self.total_cycles, 1) * 100, 1)
                    for status, count in counts.items()
                }
                for name, counts in chemical_status_counts.items()
            },
            "anomaly_detection": {
                "total_anomalies": anomaly_count,
                "anomaly_rate_pct": round(anomaly_count / max(self.total_cycles, 1) * 100, 2),
                "detector_stats": self.anomaly_detector.get_stats(),
            },
            "alerts": self.alert_manager.get_stats(),
            "ingestion": self.ingestion.get_stats(),
            "dashboard_final_state": self.controller.get_dashboard_state(),
        }
        
        # System completion alert
        self.alert_manager.system_alert(
            "Simulation Complete",
            f"Processed {self.total_cycles} cycles in {elapsed:.1f}s. "
            f"Total cost: ₹{self.total_cost:,.0f}. "
            f"Anomalies: {anomaly_count}.",
            severity="INFO"
        )
        
        return report
    
    def _flush_hourly_metrics(self):
        """Aggregate and save hourly metrics."""
        if not self.hourly_buffer:
            return
        
        import numpy as np
        buf = self.hourly_buffer
        n = len(buf)
        
        metrics = {
            "hour_start": buf[0]["timestamp"],
            "hour_end": buf[-1]["timestamp"],
            "cycles_count": n,
            "avg_ph": round(np.mean([b["ph"] for b in buf]), 2),
            "avg_conductivity": round(np.mean([b["conductivity"] for b in buf]), 0),
            "avg_temperature": round(np.mean([b["temperature"] for b in buf]), 1),
            "avg_orp": round(np.mean([b["orp"] for b in buf]), 0),
            "avg_lsi": round(np.mean([b["lsi"] for b in buf]), 3),
            "avg_rsi": round(np.mean([b["rsi"] for b in buf]), 3),
            "low_risk_pct": round(sum(1 for b in buf if b["risk_level"] == "LOW") / n * 100, 1),
            "moderate_risk_pct": round(sum(1 for b in buf if b["risk_level"] == "MODERATE") / n * 100, 1),
            "high_risk_pct": round(sum(1 for b in buf if b["risk_level"] == "HIGH") / n * 100, 1),
            "critical_risk_pct": round(sum(1 for b in buf if b["risk_level"] == "CRITICAL") / n * 100, 1),
            "total_cost_inr": round(sum(b["cost"] for b in buf), 0),
            "preemptive_pct": round(sum(1 for b in buf if b["preemptive"]) / n * 100, 1),
            "anomaly_count": sum(1 for b in buf if b["anomaly"]),
        }
        
        try:
            self.store.save_hourly_metrics(metrics)
        except Exception as e:
            logger.error(f"Failed to save hourly metrics: {e}")
        
        self.hourly_buffer = []


def print_report(report: dict):
    """Print formatted final report."""
    print("\n" + "=" * 70)
    print("  TGF AUTONOMOUS CONTROL — SIMULATION REPORT")
    print("=" * 70)
    
    sim = report["simulation"]
    print(f"\n  Tower: {sim['tower']}")
    print(f"  Cycles: {sim['total_cycles']} ({sim['simulated_days']} simulated days)")
    print(f"  Runtime: {sim['elapsed_seconds']}s ({sim['cycles_per_second']} cycles/sec)")
    
    wc = report["water_chemistry"]
    print(f"\n  ── Water Chemistry ──")
    print(f"  LSI: {wc['LSI']['mean']:.2f} ± {wc['LSI']['std']:.2f}")
    print(f"    Scaling (LSI>1.5):    {wc['LSI']['pct_scaling']:.1f}%")
    print(f"    Corrosive (LSI<-1.0): {wc['LSI']['pct_corrosive']:.1f}%")
    print(f"    Optimal:              {wc['LSI']['pct_optimal']:.1f}%")
    
    risk = report["risk_distribution"]
    print(f"\n  ── Risk Distribution ──")
    for level in ["LOW", "MODERATE", "HIGH", "CRITICAL"]:
        bar = "█" * int(risk.get(level, 0) / 2)
        print(f"    {level:10s}: {risk.get(level, 0):5.1f}% {bar}")
    
    dosing = report["dosing"]
    print(f"\n  ── Dosing Performance ──")
    print(f"  Total cost: ₹{dosing['total_chemical_cost_inr']:,.0f}")
    print(f"  Daily avg:  ₹{dosing['daily_avg_cost_inr']:,.0f}/day")
    print(f"  Preemptive: {dosing['preemptive_pct']:.1f}% of decisions")
    print(f"  Safety overrides: {dosing['safety_override_pct']:.1f}%")
    
    print(f"\n  ── Chemical Usage (total kg) ──")
    for name, kg in dosing["per_chemical_kg"].items():
        print(f"    {name:25s}: {kg:8.2f} kg")
    
    print(f"\n  ── Chemical Adequacy ──")
    for name, status in report["chemical_adequacy"].items():
        adequate = status.get("ADEQUATE", 0)
        low = status.get("LOW", 0)
        critical = status.get("CRITICAL", 0)
        print(f"    {name:25s}: OK={adequate:5.1f}% LOW={low:5.1f}% CRIT={critical:5.1f}%")
    
    ad = report["anomaly_detection"]
    print(f"\n  ── Anomaly Detection ──")
    print(f"  Total anomalies: {ad['total_anomalies']}")
    print(f"  Anomaly rate: {ad['anomaly_rate_pct']:.2f}%")
    
    alerts = report["alerts"]
    print(f"\n  ── Alert Summary ──")
    print(f"  Created: {alerts['total_created']}")
    print(f"  Deduplicated: {alerts['total_deduplicated']}")
    print(f"  Auto-resolved: {alerts['total_auto_resolved']}")
    print(f"  Active: {alerts['active_count']}")
    
    ing = report["ingestion"]
    print(f"\n  ── Data Ingestion ──")
    print(f"  Rows processed: {ing['rows_processed']}")
    print(f"  Lab readings: {ing['lab_readings_available']}")
    synth = ing['missing_primary_sensors']
    print(f"  Temperature synthesized: {'Yes' if synth['temperature_synthesized'] > 0 else 'No'}")
    print(f"  ORP synthesized: {'Yes' if synth['orp_synthesized'] > 0 else 'No'}")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="TGF Autonomous Cooling Tower Control System")
    parser.add_argument("--data", default=None,
                       help="Path to Parameters_5K.csv")
    parser.add_argument("--cycles", type=int, default=None,
                       help="Max cycles to run (default: all)")
    parser.add_argument("--speed", type=float, default=0.0,
                       help="Simulation speed multiplier (0=max, 1=realtime, 100=100x)")
    parser.add_argument("--no-api", action="store_true",
                       help="Disable dashboard API")
    parser.add_argument("--port", type=int, default=8000,
                       help="API port (default: 8000)")
    parser.add_argument("--noise", action="store_true",
                       help="Add realistic sensor noise")
    parser.add_argument("--dropout", type=float, default=0.0,
                       help="Sensor dropout rate (0-1)")
    parser.add_argument("--db", default="tgf_data.db",
                       help="Database path")
    parser.add_argument("--no-forecast", action="store_true",
                       help="Disable Chronos forecasting")
    
    args = parser.parse_args()
    
    # Find data file
    data_paths = [
        args.data,
        'Parameters_5K.csv',
        '/mnt/project/Parameters_5K.csv',
        '/mnt/user-data/uploads/Parameters_5K.csv',
        os.path.join(os.path.dirname(__file__), 'Parameters_5K.csv'),
    ]
    
    csv_path = None
    for p in data_paths:
        if p and os.path.exists(p):
            csv_path = p
            break
    
    if csv_path is None:
        logger.error("Parameters_5K.csv not found! Searched: " + str(data_paths))
        sys.exit(1)
    
    # Create and run application
    app = TGFApplication(
        csv_path=csv_path,
        db_path=args.db,
        enable_api=not args.no_api,
        api_port=args.port,
        enable_noise=args.noise,
        sensor_dropout_rate=args.dropout,
        enable_forecasting=not args.no_forecast,
    )
    
    report = app.run(
        max_cycles=args.cycles,
        speed_multiplier=args.speed,
    )
    
    # Print report
    print_report(report)
    
    # Save report
    output_dir = '/mnt/user-data/outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, 'tgf_simulation_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Report saved to {report_path}")
    
    # If API is running, keep alive
    if app.api_thread and app.api_thread.is_alive():
        logger.info(f"\nDashboard still running at http://localhost:{args.port}")
        logger.info("Press Ctrl+C to stop.")
        try:
            while not _shutdown:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    
    return report


if __name__ == "__main__":
    main()
