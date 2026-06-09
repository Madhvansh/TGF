"""
TGF Dashboard API
==================
FastAPI REST API for monitoring and controlling the TGF dosing system.

Endpoints:
    GET  /                          - System status overview
    GET  /api/status                - Current system state (JSON)
    GET  /api/sensors/current       - Latest sensor readings
    GET  /api/sensors/history       - Sensor history (query params: hours, param)
    GET  /api/dosing/current        - Current dosing rates
    GET  /api/dosing/history        - Dosing decision history
    GET  /api/chemicals             - Chemical residual states
    GET  /api/risk                  - Current risk assessment
    GET  /api/anomalies             - Anomaly detection history
    GET  /api/alerts                - Active alerts
    POST /api/alerts/{id}/ack       - Acknowledge an alert
    GET  /api/metrics/hourly        - Hourly aggregated metrics
    GET  /api/metrics/summary       - Dashboard summary stats
    GET  /api/simulation/stats      - Simulation progress
    POST /api/lab/calibrate         - Submit lab calibration data

For MVP: This runs alongside the simulation/control loop in a separate thread.
"""
import json
import time
import logging
import threading
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Try to import FastAPI, fall back gracefully
try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not installed. Dashboard API unavailable. "
                   "Install with: pip install fastapi uvicorn")


def create_api(controller=None, data_store=None, alert_manager=None,
               anomaly_detector=None, ingestion=None) -> Optional[Any]:
    """
    Create the FastAPI application.
    
    Args:
        controller: DosingController instance
        data_store: DataStore instance  
        alert_manager: AlertManager instance
        anomaly_detector: AnomalyDetector instance
        ingestion: DataIngestionPipeline instance
    
    Returns:
        FastAPI app instance, or None if FastAPI unavailable
    """
    if not FASTAPI_AVAILABLE:
        return None
    
    app = FastAPI(
        title="TGF Autonomous Cooling Tower Control",
        description="AI-driven predictive dosing system dashboard",
        version="1.0.0-MVP",
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Store references
    app.state.controller = controller
    app.state.store = data_store
    app.state.alerts = alert_manager
    app.state.anomaly = anomaly_detector
    app.state.ingestion = ingestion
    app.state.simulation_running = False
    app.state.simulation_progress = {"current": 0, "total": 0, "pct": 0}
    
    # ================================================================
    # HTML DASHBOARD (minimal, functional)
    # ================================================================
    
    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        """Minimal HTML dashboard that polls the API."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>TGF Control Dashboard</title>
    <meta charset="utf-8">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Courier New', monospace; background: #0a0a0a; color: #00ff88; padding: 20px; }
        h1 { color: #00ff88; margin-bottom: 10px; font-size: 1.4em; }
        h2 { color: #00aaff; margin: 15px 0 8px; font-size: 1.1em; border-bottom: 1px solid #333; padding-bottom: 4px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 15px; }
        .card { background: #111; border: 1px solid #333; border-radius: 6px; padding: 12px; }
        .metric { display: flex; justify-content: space-between; padding: 3px 0; font-size: 0.9em; }
        .metric .label { color: #888; }
        .metric .value { color: #fff; font-weight: bold; }
        .ok { color: #00ff88; } .warn { color: #ffaa00; } .crit { color: #ff4444; }
        .alert-item { padding: 6px; margin: 4px 0; border-left: 3px solid; font-size: 0.85em; }
        .alert-EMERGENCY { border-color: #ff0000; background: #330000; }
        .alert-CRITICAL { border-color: #ff4444; background: #221111; }
        .alert-WARNING { border-color: #ffaa00; background: #222200; }
        .alert-INFO { border-color: #00aaff; background: #001122; }
        #status-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: #555; margin-right: 8px; }
        .running { background: #00ff88 !important; }
        .stopped { background: #ff4444 !important; }
        .progress { height: 4px; background: #333; margin: 8px 0; border-radius: 2px; }
        .progress-bar { height: 100%; background: #00aaff; border-radius: 2px; transition: width 0.3s; }
        pre { font-size: 0.8em; color: #aaa; white-space: pre-wrap; }
    </style>
</head>
<body>
    <h1><span id="status-dot"></span>TGF Autonomous Cooling Tower Control</h1>
    <div id="sim-progress" style="display:none;">
        <div class="progress"><div class="progress-bar" id="prog-bar"></div></div>
        <span id="prog-text" style="font-size:0.8em;color:#888;"></span>
    </div>
    
    <div class="grid" id="dashboard">
        <div class="card" id="sensors-card"><h2>Sensors</h2><div id="sensors">Loading...</div></div>
        <div class="card" id="risk-card"><h2>Risk Assessment</h2><div id="risk">Loading...</div></div>
        <div class="card" id="dosing-card"><h2>Dosing</h2><div id="dosing">Loading...</div></div>
        <div class="card" id="chemicals-card"><h2>Chemical Residuals</h2><div id="chemicals">Loading...</div></div>
        <div class="card" id="alerts-card"><h2>Active Alerts</h2><div id="alerts">Loading...</div></div>
        <div class="card" id="stats-card"><h2>System Stats</h2><div id="stats">Loading...</div></div>
    </div>
    
    <script>
    function m(label, value, cls) {
        return `<div class="metric"><span class="label">${label}</span><span class="value ${cls||''}">${value}</span></div>`;
    }
    function riskClass(level) {
        return level === 'CRITICAL' ? 'crit' : level === 'HIGH' ? 'warn' : 'ok';
    }
    
    async function update() {
        try {
            const r = await fetch('/api/status');
            const d = await r.json();
            if (!d || d.status === 'NO_DATA') { return; }
            
            document.getElementById('status-dot').className = d.status === 'RUNNING' ? 'running' : 'stopped';
            
            // Sensors
            const s = d.sensors || {};
            document.getElementById('sensors').innerHTML = 
                m('pH', (s.pH||0).toFixed(2)) +
                m('Conductivity', (s.conductivity||0).toFixed(0) + ' µS/cm') +
                m('Temperature', (s.temperature||0).toFixed(1) + ' °C') +
                m('ORP', (s.ORP||0).toFixed(0) + ' mV');
            
            // Risk
            const ri = d.risk || {};
            const idx = d.indices || {};
            document.getElementById('risk').innerHTML =
                m('Risk Level', ri.level || 'N/A', riskClass(ri.level)) +
                m('LSI', (idx.LSI||0).toFixed(2)) +
                m('RSI', (idx.RSI||0).toFixed(2)) +
                m('Primary', ri.primary || 'none') +
                m('Preemptive', ri.preemptive ? 'YES' : 'no', ri.preemptive ? 'ok' : '');
            
            // Dosing
            const dos = d.dosing || {};
            let dosHTML = m('Blowdown', (d.blowdown||0).toFixed(3));
            dosHTML += m('Cost Today', '₹' + (d.cost_today_inr||0).toLocaleString());
            for (const [k,v] of Object.entries(dos)) {
                dosHTML += m(k.replace('AQUATREAT-','AT-'), v.toFixed(4) + ' kg');
            }
            document.getElementById('dosing').innerHTML = dosHTML;
            
            // Chemicals
            const ch = d.chemicals || {};
            let chHTML = '';
            for (const [k,v] of Object.entries(ch)) {
                const st = v.status;
                const cls = st === 'CRITICAL' ? 'crit' : st === 'LOW' ? 'warn' : 'ok';
                chHTML += m(k.replace('AQUATREAT-','AT-'), 
                    v.estimated_ppm.toFixed(1) + '/' + v.target_ppm.toFixed(0) + ' ppm [' + st + ']', cls);
            }
            document.getElementById('chemicals').innerHTML = chHTML;
            
            // Stats
            document.getElementById('stats').innerHTML = 
                m('Cycle', d.cycle) +
                m('Uptime', (d.uptime_hours||0).toFixed(1) + ' hrs') +
                m('CoC', (d.water_balance?.CoC||0).toFixed(1)) +
                m('Safety Overrides', d.safety?.overrides || 0) +
                m('E-Stop', d.safety?.emergency_stop ? 'YES' : 'No', d.safety?.emergency_stop ? 'crit' : 'ok');
                
        } catch(e) { console.error(e); }
        
        // Alerts
        try {
            const ar = await fetch('/api/alerts');
            const alerts = await ar.json();
            if (alerts.length === 0) {
                document.getElementById('alerts').innerHTML = '<div style="color:#555;">No active alerts</div>';
            } else {
                document.getElementById('alerts').innerHTML = alerts.slice(0, 10).map(a =>
                    `<div class="alert-item alert-${a.severity}"><b>[${a.severity}]</b> ${a.title}</div>`
                ).join('');
            }
        } catch(e) {}
        
        // Progress
        try {
            const pr = await fetch('/api/simulation/stats');
            const pg = await pr.json();
            if (pg.running) {
                document.getElementById('sim-progress').style.display = 'block';
                document.getElementById('prog-bar').style.width = pg.pct + '%';
                document.getElementById('prog-text').textContent = 
                    `Simulation: ${pg.current}/${pg.total} (${pg.pct}%) - ${pg.rate} cycles/sec`;
            }
        } catch(e) {}
    }
    
    update();
    setInterval(update, 2000);
    </script>
</body>
</html>
"""
    
    # ================================================================
    # API ENDPOINTS
    # ================================================================
    
    @app.get("/api/status")
    async def get_status():
        """Current system state."""
        ctrl = app.state.controller
        if ctrl is None:
            return {"status": "NOT_INITIALIZED"}
        return ctrl.get_dashboard_state()
    
    @app.get("/api/sensors/current")
    async def get_current_sensors():
        """Latest sensor readings."""
        ctrl = app.state.controller
        if ctrl and ctrl.history:
            latest = ctrl.history[-1]
            return {
                "timestamp": latest.timestamp,
                "pH": latest.chemistry.ph,
                "conductivity": latest.chemistry.conductivity_us,
                "temperature": latest.chemistry.temperature_c,
                "ORP": latest.chemistry.orp_mv,
                "TDS": latest.chemistry.tds_ppm,
            }
        return {"error": "No data yet"}
    
    @app.get("/api/sensors/history")
    async def get_sensor_history(hours: float = Query(24), limit: int = Query(500)):
        """Sensor reading history."""
        store = app.state.store
        if store:
            return store.get_recent_readings(hours=hours, limit=limit)
        return []
    
    @app.get("/api/dosing/current")
    async def get_current_dosing():
        """Current dosing decisions."""
        ctrl = app.state.controller
        if ctrl and ctrl.history:
            latest = ctrl.history[-1]
            return {
                "continuous_doses_kg": latest.safe_decision.continuous_doses_kg,
                "slug_doses": latest.safe_decision.slug_doses,
                "blowdown": latest.safe_decision.blowdown_command,
                "primary_risk": latest.safe_decision.primary_risk,
                "preemptive": latest.safe_decision.preemptive,
                "reasoning": latest.safe_decision.reasoning,
                "cost_inr": latest.total_chemical_cost_inr,
            }
        return {"error": "No data yet"}
    
    @app.get("/api/dosing/history")
    async def get_dosing_history(hours: float = Query(24), limit: int = Query(500)):
        """Dosing decision history."""
        store = app.state.store
        if store:
            return store.get_recent_decisions(hours=hours, limit=limit)
        return []
    
    @app.get("/api/chemicals")
    async def get_chemicals():
        """Current chemical residual states."""
        ctrl = app.state.controller
        if ctrl and ctrl.history:
            latest = ctrl.history[-1]
            return {
                name: {
                    "estimated_ppm": state.estimated_ppm,
                    "target_ppm": state.target_ppm,
                    "status": state.status,
                    "confidence": state.confidence,
                    "deficit_ppm": state.deficit_ppm,
                }
                for name, state in latest.tracker_snapshot.chemicals.items()
            }
        return {}
    
    @app.get("/api/risk")
    async def get_risk():
        """Current risk assessment."""
        ctrl = app.state.controller
        if ctrl and ctrl.history:
            latest = ctrl.history[-1]
            r = latest.risk_assessment
            return {
                "lsi": r.lsi,
                "rsi": r.rsi,
                "scaling_risk": r.scaling_risk,
                "corrosion_risk": r.corrosion_risk,
                "biofouling_risk": r.biofouling_risk,
                "cascade_risk": r.cascade_risk,
                "overall_risk": r.overall_risk,
                "risk_level": r.risk_level,
                "details": r.details,
            }
        return {"error": "No data yet"}
    
    @app.get("/api/anomalies")
    async def get_anomalies(limit: int = Query(50)):
        """Recent anomaly events."""
        detector = app.state.anomaly
        if detector:
            return {
                "stats": detector.get_stats(),
                "recent": [
                    {
                        "timestamp": r.timestamp,
                        "cycle": r.cycle_index,
                        "score": r.system_score,
                        "classification": r.system_classification,
                        "suspect": r.suspect_sensor,
                    }
                    for r in list(detector.anomaly_history)[-limit:]
                    if r.is_anomalous
                ]
            }
        return {"stats": {}, "recent": []}
    
    @app.get("/api/alerts")
    async def get_alerts():
        """Active alerts."""
        mgr = app.state.alerts
        if mgr:
            return [
                {
                    "id": a.id,
                    "severity": a.severity,
                    "category": a.category,
                    "title": a.title,
                    "message": a.message,
                    "timestamp": a.timestamp,
                    "occurrences": a.occurrence_count,
                    "acknowledged": a.acknowledged,
                }
                for a in mgr.get_active_alerts()
            ]
        return []
    
    @app.post("/api/alerts/{alert_id}/ack")
    async def acknowledge_alert(alert_id: int):
        """Acknowledge an alert."""
        mgr = app.state.alerts
        if mgr and mgr.acknowledge(alert_id):
            return {"status": "acknowledged"}
        raise HTTPException(404, "Alert not found")
    
    @app.get("/api/metrics/summary")
    async def get_summary():
        """Dashboard summary statistics."""
        store = app.state.store
        if store:
            return store.get_dashboard_summary()
        return {}
    
    @app.get("/api/metrics/hourly")
    async def get_hourly_metrics(hours: float = Query(168)):
        """Hourly aggregated metrics."""
        store = app.state.store
        if store:
            return store.get_hourly_metrics(hours=hours)
        return []
    
    @app.get("/api/simulation/stats")
    async def get_simulation_stats():
        """Simulation progress."""
        prog = app.state.simulation_progress
        return {
            "running": app.state.simulation_running,
            "current": prog.get("current", 0),
            "total": prog.get("total", 0),
            "pct": prog.get("pct", 0),
            "rate": prog.get("rate", 0),
        }
    
    @app.get("/api/ingestion/stats")
    async def get_ingestion_stats():
        """Data ingestion statistics."""
        ing = app.state.ingestion
        if ing:
            return ing.get_stats()
        return {}
    
    @app.post("/api/lab/calibrate")
    async def lab_calibrate(lab_data: dict):
        """Submit lab calibration data."""
        ctrl = app.state.controller
        if ctrl is None:
            raise HTTPException(503, "Controller not initialized")
        
        try:
            ctrl.calibrate_from_lab(
                lab_results=lab_data.get("results", {}),
                current_conductivity=lab_data.get("conductivity", 2000),
            )
            return {"status": "calibrated", "data": lab_data}
        except Exception as e:
            raise HTTPException(500, str(e))
    
    return app


def start_api_server(app, host: str = "0.0.0.0", port: int = 8000):
    """Start the API server in a background thread."""
    if not FASTAPI_AVAILABLE or app is None:
        logger.warning("Cannot start API server: FastAPI not available")
        return None
    
    def _run():
        uvicorn.run(app, host=host, port=port, log_level="warning")
    
    thread = threading.Thread(target=_run, daemon=True, name="tgf-api")
    thread.start()
    logger.info(f"Dashboard API started at http://{host}:{port}")
    return thread
