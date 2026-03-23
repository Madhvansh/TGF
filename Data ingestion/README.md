# TGF Autonomous Cooling Tower Control — Complete MVP

**AI-driven predictive dosing: MPC + Chronos-2 + Anomaly Detection + Real-time Dashboard**

## Architecture (Complete)

```
Parameters_5K.csv
    │
    ▼
┌──────────────────────────────────┐
│ Data Ingestion Pipeline    [NEW] │
│ • Streams CSV as real-time       │
│   sensor feeds (5-min cycles)    │
│ • Sensor noise + dropout sim     │
│ • Lab data extraction            │
│ • Temperature/ORP synthesis      │
└──────────┬───────────────────────┘
           │
           ├──────────────────────────┐
           ▼                          ▼
┌─────────────────────┐   ┌──────────────────────────┐
│ Anomaly Detector    │   │ Chronos-2 Forecaster     │
│ [NEW]               │   │ (Zero-shot, p10/50/90)   │
│ • Z-score detection │   │ • Statistical fallback   │
│ • Rate-of-change    │   └──────────┬───────────────┘
│ • Cross-parameter   │              │
│ • MOMENT ready      │              ▼
└──────┬──────────────┘   ┌──────────────────────────┐
       │                  │ Physics Engine            │
       │                  │ (LSI/RSI/CoC/Risk)        │
       │                  └──────────┬───────────────┘
       │                             │
       ▼                             ▼
┌─────────────────────┐   ┌──────────────────────────┐
│ Alert Manager [NEW] │   │ Chemical Residual Tracker │
│ • Deduplication     │◀──│ (Mass balance, 7 chems)  │
│ • Escalation        │   └──────────┬───────────────┘
│ • Rate limiting     │              │
│ • Auto-resolve      │              ▼
└──────┬──────────────┘   ┌──────────────────────────────┐
       │                  │ MPC Dosing Optimizer           │
       │                  │ (scipy L-BFGS-B, 2-hour)      │
       │                  └──────────┬─────────────────────┘
       │                             │
       │                             ▼
       │                  ┌──────────────────────────────┐
       │                  │ Safety Layer                   │
       │                  │ (PID + limits + e-stop)        │
       │                  └──────────┬─────────────────────┘
       │                             │
       ├─────────────────────────────┤
       │                             │
       ▼                             ▼
┌─────────────────────┐   ┌──────────────────────────┐
│ SQLite DataStore    │   │ Dashboard API      [NEW] │
│ [NEW]               │   │ FastAPI + HTML dashboard  │
│ • sensor_readings   │   │ http://localhost:8000     │
│ • control_decisions │   └──────────────────────────┘
│ • anomaly_events    │
│ • alerts            │
│ • hourly_metrics    │
│ • lab_calibrations  │
└─────────────────────┘
```

## Quick Start

```bash
# Install dependencies
pip install numpy pandas scipy fastapi uvicorn

# Run full simulation with dashboard
cd tgf_dosing
python main.py

# Open dashboard at http://localhost:8000

# Options
python main.py --cycles 500          # Limit cycles
python main.py --no-api              # Headless (no web server)
python main.py --noise --dropout 0.02 # Simulate sensor issues
python main.py --speed 100           # 100x realtime speed
```

## What's New (6 modules implemented)

### 1. `infrastructure/data_ingestion.py` — Sensor Simulator
Streams Parameters_5K.csv as if 4 real sensors (pH, Conductivity, Temp, ORP) are producing readings every 5 minutes. Synthesizes temperature (Indian seasonal + diurnal) and ORP (biocide decay cycle) since the dataset lacks these. Adds configurable sensor noise and dropout simulation. Extracts lab data (hardness, alkalinity) from the dataset for periodic calibration.

### 2. `infrastructure/anomaly_detector.py` — Real-time Anomaly Detection
Multi-method anomaly detection feeding directly into the control loop. Uses Z-score, rate-of-change, and cross-parameter analysis. Outputs scores (0-1) with NORMAL/ANOMALOUS/CRITICAL classifications that adjust MPC behavior. **MOMENT foundation model integration point is ready** — the architecture accepts a `load_moment_model()` call that will plug reconstruction-based detection into the same pipeline.

### 3. `infrastructure/data_store.py` — SQLite Persistence
Seven tables capturing the complete operational state: sensor readings, control decisions, chemical residuals, anomaly events, alerts, hourly metrics, and lab calibrations. WAL mode for concurrent read performance. Every control cycle persists sensor data + MPC decisions + costs.

### 4. `infrastructure/alert_manager.py` — Alert System
Production-grade alerting with deduplication (same alert not repeated within 5 min), auto-escalation (WARNING→CRITICAL after configurable timeout), rate limiting (max alerts/hour/category), and auto-resolution (clears when condition resolves). Categories: anomaly, safety, chemical, sensor, water_chemistry, system.

### 5. `infrastructure/dashboard_api.py` — REST API + Web Dashboard
FastAPI server with 15 endpoints covering sensors, dosing, chemicals, risk, anomalies, alerts, and metrics. Includes a real-time HTML dashboard (auto-refreshing every 2 seconds) showing live sensor values, risk levels, chemical states, active alerts, and simulation progress.

### 6. `main.py` — Unified Application Entry Point
Orchestrates the complete control loop: Data Ingestion → Anomaly Detection → Dosing Control → Persistence → Alerting → Dashboard. Single command starts everything. CLI arguments for all configuration. Graceful shutdown on Ctrl+C.

## File Structure

```
tgf_dosing/
├── main.py                              ← START HERE
├── run_simulation.py                     # Legacy batch runner
├── config/
│   └── tower_config.py                   # Tower specs, 7 chemicals, limits
├── core/
│   ├── physics_engine.py                 # LSI/RSI/CoC/risk
│   ├── chemical_tracker.py               # Mass balance residual tracking
│   ├── chronos_forecaster.py             # Chronos-2 + statistical fallback
│   ├── mpc_optimizer.py                  # MPC dosing optimization
│   ├── safety_layer.py                   # PID backup + hard limits
│   └── dosing_controller.py              # Main control orchestrator
├── infrastructure/
│   ├── data_ingestion.py          [NEW]  # CSV → real-time sensor stream
│   ├── anomaly_detector.py        [NEW]  # Multi-method anomaly detection
│   ├── data_store.py              [NEW]  # SQLite persistence (7 tables)
│   ├── alert_manager.py           [NEW]  # Dedup + escalation + alerting
│   └── dashboard_api.py           [NEW]  # FastAPI REST + HTML dashboard
└── tgf_data.db                    [NEW]  # Auto-created SQLite database
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Live HTML dashboard |
| GET | `/api/status` | Current system state |
| GET | `/api/sensors/current` | Latest sensor readings |
| GET | `/api/sensors/history?hours=24` | Sensor history |
| GET | `/api/dosing/current` | Current dosing rates + reasoning |
| GET | `/api/chemicals` | Chemical residual states |
| GET | `/api/risk` | Risk assessment details |
| GET | `/api/anomalies` | Anomaly detection history |
| GET | `/api/alerts` | Active alerts |
| POST | `/api/alerts/{id}/ack` | Acknowledge an alert |
| GET | `/api/metrics/summary` | Dashboard summary stats |
| GET | `/api/metrics/hourly` | Hourly aggregated metrics |
| GET | `/api/simulation/stats` | Simulation progress |
| GET | `/api/ingestion/stats` | Data ingestion statistics |
| POST | `/api/lab/calibrate` | Submit lab calibration |
