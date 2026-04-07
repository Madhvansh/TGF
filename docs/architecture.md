# TGF System Architecture

## Five-Layer Architecture

```
Layer 5: APPLICATION
    Dashboard (FastAPI + Chart.js) | REST API (18 endpoints) | Alerting

Layer 4: AI / CONTROL
    MOMENT Anomaly Detection | Chronos-2 Forecasting | MPC Optimizer

Layer 3: COMMUNICATION
    Data Ingestion Pipeline | SQLite Persistence (7 tables, WAL mode)

Layer 2: EDGE (simulated in MVP)
    Sensor Processing | Noise Models | Dropout Simulation

Layer 1: PHYSICAL (dataset in MVP)
    pH Sensor | Conductivity | Temperature | ORP | TDS | Hardness | Alkalinity
```

## Data Flow

```
Parameters_5K.csv (5,614 rows, 18 parameters)
    |
    v
DataIngestionPipeline (CSV streaming, optional noise/dropout)
    |
    v
AnomalyDetector (5-layer: range -> z-score -> rate -> cross-param -> MOMENT)
    |                    \
    |                     --> AlertManager (dedup, escalation, auto-resolve)
    v
DosingController
    |-- PhysicsEngine (LSI/RSI/CoC calculation)
    |-- ChemicalTracker (7-chemical mass balance)
    |-- ChronosForecaster (zero-shot prediction)
    |-- MPCOptimizer (L-BFGS-B, 12-step lookahead)
    |-- SafetyLayer (hard limits, PID backup, emergency stop)
    |
    v
DataStore (SQLite: sensor_readings, control_decisions, anomaly_events,
           alerts, chemical_residuals, hourly_metrics, lab_calibrations)
    |
    v
Dashboard API (FastAPI: 18 REST endpoints + live Chart.js dashboard)
```

## Module Inventory

| Module | Path | Purpose |
|--------|------|---------|
| Main Application | `tgf_dosing/main.py` | Unified entry point, control loop orchestrator |
| Tower Config | `tgf_dosing/config/tower_config.py` | Tower specs, 7 chemical definitions |
| Settings | `tgf_dosing/config/settings.py` | Environment variable overrides |
| Physics Engine | `tgf_dosing/core/physics_engine.py` | LSI/RSI calculation, risk assessment |
| Chemical Tracker | `tgf_dosing/core/chemical_tracker.py` | Mass balance for 7 chemicals |
| Chronos Forecaster | `tgf_dosing/core/chronos_forecaster.py` | Zero-shot probabilistic forecasting |
| MPC Optimizer | `tgf_dosing/core/mpc_optimizer.py` | Model Predictive Control dosing |
| Safety Layer | `tgf_dosing/core/safety_layer.py` | PID backup + hard limits + emergency stop |
| Dosing Controller | `tgf_dosing/core/dosing_controller.py` | Orchestrates all core modules |
| Cascade Detector | `tgf_dosing/core/cascade_detector.py` | Granger causality cascade detection |
| Explainer | `tgf_dosing/core/explainer.py` | XAI for dosing decisions |
| Virtual Sensor | `tgf_dosing/core/virtual_sensor.py` | Physics-informed ML estimator |
| Data Ingestion | `tgf_dosing/infrastructure/data_ingestion.py` | CSV-to-sensor stream with noise models |
| Anomaly Detector | `tgf_dosing/infrastructure/anomaly_detector.py` | Statistical + MOMENT anomaly detection |
| Data Store | `tgf_dosing/infrastructure/data_store.py` | SQLite persistence (7 tables) |
| Alert Manager | `tgf_dosing/infrastructure/alert_manager.py` | Dedup, escalation, auto-resolve |
| Dashboard API | `tgf_dosing/infrastructure/dashboard_api.py` | FastAPI + embedded HTML dashboard |
| Drift Detector | `tgf_dosing/infrastructure/drift_detector.py` | Data drift monitoring (ADWIN) |
| MOMENT Detector | `tgf_dosing/models/moment_detector.py` | MOMENT foundation model wrapper |
| Online Detector | `tgf_dosing/models/online_detector.py` | Online anomaly detection (river) |
| Backtester | `tgf_dosing/validation/backtester.py` | Walk-forward validation |

## MOMENT Foundation Model

MOMENT (ICML 2024, Goswami et al.) is a pre-trained time-series foundation model with 385M parameters, trained on 1.13B+ observations.

Integration via `MomentAnomalyDetector`:

1. **Input**: Rolling window of 512 timesteps across all sensor channels
2. **Processing**: RevIN normalization -> Transformer encoder -> Reconstruction
3. **Output**: Per-channel reconstruction error scores
4. **Thresholding**: Adaptive POT (Peaks Over Threshold)

Score blending when MOMENT is available:
```
system_score = max(param_scores) * 0.4 + mean(param_scores) * 0.2 + moment_score * 0.4
```

Graceful degradation without MOMENT:
```
system_score = max(param_scores) * 0.6 + mean(param_scores) * 0.3
```

## MPC Optimizer

- **Algorithm**: L-BFGS-B with differential evolution fallback
- **Horizon**: 12 steps (2 hours at 5-minute intervals)
- **Execution**: Receding horizon -- only first step applied, re-optimized each cycle
- **Cost function**: Chemical cost + risk penalties + rate smoothing + forecast penalties + CPCB compliance

## Five-Layer Anomaly Detection

1. **Range check**: Physically possible values
2. **Z-score**: Statistical deviation from rolling 24-hour window
3. **Rate-of-change**: Maximum delta per 5-minute cycle
4. **Cross-parameter**: Correlation violations between parameters
5. **MOMENT reconstruction**: Deep learning pattern detection

## Safety Layer (5-Tier)

1. **Sensor fault detection**: Impossible values trigger PID fallback
2. **Hard limits**: Absolute max/min for every chemical
3. **Rate limiting**: Max 20% change per cycle
4. **PID backup**: Classical controllers with anti-windup
5. **Emergency stop**: Halt all dosing on multiple sensor failures

## Chemical Program

7 Aquatech chemicals tracked simultaneously:

| Chemical | Function | Mode |
|----------|----------|------|
| AQUATREAT-2501 | Scale + Corrosion Inhibitor | Continuous |
| AQUATREAT-1196 | Scale Inhibitor | Continuous |
| AQUATREAT-2150 | Corrosion Inhibitor | Continuous |
| AQUATREAT-3331 | Oxidizing Biocide | Slug |
| AQUATREAT-399 | Oxidizing Biocide | Alternating |
| AQUATREAT-4612 | Non-Oxidizing Biocide | Continuous |
| AQUATREAT-6625 | Dispersant | Continuous |

Each chemical tracks: target ppm, active fraction, density, half-life, Arrhenius activation energy, cost per kg, and max dose rate.
