# TGF -- Cooling-Tower Water-Treatment Control

Physics-informed MPC + foundation-model anomaly detection for industrial cooling towers. **In operational use since 2025 as an advisory system at eight cooling towers across two Indian plants** -- DCM Shriram Alkali and Atul Ltd (named with permission) -- where operators act on its water-chemistry risk indices, forecasts, and dosing recommendations (initially maintainer-operated remotely; self-hosted on-site at DCM Shriram Alkali, four towers, since June 2026). Autonomous closed-loop dosing is on the roadmap: TGF advises; it does not yet command dosing hardware.

## Production status & provenance

**Production status (advisory).** TGF has been in **advisory production use since
2025** at **eight cooling towers across two Indian plants — DCM Shriram Alkali and
Atul Ltd** (named with their permission). The engagement began in 2025 with a
data-gathering phase, after which the maintainer ran TGF remotely on incoming
plant data and plant teams acted on its outputs; the plants later accessed TGF
directly. Since **June 2026**, TGF has been **self-hosted on-site at DCM Shriram
Alkali (four towers)** and runs without maintainer involvement — two
deployment-confirmation letters document that deployment (see
[Deployment verification](#deployment-verification)). The **Atul Ltd** deployment
(four further towers) continues on the same advisory basis and is attested by the
maintainer; that plant does not issue public confirmations. Plant teams run TGF
on tower water data and act on its **LSI/RSI risk indices, anomaly alerts, and
forecast-informed dosing recommendations**; **a human authorizes every dose.**
TGF is **not** wired to dosing hardware — the closed-loop controller is validated
in **backtest only** (5,614 historical water-analysis records from DCM Shriram
Alkali), and autonomous actuation is on the roadmap. TGF depends on
[cooling-tower-chem](https://github.com/Madhvansh/cooling-tower-chem) for its
water-chemistry indices.

## Deployment verification

Two deployment-confirmation letters, both dated 21 July 2026 and obtained at the
maintainer's request, document the DCM Shriram Alkali deployment (four cooling
towers, self-hosted at the plant since June 2026). The letters describe TGF's use
for monitoring and analysis of tower operations; statements elsewhere in this
README about dosing recommendations, and about the Atul Ltd deployment (four
further towers), are the maintainer's own and are not covered by these letters.

- [Letter from HydroTech Services, Vapi — the supplier that provided TGF to the plant; signed by its proprietor](docs/deployment/2026-07-21-hydrotech-services-supplier-letter.pdf). (The centered header line "DCM SIGNED CONFIRMATION" appears in the letter as issued; it is not a DCM document — this letter is HydroTech's own.)
- [Letter from DCM Shriram Alkali — company-level, unsigned (titles only, no individual names)](docs/deployment/2026-07-21-dcm-shriram-alkali-company-letter.pdf). DCM has told the maintainer that its policy does not permit naming employees or the site on public documents.

To verify directly: contact the maintainer (choksimac167005@gmail.com), who will
put evaluators in touch with the plant's utility head and plant manager; they
have agreed to confirm the deployment on request.

See [EVIDENCE.md](EVIDENCE.md) for a claim-to-artifact index mapping each
load-bearing README claim to a checkable artifact.

## Production use

- **Where:** eight industrial cooling towers across two Indian plants (DCM Shriram Alkali and Atul Ltd), named with the operators' permission.
- **Since:** 2025, in three phases -- (1) 2025: remote, maintainer-operated advisory, with the maintainer running TGF on incoming plant data and plant teams acting on its outputs; (2) June 2026: self-hosted on-site at DCM Shriram Alkali (four towers), running without maintainer involvement; (3) Atul Ltd (four further towers), which continues on the same advisory basis and is maintainer-attested. TGF was developed against these plants' water-analysis data and open-sourced in 2026.
- **Verification:** deployment-confirmation letters document the DCM Shriram Alkali deployment -- see [Deployment verification](#deployment-verification).
- **Mode:** advisory / decision support. Plant teams run TGF's analysis on tower water data and act on its outputs -- LSI/RSI scaling-corrosion risk assessment, anomaly alerts, and forecast-informed dosing recommendations. Humans stay in the loop for every dose.
- **Boundary (stated plainly):** TGF has not yet been connected to dosing hardware. The closed-loop controller exists and is validated in backtest below; wiring it to pumps on real towers is the roadmap, not the present.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20RPi%20%7C%20Windows-lightgrey)

## Try it

**[Try the live demo →](https://madhvansh.github.io/TGF/demo/)** — every TGF component on a simulated tower, no install, no login.

## The Problem

India's cooling towers lose 2--5% of operational capacity annually to scaling, corrosion, and biofouling. Manual operators test water 2--3 times per day and react to problems *after* they appear, wasting 15--25% of chemicals through over-dosing. A single unplanned shutdown from scaling or corrosion costs $50K--$200K. The industry charges $50K--$200K/year for automation -- TGF targets $20K/year total cost.

## What TGF Does

By design, TGF ingests water parameters (pH and conductivity, plus temperature and ORP where instrumented) on a 5-minute loop, detects anomalies using a 385M-parameter foundation model, forecasts parameter trajectories 24 hours ahead, and optimizes chemical doses using Model Predictive Control -- all with a 5-tier safety layer that enforces hard limits by clamping. Validation to date uses historical lab water-analysis records (see the backtest results below), not live 5-minute telemetry; the released datasets carry pH and conductivity but not temperature or ORP.

```mermaid
graph TD
    A[Sensors: pH, Conductivity, Temp, ORP] --> B[Data Ingestion Pipeline]
    B --> C[5-Layer Anomaly Detection]
    C --> D{Anomaly?}
    D -->|Yes| E[Alert Manager]
    D -->|No| F[Dosing Controller]
    C --> F
    F --> G[Physics Engine: LSI / RSI / CoC]
    F --> H[Chronos-T5 Forecaster: 24h Ahead]
    F --> I[MPC Optimizer: L-BFGS-B]
    F --> J[Chemical Tracker: 7 Chemicals]
    G --> I
    H --> I
    I --> K[5-Tier Safety Layer]
    K --> L[Dosing Commands]
    L --> M[SQLite: 7 Tables]
    M --> N[FastAPI Dashboard: 18 Endpoints]
```

## Backtest results (closed-loop controller, simulated on historical data)

The advisory outputs are in production use (see above). The **closed-loop controller** metrics below are a different, stricter claim, and come from **replaying 5,614 historical water-analysis records through the controller in simulation** -- not from autonomous operation. The records are periodic lab reports (labelled roughly 2012--2025, with gaps; per-row timestamps were not preserved in the released CSV) consolidated across multiple cooling towers at DCM Shriram Alkali plants, used with the operator's permission. The manual-dosing baseline is **modeled, not measured**, and TGF has not commanded dosing on physical equipment.

| Metric (backtest) | Value |
|--------|-------|
| Cycles in optimal LSI range | **86.2%** |
| Critical-risk cycles | **0%** |
| Preemptive dosing rate | **78%** |
| Modeled chemical savings vs manual baseline | **16.7%** |
| Constraint violations (dose clamps exceeded) | **0 / 5,614** |

## Technical Architecture

### MPC Optimizer

L-BFGS-B with 24-step receding horizon (2 hours at 5-minute intervals). 10-component cost function balancing chemical cost, scaling/corrosion/biofouling risk, Chronos-T5 forecast penalties, CPCB discharge compliance, and rate smoothing. Differential evolution fallback for non-convex cases. Only the first step is executed, then re-optimized next cycle.

### MOMENT Anomaly Detection

385M-parameter pre-trained time-series foundation model ([ICML 2024](https://arxiv.org/abs/2402.03885)). Reconstruction-based anomaly detection with RevIN normalization and adaptive POT (Peaks Over Threshold) thresholding. Validated against two alternatives:

| Model | Type | Params | Val Loss | Status |
|-------|------|--------|----------|--------|
| **MOMENT** | Foundation (pre-trained) | 385M | 0.0069 | Selected |
| TransNAS-TSAD | NAS + Transformer | ~2M | 0.0145 | Evaluated |
| VTT | Variable Temporal Transformer | ~500K | 0.0203 | Evaluated |

These are internal validation figures; the trained checkpoint and raw artifacts are not yet published, so treat them as indicative rather than independently reproducible.

Detection uses 5 layers: range check, z-score, rate-of-change, cross-parameter correlation, and MOMENT reconstruction error.

### Chronos-T5 Forecaster

Amazon's zero-shot probabilistic forecasting model. Returns p10/p50/p90 quantiles for a 24-hour horizon. Enables preemptive dosing -- treating problems before they manifest. When Chronos-T5 predicts pH will exceed 8.5 in 6 hours, the MPC begins scale inhibitor dosing immediately at a lower rate, rather than reacting later at a higher rate.

### Cascade Detector

Granger-causality state machine detecting failure chains: corrosion &rarr; particles &rarr; biofilm &rarr; scale. Tests causal links between iron&rarr;turbidity, turbidity&rarr;free chlorine, and free chlorine&rarr;hardness. Left-to-right state transitions only (no backtracking).

### Physics Engine

Langelier Saturation Index (LSI) and Ryznar Stability Index (RSI) from classical water chemistry equations. Cycles of Concentration (CoC) estimated from conductivity. Virtual sensor provides physics-informed ML hybrid estimates of hardness and alkalinity when lab data is unavailable.

The water-chemistry index math has been extracted into a standalone, dependency-free library — **[cooling-tower-chem](https://github.com/Madhvansh/cooling-tower-chem)** (LSI, RSI, Puckorius PSI, Larson-Skold, cooling-tower water balance) — which TGF now uses as its single source of truth for these calculations.

### Safety Layer

5-tier defense-in-depth:

1. **Sensor fault detection** -- impossible values trigger PID fallback
2. **Hard limits** -- absolute max/min for every chemical dose
3. **Rate limiting** -- max 20% change per 5-minute cycle
4. **PID backup** -- classical controllers with anti-windup saturation
5. **Emergency stop** -- halt all dosing on multiple sensor failures

### Chemical Tracker

Mass balance for 7 Aquatech chemicals with Arrhenius temperature-dependent decay rates, Bayesian confidence scoring, and calibration from ORP sensor feedback and weekly lab tests.

## Why MPC, Not Reinforcement Learning

1. **Works with ~5K samples** -- RL needs millions of interactions
2. **Hard safety constraints** -- enforced by clamping, not soft penalties
3. **Explainable** -- "dosing because LSI forecast at +1.8 in 6 hours"
4. **Faster to a working prototype** -- RL needs months of simulation-environment development
5. **No constraint violations in backtest** during exploration

With ~5,600 historical water-analysis records and hard safety requirements for eventual industrial deployment, MPC was the principled engineering choice.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run simulation (headless, 100 cycles)
python -m tgf_dosing.main --data data/Parameters_5K.csv --cycles 100 --no-api --speed 0

# Run with dashboard (open http://localhost:8000)
python -m tgf_dosing.main --data data/Parameters_5K.csv --cycles 500 --speed 0

# Run with MOMENT anomaly detection
python -m tgf_dosing.main --data data/Parameters_5K.csv --moment-checkpoint checkpoints/moment_model.pkl

# Run tests
pytest tests/ -v
```

## Project Structure

```
tgf_dosing/                      # Production package (~10K LOC)
    config/                      # Tower specs, 7 chemical definitions
    core/                        # Physics engine, MPC, safety layer, cascade detection
    infrastructure/              # Data pipeline, anomaly detection, API, persistence
    models/                      # MOMENT wrapper, online detection
    validation/                  # Walk-forward backtester
tests/                           # 10 test files (pytest)
data/                            # Curated datasets from real Indian cooling towers
research/                        # Model experiments (MOMENT, TransNAS, VTT, Tempura)
docs/                            # Architecture, API reference, hardware guide
scripts/                         # Model download utility
```

## Roadmap

- Edge deployment on Raspberry Pi 4 ($12K hardware per tower)
- ONNX export for MOMENT inference at the edge
- Multi-tower fleet management

## Citations

- **MOMENT**: Goswami et al., "MOMENT: A Family of Open Time-series Foundation Models", ICML 2024
- **Chronos**: Ansari et al., "Chronos: Learning the Language of Time Series", Amazon, 2024
- **LSI**: Langelier, W.F., "The Analytical Control of Anti-Corrosion Water Treatment", 1936
- **RSI**: Ryznar, J.W., "A New Index for Determining Amount of Calcium Carbonate Scale Formed by a Water", 1944

## License

Apache 2.0 -- see [LICENSE](LICENSE).
