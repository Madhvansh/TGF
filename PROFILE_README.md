# Madhvansh

Building autonomous industrial control systems with physics-informed ML.

## What I'm Building

I'm the solo founder of **TGF** -- an AI system that autonomously controls chemical dosing for industrial cooling towers. It replaces $50K--$200K/year human-operated systems with a $20K/year AI controller, validated on 12 years of real sensor data from Indian cement plants.

The system combines Model Predictive Control, time-series foundation models (MOMENT, Chronos-2), and classical water chemistry (Langelier/Ryznar saturation indices) into a production control loop that runs every 5 minutes with zero human intervention.

## Flagship Project

| | |
|---|---|
| [**TGF**](https://github.com/Madhvansh/TGF) | Autonomous Cooling Tower Water Treatment |
| Stack | Python, PyTorch, FastAPI, SciPy, statsmodels |
| Key Tech | MPC optimizer, MOMENT 385M anomaly detection, Chronos-2 forecasting, 5-tier safety layer, Granger causality cascade detection |
| Results | 86.2% optimal cycles, 0% critical, 78% preemptive dosing, 16.7% cost savings |
| Data | 5,614 readings from real Indian cooling towers (2013--2025) |

## Technical Focus

- Physics-informed machine learning -- hybrid models combining domain equations with neural networks
- Model Predictive Control for safety-critical industrial systems
- Time-series foundation models (MOMENT, Chronos-2)
- Industrial IoT and edge deployment (Raspberry Pi, ONNX)
- Autonomous control for process industries

## Currently

- Preparing TGF for edge deployment on Raspberry Pi 4 ($12K hardware per tower)
- Evaluating ONNX export for real-time MOMENT inference at the edge
- Open to collaborators in industrial water treatment and process control
