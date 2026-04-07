"""
TGF Dashboard API
==================
FastAPI REST API + investor-grade HTML dashboard for the TGF dosing system.

Endpoints:
    GET  /                          - Professional HTML dashboard with live charts
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
    GET  /api/charts/timeseries     - Chart-ready sensor + LSI arrays
    GET  /api/charts/roi            - ROI comparison (TGF vs manual baseline)
    GET  /api/charts/risk           - Risk distribution counts
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


# ============================================================================
# INVESTOR-GRADE HTML DASHBOARD
# ============================================================================

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <title>TGF - Autonomous Cooling Tower Control</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.1.0/dist/chartjs-plugin-annotation.min.js"></script>
    <style>
        :root {
            --bg: #0f1923;
            --card: #162231;
            --card-border: #1e3448;
            --accent: #0ea5e9;
            --accent2: #06b6d4;
            --positive: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --text: #e2e8f0;
            --text-dim: #94a3b8;
            --text-bright: #ffffff;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Inter', system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }

        /* Header */
        .header { background: linear-gradient(135deg, #0c1825 0%, #162231 100%); border-bottom: 1px solid var(--card-border); padding: 16px 32px; display: flex; align-items: center; justify-content: space-between; }
        .header-left { display: flex; align-items: center; gap: 16px; }
        .logo { font-size: 1.8em; font-weight: 700; color: var(--accent); letter-spacing: -0.5px; }
        .logo-sub { font-size: 0.85em; font-weight: 400; color: var(--text-dim); }
        .status-badge { display: flex; align-items: center; gap: 8px; padding: 6px 16px; border-radius: 20px; font-size: 0.8em; font-weight: 600; }
        .status-running { background: rgba(16,185,129,0.15); color: var(--positive); border: 1px solid rgba(16,185,129,0.3); }
        .status-stopped { background: rgba(239,68,68,0.15); color: var(--danger); border: 1px solid rgba(239,68,68,0.3); }
        .dot { width: 8px; height: 8px; border-radius: 50%; animation: pulse 2s infinite; }
        .dot-green { background: var(--positive); }
        .dot-red { background: var(--danger); animation: none; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

        /* Progress */
        .progress-wrap { padding: 0 32px 12px; display: none; }
        .progress-bar-bg { height: 4px; background: #1e3448; border-radius: 4px; overflow: hidden; }
        .progress-bar-fill { height: 100%; background: linear-gradient(90deg, var(--accent), var(--accent2)); border-radius: 4px; transition: width 0.5s ease; }
        .progress-text { font-size: 0.75em; color: var(--text-dim); margin-top: 4px; }

        /* Layout */
        .container { padding: 20px 32px 32px; max-width: 1600px; margin: 0 auto; }

        /* KPI Row */
        .kpi-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 20px; }
        .kpi-card { background: var(--card); border: 1px solid var(--card-border); border-radius: 12px; padding: 20px; }
        .kpi-label { font-size: 0.75em; font-weight: 500; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }
        .kpi-value { font-size: 1.8em; font-weight: 700; color: var(--text-bright); }
        .kpi-sub { font-size: 0.8em; color: var(--text-dim); margin-top: 4px; }
        .kpi-badge { display: inline-block; padding: 4px 12px; border-radius: 6px; font-size: 0.8em; font-weight: 600; }
        .badge-low { background: rgba(16,185,129,0.15); color: var(--positive); }
        .badge-moderate { background: rgba(245,158,11,0.15); color: var(--warning); }
        .badge-high { background: rgba(249,115,22,0.15); color: #f97316; }
        .badge-critical { background: rgba(239,68,68,0.15); color: var(--danger); }

        /* Chart Grid */
        .chart-row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }
        .chart-card { background: var(--card); border: 1px solid var(--card-border); border-radius: 12px; padding: 20px; }
        .chart-title { font-size: 0.85em; font-weight: 600; color: var(--text); margin-bottom: 12px; }
        .chart-wrap { position: relative; height: 260px; }

        /* Bottom panels */
        .bottom-row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }
        .panel { background: var(--card); border: 1px solid var(--card-border); border-radius: 12px; padding: 20px; }
        .panel-title { font-size: 0.85em; font-weight: 600; color: var(--text); margin-bottom: 12px; }

        /* Alerts */
        .alert-item { padding: 10px 14px; margin-bottom: 8px; border-radius: 8px; border-left: 3px solid; font-size: 0.82em; }
        .alert-EMERGENCY { border-color: #dc2626; background: rgba(220,38,38,0.1); }
        .alert-CRITICAL { border-color: var(--danger); background: rgba(239,68,68,0.08); }
        .alert-WARNING { border-color: var(--warning); background: rgba(245,158,11,0.08); }
        .alert-INFO { border-color: var(--accent); background: rgba(14,165,233,0.08); }
        .alert-sev { font-weight: 600; margin-right: 6px; }
        .alert-title { color: var(--text); }
        .alert-none { color: var(--text-dim); font-style: italic; }

        /* Stats metrics */
        .stat-row { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #1e3448; }
        .stat-row:last-child { border-bottom: none; }
        .stat-label { color: var(--text-dim); font-size: 0.85em; }
        .stat-value { color: var(--text-bright); font-weight: 600; font-size: 0.85em; }

        /* Risk bar */
        .risk-bars { display: flex; gap: 4px; height: 24px; border-radius: 6px; overflow: hidden; margin: 8px 0; }
        .risk-seg { display: flex; align-items: center; justify-content: center; font-size: 0.65em; font-weight: 600; color: #fff; min-width: 20px; }
        .risk-low { background: var(--positive); }
        .risk-mod { background: var(--warning); }
        .risk-high { background: #f97316; }
        .risk-crit { background: var(--danger); }

        /* Footer */
        .footer { text-align: center; padding: 16px 32px; color: var(--text-dim); font-size: 0.75em; border-top: 1px solid var(--card-border); }

        /* Responsive */
        @media (max-width: 1024px) { .kpi-row, .chart-row, .bottom-row { grid-template-columns: 1fr; } }
        .placeholder { color: var(--text-dim); font-size: 0.85em; text-align: center; padding: 60px 0; }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-left">
            <div><span class="logo">TGF</span> <span class="logo-sub">Autonomous Cooling Tower Control</span></div>
        </div>
        <div style="display:flex;align-items:center;gap:12px;">
            <button id="pause-btn" onclick="togglePause()" style="padding:6px 16px;border-radius:8px;border:1px solid var(--card-border);background:var(--card);color:var(--text);cursor:pointer;font-size:0.8em;font-weight:600;display:none;">Pause</button>
            <div style="display:flex;gap:8px;align-items:center;">
                <a href="/api/export/readings.csv" style="color:var(--text-dim);font-size:0.75em;text-decoration:none;padding:4px 10px;border:1px solid var(--card-border);border-radius:6px;" title="Export sensor readings">CSV</a>
                <a href="/api/export/decisions.csv" style="color:var(--text-dim);font-size:0.75em;text-decoration:none;padding:4px 10px;border:1px solid var(--card-border);border-radius:6px;" title="Export dosing decisions">Decisions</a>
            </div>
            <div id="status-badge" class="status-badge status-stopped"><span class="dot dot-red" id="dot"></span><span id="status-text">Initializing</span></div>
        </div>
    </div>

    <div class="progress-wrap" id="progress-wrap">
        <div class="progress-bar-bg"><div class="progress-bar-fill" id="prog-bar"></div></div>
        <div class="progress-text" id="prog-text"></div>
    </div>

    <div class="container">
        <!-- KPI Cards -->
        <div class="kpi-row">
            <div class="kpi-card">
                <div class="kpi-label">Risk Level</div>
                <div id="kpi-risk"><span class="kpi-badge badge-low">LOW</span></div>
                <div class="kpi-sub" id="kpi-risk-sub">Waiting for data...</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Langelier Saturation Index</div>
                <div class="kpi-value" id="kpi-lsi">--</div>
                <div class="kpi-sub" id="kpi-lsi-sub">Optimal: -0.5 to 1.5</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Cycle Cost</div>
                <div class="kpi-value" id="kpi-cost">--</div>
                <div class="kpi-sub" id="kpi-cost-sub">Per 5-min cycle</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Chemical Savings</div>
                <div class="kpi-value" style="color:var(--positive)" id="kpi-savings">--</div>
                <div class="kpi-sub" id="kpi-savings-sub">vs manual dosing baseline</div>
            </div>
        </div>

        <!-- Advanced Feature Badges -->
        <div class="kpi-row" style="margin-top:8px">
            <div class="kpi-card" style="flex:1">
                <div class="kpi-label">Virtual Sensor</div>
                <div id="badge-vs"><span class="kpi-badge badge-low">OFF</span></div>
            </div>
            <div class="kpi-card" style="flex:1">
                <div class="kpi-label">Cascade State</div>
                <div id="badge-cascade"><span class="kpi-badge badge-low">HEALTHY</span></div>
            </div>
            <div class="kpi-card" style="flex:1">
                <div class="kpi-label">Drift Status</div>
                <div id="badge-drift"><span class="kpi-badge badge-low">STABLE</span></div>
            </div>
            <div class="kpi-card" style="flex:2">
                <div class="kpi-label">Last Dosing Explanation</div>
                <div class="kpi-sub" id="dosing-explanation" style="font-size:0.85rem;min-height:2em">Waiting for data...</div>
            </div>
        </div>

        <!-- Charts Row 1: Sensors + LSI -->
        <div class="chart-row">
            <div class="chart-card">
                <div class="chart-title">Sensor Trends (Live)</div>
                <div class="chart-wrap"><canvas id="sensorChart"></canvas></div>
            </div>
            <div class="chart-card">
                <div class="chart-title">LSI / RSI Index Trend</div>
                <div class="chart-wrap"><canvas id="lsiChart"></canvas></div>
            </div>
        </div>

        <!-- Charts Row 2: Chemicals + ROI -->
        <div class="chart-row">
            <div class="chart-card">
                <div class="chart-title">Chemical Residual Levels</div>
                <div class="chart-wrap"><canvas id="chemChart"></canvas></div>
            </div>
            <div class="chart-card">
                <div class="chart-title">Cumulative Cost: TGF AI vs Manual Dosing</div>
                <div class="chart-wrap"><canvas id="roiChart"></canvas></div>
            </div>
        </div>

        <!-- Charts Row 3: Anomaly Timeline -->
        <div class="chart-row">
            <div class="chart-card" style="grid-column: 1 / -1;">
                <div class="chart-title">Anomaly Detection Timeline</div>
                <div class="chart-wrap" style="height:180px;"><canvas id="anomalyChart"></canvas></div>
            </div>
        </div>

        <!-- Bottom Row: Alerts + Stats -->
        <div class="bottom-row">
            <div class="panel">
                <div class="panel-title">Active Alerts</div>
                <div id="alerts-panel"><div class="alert-none">No active alerts</div></div>
            </div>
            <div class="panel">
                <div class="panel-title">System Statistics</div>
                <div id="risk-dist-label" style="font-size:0.75em;color:var(--text-dim);margin-bottom:4px;">Risk Distribution</div>
                <div class="risk-bars" id="risk-bars">
                    <div class="risk-seg risk-low" style="flex:1">--</div>
                </div>
                <div id="stats-panel">
                    <div class="stat-row"><span class="stat-label">Cycle</span><span class="stat-value" id="st-cycle">--</span></div>
                    <div class="stat-row"><span class="stat-label">Simulated Time</span><span class="stat-value" id="st-uptime">--</span></div>
                    <div class="stat-row"><span class="stat-label">Cycles of Concentration</span><span class="stat-value" id="st-coc">--</span></div>
                    <div class="stat-row"><span class="stat-label">Safety Overrides</span><span class="stat-value" id="st-safety">--</span></div>
                    <div class="stat-row"><span class="stat-label">Anomaly Rate</span><span class="stat-value" id="st-anomaly">--</span></div>
                    <div class="stat-row"><span class="stat-label">Preemptive Dosing</span><span class="stat-value" id="st-preemptive">--</span></div>
                </div>
            </div>
        </div>
    </div>

    <div class="footer">TGF &mdash; AI-Driven Autonomous Water Treatment &bull; Predictive Dosing with MPC Optimization &bull; Dataset Simulation</div>

<script>
// ── Chart.js global config ──
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = '#1e3448';
Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
Chart.defaults.font.size = 11;

// ── Chart instances (created on first data) ──
let sensorChart = null, lsiChart = null, chemChart = null, roiChart = null, anomalyChart = null;
let roiDataCache = null;
let isPaused = false;

function createSensorChart(data) {
    const ctx = document.getElementById('sensorChart').getContext('2d');
    sensorChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.labels,
            datasets: [
                { label: 'pH', data: data.ph, borderColor: '#0ea5e9', borderWidth: 1.5, pointRadius: 0, yAxisID: 'y', tension: 0.3 },
                { label: 'Temp (C)', data: data.temperature, borderColor: '#f59e0b', borderWidth: 1.5, pointRadius: 0, yAxisID: 'y', tension: 0.3 },
                { label: 'ORP (mV)', data: data.orp, borderColor: '#8b5cf6', borderWidth: 1.5, pointRadius: 0, yAxisID: 'y1', tension: 0.3 },
                { label: 'Cond (uS/cm)', data: data.conductivity, borderColor: '#10b981', borderWidth: 1.5, pointRadius: 0, yAxisID: 'y1', tension: 0.3 },
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: { legend: { position: 'top', labels: { boxWidth: 12, padding: 12 } } },
            scales: {
                x: { display: true, ticks: { maxTicksLimit: 8, callback: v => 'C' + v } },
                y: { position: 'left', title: { display: true, text: 'pH / Temp' }, grid: { color: '#1e3448' } },
                y1: { position: 'right', title: { display: true, text: 'ORP / Cond' }, grid: { drawOnChartArea: false } }
            }
        }
    });
}

function createLSIChart(data) {
    const ctx = document.getElementById('lsiChart').getContext('2d');
    lsiChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.labels,
            datasets: [
                { label: 'LSI', data: data.lsi, borderColor: '#06b6d4', borderWidth: 2, pointRadius: 0, tension: 0.3, fill: false },
                { label: 'RSI', data: data.rsi, borderColor: '#a855f7', borderWidth: 1.5, pointRadius: 0, tension: 0.3, borderDash: [4,4], fill: false },
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { position: 'top', labels: { boxWidth: 12, padding: 12 } },
                annotation: {
                    annotations: {
                        optZone: {
                            type: 'box', yMin: -0.5, yMax: 1.5,
                            backgroundColor: 'rgba(16,185,129,0.08)', borderColor: 'rgba(16,185,129,0.3)',
                            borderWidth: 1, label: { display: true, content: 'Optimal', position: 'start', color: '#10b981', font: { size: 10 } }
                        }
                    }
                }
            },
            scales: {
                x: { ticks: { maxTicksLimit: 8, callback: v => 'C' + v } },
                y: { title: { display: true, text: 'Index Value' }, grid: { color: '#1e3448' } }
            }
        }
    });
}

function createChemChart(chemData) {
    const ctx = document.getElementById('chemChart').getContext('2d');
    const names = Object.keys(chemData).map(n => n.replace('AQUATREAT-', 'AT-'));
    const estimated = Object.values(chemData).map(c => c.estimated_ppm);
    const targets = Object.values(chemData).map(c => c.target_ppm);
    const colors = Object.values(chemData).map(c => c.status === 'CRITICAL' ? '#ef4444' : c.status === 'LOW' ? '#f59e0b' : '#10b981');

    chemChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: names,
            datasets: [
                { label: 'Current (ppm)', data: estimated, backgroundColor: colors, borderRadius: 4 },
                { label: 'Target (ppm)', data: targets, backgroundColor: 'rgba(148,163,184,0.25)', borderRadius: 4 },
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false, indexAxis: 'y',
            plugins: { legend: { position: 'top', labels: { boxWidth: 12, padding: 12 } } },
            scales: {
                x: { title: { display: true, text: 'ppm' }, grid: { color: '#1e3448' } },
                y: { grid: { display: false } }
            }
        }
    });
}

function createROIChart(roi) {
    const ctx = document.getElementById('roiChart').getContext('2d');
    roiChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: roi.labels,
            datasets: [
                { label: 'TGF AI Dosing', data: roi.tgf_costs, borderColor: '#0ea5e9', borderWidth: 2, pointRadius: 0, fill: true, backgroundColor: 'rgba(14,165,233,0.08)', tension: 0.3 },
                { label: 'Manual Dosing (est.)', data: roi.baseline_costs, borderColor: '#ef4444', borderWidth: 2, pointRadius: 0, borderDash: [6,3], fill: false, tension: 0.3 },
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: { legend: { position: 'top', labels: { boxWidth: 12, padding: 12 } } },
            scales: {
                x: { ticks: { maxTicksLimit: 8, callback: v => 'C' + v } },
                y: { title: { display: true, text: 'Cumulative Cost (INR)' }, grid: { color: '#1e3448' }, ticks: { callback: v => v >= 1000 ? (v/1000).toFixed(0) + 'K' : v } }
            }
        }
    });
}

// ── Update functions ──
function updateCharts(data) {
    if (!data || !data.labels || data.labels.length === 0) return;
    if (!sensorChart) { createSensorChart(data); createLSIChart(data); }
    else {
        [sensorChart, lsiChart].forEach(c => { c.data.labels = data.labels; });
        sensorChart.data.datasets[0].data = data.ph;
        sensorChart.data.datasets[1].data = data.temperature;
        sensorChart.data.datasets[2].data = data.orp;
        sensorChart.data.datasets[3].data = data.conductivity;
        lsiChart.data.datasets[0].data = data.lsi;
        lsiChart.data.datasets[1].data = data.rsi;
        sensorChart.update('none');
        lsiChart.update('none');
    }
}

function updateROI(roi) {
    if (!roi || !roi.labels || roi.labels.length < 2) return;
    roiDataCache = roi;
    if (!roiChart) { createROIChart(roi); }
    else {
        roiChart.data.labels = roi.labels;
        roiChart.data.datasets[0].data = roi.tgf_costs;
        roiChart.data.datasets[1].data = roi.baseline_costs;
        roiChart.update('none');
    }
}

function updateChemicals(chemData) {
    if (!chemData || Object.keys(chemData).length === 0) return;
    if (!chemChart) { createChemChart(chemData); }
    else {
        const est = Object.values(chemData).map(c => c.estimated_ppm);
        const tgt = Object.values(chemData).map(c => c.target_ppm);
        const cols = Object.values(chemData).map(c => c.status === 'CRITICAL' ? '#ef4444' : c.status === 'LOW' ? '#f59e0b' : '#10b981');
        chemChart.data.datasets[0].data = est;
        chemChart.data.datasets[0].backgroundColor = cols;
        chemChart.data.datasets[1].data = tgt;
        chemChart.update('none');
    }
}

function createAnomalyChart(data) {
    const ctx = document.getElementById('anomalyChart').getContext('2d');
    const colors = data.map(d => d.system_classification === 'CRITICAL' ? '#ef4444' : '#f59e0b');
    anomalyChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Anomaly Score',
                data: data.map(d => ({x: d.cycle_index, y: d.system_score})),
                backgroundColor: colors,
                pointRadius: 4,
                pointHoverRadius: 6,
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => {
                            const d = data[ctx.dataIndex];
                            return d ? d.system_classification + ' (score: ' + (d.system_score||0).toFixed(2) + ') - ' + (d.suspect_sensor || 'multi') : '';
                        }
                    }
                }
            },
            scales: {
                x: { title: { display: true, text: 'Cycle' }, grid: { color: '#1e3448' } },
                y: { title: { display: true, text: 'Score' }, min: 0, max: 1, grid: { color: '#1e3448' } }
            }
        }
    });
}

function updateAnomalyChart(data) {
    if (!data || data.length === 0) return;
    if (!anomalyChart) { createAnomalyChart(data); return; }
    const colors = data.map(d => d.system_classification === 'CRITICAL' ? '#ef4444' : '#f59e0b');
    anomalyChart.data.datasets[0].data = data.map(d => ({x: d.cycle_index, y: d.system_score}));
    anomalyChart.data.datasets[0].backgroundColor = colors;
    anomalyChart.update('none');
}

async function togglePause() {
    const endpoint = isPaused ? '/api/control/resume' : '/api/control/pause';
    try {
        await fetch(endpoint, {method: 'POST'});
        isPaused = !isPaused;
        document.getElementById('pause-btn').textContent = isPaused ? 'Resume' : 'Pause';
        document.getElementById('pause-btn').style.borderColor = isPaused ? 'var(--warning)' : 'var(--card-border)';
    } catch(e) {}
}

function badgeClass(level) {
    const l = (level || '').toUpperCase();
    if (l === 'CRITICAL') return 'badge-critical';
    if (l === 'HIGH') return 'badge-high';
    if (l === 'MODERATE') return 'badge-moderate';
    return 'badge-low';
}

async function update() {
    // Status
    try {
        const r = await fetch('/api/status');
        const d = await r.json();
        if (!d || d.status === 'NOT_INITIALIZED' || d.status === 'NO_DATA') return;

        const running = d.status === 'RUNNING';
        const badge = document.getElementById('status-badge');
        const dot = document.getElementById('dot');
        badge.className = 'status-badge ' + (running ? 'status-running' : 'status-stopped');
        dot.className = 'dot ' + (running ? 'dot-green' : 'dot-red');
        document.getElementById('status-text').textContent = running ? 'Running' : 'Complete';

        // KPIs
        const ri = d.risk || {};
        const idx = d.indices || {};
        document.getElementById('kpi-risk').innerHTML = '<span class="kpi-badge ' + badgeClass(ri.level) + '">' + (ri.level || 'N/A') + '</span>';
        document.getElementById('kpi-risk-sub').textContent = 'Primary: ' + (ri.primary || 'none');
        document.getElementById('kpi-lsi').textContent = (idx.LSI || 0).toFixed(2);
        const lsiVal = idx.LSI || 0;
        document.getElementById('kpi-lsi-sub').textContent = (lsiVal >= -0.5 && lsiVal <= 1.5) ? 'Within optimal range' : 'Outside optimal range';
        document.getElementById('kpi-cost').textContent = '\u20B9' + Math.round(d.cost_today_inr || 0).toLocaleString();

        // Stats panel
        document.getElementById('st-cycle').textContent = d.cycle || '--';
        document.getElementById('st-uptime').textContent = ((d.uptime_hours || 0) * 60).toFixed(0) + ' min (' + (d.uptime_hours || 0).toFixed(1) + ' hrs)';
        document.getElementById('st-coc').textContent = (d.water_balance?.CoC || 0).toFixed(1);
        document.getElementById('st-safety').textContent = d.safety?.overrides || 0;

        // Chemicals
        updateChemicals(d.chemicals);

    } catch(e) { console.error('status', e); }

    // Chart timeseries
    try {
        const r = await fetch('/api/charts/timeseries?limit=200');
        const data = await r.json();
        updateCharts(data);
    } catch(e) { console.error('charts', e); }

    // ROI
    try {
        const r = await fetch('/api/charts/roi');
        const roi = await r.json();
        updateROI(roi);
        if (roi.savings_pct != null) {
            document.getElementById('kpi-savings').textContent = roi.savings_pct.toFixed(1) + '%';
            document.getElementById('kpi-savings-sub').textContent = 'Saved \u20B9' + Math.round(roi.savings_inr || 0).toLocaleString() + ' vs manual';
        }
    } catch(e) {}

    // Risk distribution
    try {
        const r = await fetch('/api/charts/risk');
        const rd = await r.json();
        const total = (rd.LOW||0)+(rd.MODERATE||0)+(rd.HIGH||0)+(rd.CRITICAL||0);
        if (total > 0) {
            let html = '';
            if (rd.LOW) html += '<div class="risk-seg risk-low" style="flex:' + rd.LOW + '">' + Math.round(rd.LOW/total*100) + '%</div>';
            if (rd.MODERATE) html += '<div class="risk-seg risk-mod" style="flex:' + rd.MODERATE + '">' + Math.round(rd.MODERATE/total*100) + '%</div>';
            if (rd.HIGH) html += '<div class="risk-seg risk-high" style="flex:' + rd.HIGH + '">' + Math.round(rd.HIGH/total*100) + '%</div>';
            if (rd.CRITICAL) html += '<div class="risk-seg risk-crit" style="flex:' + rd.CRITICAL + '">' + Math.round(rd.CRITICAL/total*100) + '%</div>';
            document.getElementById('risk-bars').innerHTML = html;
        }
        // Anomaly rate
        if (rd.anomaly_rate_pct != null) document.getElementById('st-anomaly').textContent = rd.anomaly_rate_pct.toFixed(1) + '%';
        if (rd.preemptive_pct != null) document.getElementById('st-preemptive').textContent = rd.preemptive_pct.toFixed(1) + '%';
    } catch(e) {}

    // Alerts
    try {
        const r = await fetch('/api/alerts');
        const alerts = await r.json();
        const el = document.getElementById('alerts-panel');
        if (!alerts || alerts.length === 0) {
            el.innerHTML = '<div class="alert-none">No active alerts</div>';
        } else {
            el.innerHTML = alerts.slice(0, 6).map(a =>
                '<div class="alert-item alert-' + a.severity + '">' +
                '<span class="alert-sev">[' + a.severity + ']</span>' +
                '<span class="alert-title">' + a.title + '</span></div>'
            ).join('');
        }
    } catch(e) {}

    // Anomaly timeline
    try {
        const r = await fetch('/api/charts/anomalies?limit=200');
        const data = await r.json();
        updateAnomalyChart(data);
    } catch(e) {}

    // Progress
    try {
        const r = await fetch('/api/simulation/stats');
        const pg = await r.json();
        const wrap = document.getElementById('progress-wrap');
        const pauseBtn = document.getElementById('pause-btn');
        if (pg.running) {
            wrap.style.display = 'block';
            pauseBtn.style.display = 'inline-block';
            const elapsed = pg.rate > 0 ? Math.round(pg.current / pg.rate) : 0;
            const eta = pg.rate > 0 ? Math.round((pg.total - pg.current) / pg.rate) : 0;
            const elapsedStr = elapsed >= 60 ? Math.floor(elapsed/60) + 'm ' + (elapsed%60) + 's' : elapsed + 's';
            const etaStr = eta >= 60 ? Math.floor(eta/60) + 'm ' + (eta%60) + 's' : eta + 's';
            document.getElementById('prog-bar').style.width = pg.pct + '%';
            document.getElementById('prog-text').textContent =
                'Simulation: ' + pg.current + '/' + pg.total + ' (' + pg.pct + '%) \u2022 ' +
                pg.rate + ' cycles/sec \u2022 Elapsed: ' + elapsedStr + ' \u2022 ETA: ' + etaStr;
        } else if (pg.current > 0) {
            wrap.style.display = 'block';
            pauseBtn.style.display = 'none';
            document.getElementById('prog-bar').style.width = '100%';
            document.getElementById('prog-text').textContent = 'Simulation complete: ' + pg.current + ' cycles processed';
        }
    } catch(e) {}

    // Advanced feature badges
    try {
        const [vsR, cascR, driftR, explR] = await Promise.all([
            fetch('/api/virtual_sensor/status'),
            fetch('/api/cascade/state'),
            fetch('/api/drift/status'),
            fetch('/api/dosing/explanation'),
        ]);
        const vs = await vsR.json();
        const casc = await cascR.json();
        const drift = await driftR.json();
        const expl = await explR.json();

        // Virtual sensor badge
        const vsEl = document.getElementById('badge-vs');
        if (vs.available) {
            const cls = vs.confidence === 'GREEN' ? 'badge-low' : vs.confidence === 'AMBER' ? 'badge-moderate' : 'badge-high';
            vsEl.innerHTML = '<span class="kpi-badge ' + cls + '">' + vs.confidence + '</span>';
        } else {
            vsEl.innerHTML = '<span class="kpi-badge" style="background:#555">OFF</span>';
        }

        // Cascade badge
        const cascEl = document.getElementById('badge-cascade');
        const cascCls = casc.state === 'HEALTHY' ? 'badge-low' : casc.state_index <= 2 ? 'badge-moderate' : 'badge-high';
        cascEl.innerHTML = '<span class="kpi-badge ' + cascCls + '">' + casc.state + '</span>';

        // Drift badge
        const driftEl = document.getElementById('badge-drift');
        if (drift.total_drift_events > 0) {
            driftEl.innerHTML = '<span class="kpi-badge badge-moderate">DRIFT (' + drift.total_drift_events + ')</span>';
        } else {
            driftEl.innerHTML = '<span class="kpi-badge badge-low">STABLE</span>';
        }

        // Explanation
        if (expl.explanation) {
            document.getElementById('dosing-explanation').textContent = expl.explanation;
        }
    } catch(e) {}
}

update();
setInterval(update, 2000);
</script>
</body>
</html>"""


# ============================================================================
# API CREATION
# ============================================================================

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
    app.state.roi_data = {"tgf_costs": [], "baseline_costs": [], "labels": []}
    app.state.paused = False

    # ================================================================
    # HTML DASHBOARD
    # ================================================================

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        return DASHBOARD_HTML

    # ================================================================
    # ORIGINAL API ENDPOINTS (all 15 preserved)
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

    # ================================================================
    # NEW CHART ENDPOINTS
    # ================================================================

    @app.get("/api/charts/timeseries")
    async def get_chart_timeseries(limit: int = Query(200)):
        """Chart-ready sensor + LSI/RSI arrays."""
        store = app.state.store
        if store:
            return store.get_chart_data(limit=limit)
        return {"labels": [], "ph": [], "conductivity": [], "temperature": [],
                "orp": [], "lsi": [], "rsi": [], "risk_levels": [], "costs": []}

    @app.get("/api/charts/roi")
    async def get_chart_roi():
        """ROI comparison: TGF AI vs manual dosing baseline."""
        roi = app.state.roi_data
        if roi and roi.get("labels"):
            tgf = roi["tgf_costs"]
            base = roi["baseline_costs"]
            total_tgf = tgf[-1] if tgf else 0
            total_base = base[-1] if base else 0
            savings = total_base - total_tgf
            pct = (savings / total_base * 100) if total_base > 0 else 0
            return {
                "labels": roi["labels"],
                "tgf_costs": tgf,
                "baseline_costs": base,
                "savings_pct": round(pct, 1),
                "savings_inr": round(savings, 0),
            }
        return {"labels": [], "tgf_costs": [], "baseline_costs": [],
                "savings_pct": 0, "savings_inr": 0}

    @app.get("/api/charts/risk")
    async def get_chart_risk():
        """Risk distribution + key rates."""
        store = app.state.store
        result = {"LOW": 0, "MODERATE": 0, "HIGH": 0, "CRITICAL": 0,
                  "anomaly_rate_pct": 0, "preemptive_pct": 0}
        if store:
            summary = store.get_dashboard_summary()
            rd = summary.get("risk_distribution", {})
            result.update({k: rd.get(k, 0) for k in ["LOW", "MODERATE", "HIGH", "CRITICAL"]})

        detector = app.state.anomaly
        if detector:
            stats = detector.get_stats()
            result["anomaly_rate_pct"] = stats.get("anomaly_rate_pct", 0)

        ctrl = app.state.controller
        if ctrl and hasattr(ctrl, 'history') and ctrl.history:
            pre_count = sum(1 for h in ctrl.history if h.safe_decision.preemptive)
            result["preemptive_pct"] = round(pre_count / len(ctrl.history) * 100, 1) if ctrl.history else 0

        return result

    # ================================================================
    # ADDITIONAL CHART & EXPORT ENDPOINTS
    # ================================================================

    @app.get("/api/charts/anomalies")
    async def get_chart_anomalies(limit: int = Query(200)):
        """Anomaly timeline for scatter chart."""
        store = app.state.store
        if store:
            return store.get_anomaly_timeline(limit=limit)
        return []

    @app.get("/api/charts/dosing_history")
    async def get_chart_dosing_history(limit: int = Query(200)):
        """Per-chemical dosing history for stacked area chart."""
        store = app.state.store
        if store:
            return store.get_dosing_history(limit=limit)
        return {"labels": [], "chemicals": {}}

    @app.get("/api/report/summary")
    async def get_report_summary():
        """Full simulation summary for end-of-run report."""
        store = app.state.store
        if store:
            return store.get_simulation_summary()
        return {}

    @app.get("/api/export/readings.csv")
    async def export_readings_csv():
        """Export sensor readings as CSV download."""
        from fastapi.responses import StreamingResponse
        import io, csv
        store = app.state.store
        if not store:
            raise HTTPException(503, "No data store")
        rows = store.get_recent_readings(hours=99999, limit=99999)
        if not rows:
            raise HTTPException(404, "No readings")

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=tgf_readings.csv"}
        )

    @app.get("/api/export/decisions.csv")
    async def export_decisions_csv():
        """Export dosing decisions as CSV download."""
        from fastapi.responses import StreamingResponse
        import io, csv
        store = app.state.store
        if not store:
            raise HTTPException(503, "No data store")
        rows = store.get_recent_decisions(hours=99999, limit=99999)
        if not rows:
            raise HTTPException(404, "No decisions")

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=tgf_decisions.csv"}
        )

    @app.get("/api/export/anomalies.csv")
    async def export_anomalies_csv():
        """Export anomaly events as CSV download."""
        from fastapi.responses import StreamingResponse
        import io, csv
        store = app.state.store
        if not store:
            raise HTTPException(503, "No data store")
        rows = store.get_anomaly_timeline(limit=99999)
        if not rows:
            raise HTTPException(404, "No anomalies")

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=tgf_anomalies.csv"}
        )

    # ================================================================
    # PAUSE/RESUME CONTROL
    # ================================================================

    @app.post("/api/control/pause")
    async def pause_simulation():
        """Pause the simulation."""
        app.state.paused = True
        return {"status": "paused"}

    @app.post("/api/control/resume")
    async def resume_simulation():
        """Resume the simulation."""
        app.state.paused = False
        return {"status": "running"}

    @app.get("/api/control/state")
    async def get_control_state():
        """Get pause/resume state."""
        return {"paused": getattr(app.state, 'paused', False)}

    # ================================================================
    # ADVANCED FEATURE ENDPOINTS
    # ================================================================

    @app.get("/api/virtual_sensor/status")
    async def get_virtual_sensor_status():
        """Virtual sensor predictions and confidence."""
        vs = getattr(app.state, 'virtual_sensor', None)
        result = {
            "available": False,
            "confidence": "RED",
        }
        if vs and vs.available:
            result["available"] = True
            result["confidence"] = getattr(app.state, 'virtual_sensor_confidence', 'RED')
        return result

    @app.get("/api/cascade/state")
    async def get_cascade_state():
        """Current cascade detection state."""
        cd = getattr(app.state, 'cascade_detector', None)
        if cd:
            return cd.get_status()
        return {"state": "HEALTHY", "state_index": 0}

    @app.get("/api/drift/status")
    async def get_drift_status():
        """Drift detection status."""
        dd = getattr(app.state, 'drift_detector', None)
        if dd:
            return dd.get_status()
        return {"available": False}

    @app.get("/api/dosing/explanation")
    async def get_dosing_explanation():
        """Latest dosing explanation text."""
        explanation = getattr(app.state, 'last_explanation', '')
        return {"explanation": explanation}

    @app.get("/api/validation/backtest")
    async def get_backtest_results():
        """Cached backtest comparison results."""
        import os
        report_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'outputs', 'backtest_report.json'
        )
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                return json.load(f)
        return {"error": "No backtest results available. Run backtester first."}

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
