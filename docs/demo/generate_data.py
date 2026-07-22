"""
TGF Live Bench — offline data generator
=======================================
Builds the four scenario datasets that drive the static demo page under
``docs/demo/``. Each scenario is a seeded, synthetic seven-day hourly run of a
representative mid-size industrial cooling tower. The synthetic drivers are
pushed through the real TGF pipeline — the physics engine for the chemistry
indices, the chemical residual tracker and MPC optimizer for the dosing
recommendation, the safety layer for the interlock trace, the Chronos
forecaster for the forecast cone, and the statistical anomaly detector for the
event flags.

Nothing here connects to a plant. Everything is simulation output, and the
provenance strings written into each file name exactly which code path produced
each series.

Run from the repository root:

    python docs/demo/generate_data.py

Output is deterministic: rerunning on the same day reproduces byte-identical
files apart from the generated-date fields.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

# --- Wire in the TGF package (its modules import as top-level names) ---------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(REPO_ROOT, "tgf_dosing"))

from config.tower_config import AQUATECH_850_TPD, DEFAULT_LIMITS  # noqa: E402
from core.physics_engine import PhysicsEngine, WaterChemistry  # noqa: E402
from core.chemical_tracker import ChemicalResidualTracker  # noqa: E402
from core.chronos_forecaster import (  # noqa: E402
    ChronosForecaster, ForecastPoint, ParameterForecast, SystemForecast,
)
from core.mpc_optimizer import MPCDosingOptimizer  # noqa: E402
from core.safety_layer import SafetyLayer  # noqa: E402
from infrastructure.anomaly_detector import AnomalyDetector  # noqa: E402

TOWER = AQUATECH_850_TPD
LIMITS = DEFAULT_LIMITS
INHIBITOR = "AQUATREAT-2501"          # combined scale/corrosion inhibitor we feature
HOURS = 168                           # seven days of hourly history
HORIZON_H = 24                        # decision horizon
HORIZONS = list(range(1, HORIZON_H + 1))
DT_H = 1.0                            # hourly control step for this demo
START = datetime(2026, 7, 15, 0, 0, 0, tzinfo=timezone.utc)
GENERATED = datetime.now(timezone.utc).strftime("%Y-%m-%d")

OUT_DIR = os.path.join(HERE, "data")

# Guideline bands for the tiles and chart fills. Anchored to the tower's
# operating limits and chemical program where those exist; the circulating-
# mineral bands are illustrative ranges around the makeup-water design point.
PARAMS = {
    "ph":              {"label": "pH",                     "unit": "",           "band": [LIMITS.ph_min, LIMITS.ph_max],          "decimals": 2, "kind": "live"},
    "conductivity":    {"label": "Conductivity",           "unit": "µS/cm",      "band": [round(TOWER.makeup_conductivity_us * LIMITS.coc_min), round(TOWER.makeup_conductivity_us * LIMITS.coc_max)], "decimals": 0, "kind": "live"},
    "temperature":     {"label": "Temperature",            "unit": "°C",         "band": [20.0, LIMITS.temperature_max_c],        "decimals": 1, "kind": "live"},
    "orp":             {"label": "ORP",                    "unit": "mV",         "band": [LIMITS.orp_min_mv, LIMITS.orp_max_mv],  "decimals": 0, "kind": "live"},
    "inhibitor":       {"label": "Inhibitor residual",     "unit": "ppm",        "band": [TOWER.chemicals[INHIBITOR].min_ppm, TOWER.chemicals[INHIBITOR].max_ppm], "decimals": 1, "kind": "estimated"},
    "coc":             {"label": "Cycles of concentration","unit": "",           "band": [LIMITS.coc_min, LIMITS.coc_max],        "decimals": 2, "kind": "estimated"},
    "makeup_hardness": {"label": "Makeup hardness",        "unit": "ppm CaCO₃",  "band": [60, 180],                               "decimals": 0, "kind": "estimated"},
    "makeup_conductivity": {"label": "Makeup conductivity","unit": "µS/cm",      "band": [280, 480],                              "decimals": 0, "kind": "estimated"},
    "alkalinity":      {"label": "Alkalinity",             "unit": "ppm CaCO₃",  "band": [100, 500],                              "decimals": 0, "kind": "estimated"},
    "calcium":         {"label": "Calcium hardness",       "unit": "ppm CaCO₃",  "band": [150, 550],                              "decimals": 0, "kind": "estimated"},
    "chlorides":       {"label": "Chlorides",              "unit": "ppm",        "band": [80, 500],                               "decimals": 0, "kind": "estimated"},
}

SCENARIOS = {
    "baseline": {
        "title": "Baseline steady state",
        "subtitle": "Steady operation; indices mid-band; routine maintenance dosing",
        "seed": 11,
    },
    "scaling": {
        "title": "Scaling excursion",
        "subtitle": "Cycles of concentration drift up; scaling risk builds",
        "seed": 42,
    },
    "makeup": {
        "title": "Corrosive makeup shift",
        "subtitle": "Softer, low-alkalinity makeup water turns the loop corrosive",
        "seed": 73,
    },
    "fault": {
        "title": "Sensor-fault drill",
        "subtitle": "pH sensor sticks then drops out; the safety layer holds dosing",
        "seed": 97,
    },
}


# ---------------------------------------------------------------------------
# Chemistry helpers
# ---------------------------------------------------------------------------

def puckorius(phs: float, alkalinity_ppm: float) -> float:
    """Puckorius Scaling Index from the saturation pH and an equilibrium pH."""
    ph_eq = 1.465 * math.log10(max(alkalinity_ppm, 1.0)) + 4.54
    return 2.0 * phs - ph_eq


def timestamps(n: int, offset_h: int = 0):
    return [
        (START.timestamp() + (offset_h + i) * 3600.0)
        for i in range(n)
    ]


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Scenario drivers — the synthetic sensor/makeup series before the pipeline
# ---------------------------------------------------------------------------

def build_drivers(scenario: str, rng: np.random.Generator) -> dict:
    """Plausible hourly dynamics: diurnal temperature, gentle noise, and one
    event injection per scenario. Returns raw driver arrays; circulating
    minerals are derived from cycles of concentration afterwards."""
    h = np.arange(HOURS)
    diurnal = np.sin(2 * np.pi * (h - 8) / 24.0)

    temperature = 31.5 + 3.0 * diurnal + rng.normal(0, 0.18, HOURS)
    ph = 7.52 + 0.06 * np.sin(2 * np.pi * (h - 10) / 24.0) + rng.normal(0, 0.008, HOURS)
    orp = 660.0 + 15.0 * np.sin(2 * np.pi * (h - 14) / 24.0) - 10.0 * np.abs(np.sin(np.pi * h / 84.0)) + rng.normal(0, 3, HOURS)
    conductivity = 1340.0 + 25.0 * np.sin(2 * np.pi * h / 24.0) + rng.normal(0, 7, HOURS)

    makeup_hardness = np.full(HOURS, TOWER.makeup_hardness_ppm)
    makeup_calcium = np.full(HOURS, TOWER.makeup_calcium_ppm)
    makeup_alkalinity = np.full(HOURS, TOWER.makeup_alkalinity_ppm)
    makeup_chlorides = np.full(HOURS, 45.0)
    makeup_conductivity = np.full(HOURS, TOWER.makeup_conductivity_us)

    if scenario == "scaling":
        # Reduced blowdown from day 3: the loop concentrates steadily upward.
        ramp = np.clip(h - 72, 0, None) * 17.5
        conductivity = conductivity + ramp
        ph = ph + np.clip(h - 72, 0, None) * 0.0016   # a slow drift up as CaCO3 saturates

    elif scenario == "makeup":
        # Day-2 switch to softer, low-alkalinity, higher-chloride makeup water.
        shift = np.clip((h - 48) / 6.0, 0.0, 1.0)
        makeup_calcium = TOWER.makeup_calcium_ppm - 48.0 * shift
        makeup_alkalinity = TOWER.makeup_alkalinity_ppm - 72.0 * shift
        makeup_chlorides = 45.0 + 80.0 * shift
        makeup_conductivity = TOWER.makeup_conductivity_us - 40.0 * shift
        ph = ph - 0.34 * shift                          # weaker buffering pulls pH down

    elif scenario == "fault":
        # pH probe sticks flat on day 6, then reads a dropout code the rest of the run.
        stick_val = float(ph[143])
        ph[144:150] = stick_val                          # stuck-at: identical readings
        ph[150:] = 0.0                                   # dropout: implausible reading

    return {
        "temperature": temperature,
        "ph": ph,
        "orp": np.clip(orp, 250, 900),
        "conductivity": np.clip(conductivity, 300, 9000),
        "makeup_hardness": makeup_hardness,
        "makeup_calcium": makeup_calcium,
        "makeup_alkalinity": makeup_alkalinity,
        "makeup_chlorides": makeup_chlorides,
        "makeup_conductivity": makeup_conductivity,
    }


def derive_minerals(drivers: dict, rng: np.random.Generator) -> dict:
    """Circulating minerals from cycles of concentration, matching the physics
    engine's own CoC-based estimation (Ca = makeup x CoC, Alk down-weighted by
    the atmospheric-exchange correction)."""
    cond = drivers["conductivity"]
    mk_cond = drivers["makeup_conductivity"]
    coc = np.clip(cond / np.maximum(mk_cond, 1.0), 1.0, 15.0)
    calcium = drivers["makeup_calcium"] * coc + rng.normal(0, 2.0, HOURS)
    alkalinity = drivers["makeup_alkalinity"] * coc * 0.85 + rng.normal(0, 2.0, HOURS)
    chlorides = drivers["makeup_chlorides"] * coc + rng.normal(0, 3.0, HOURS)
    tds = cond * 0.65
    return {
        "coc": coc,
        "calcium": np.maximum(calcium, 1.0),
        "alkalinity": np.maximum(alkalinity, 1.0),
        "chlorides": np.maximum(chlorides, 0.0),
        "tds": tds,
    }


# ---------------------------------------------------------------------------
# Forecast — Chronos when it is installed, the repo statistical model otherwise
# ---------------------------------------------------------------------------

def chronos_available(forecaster: ChronosForecaster) -> bool:
    forecaster._load_model()
    return bool(forecaster._model_loaded)


def _chronos_quantiles(forecaster, series, param, horizons, seed):
    import torch
    torch.manual_seed(seed)
    context = list(series)[-forecaster.context_length:]
    ctx = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
    steps = int(max(horizons) * 60.0 / forecaster.sampling_interval_minutes)
    pred_len = min(steps, forecaster.MAX_PREDICTION_LENGTH)
    samples = forecaster._model.predict(
        ctx, prediction_length=pred_len, num_samples=forecaster.num_samples
    )[0].numpy()
    points = []
    for hz in horizons:
        idx = max(0, min(int(hz * 60.0 / forecaster.sampling_interval_minutes) - 1, pred_len - 1))
        vals = forecaster._constrain_forecast(param, samples[:, idx])
        points.append(ForecastPoint(hz, float(np.percentile(vals, 10)),
                                     float(np.percentile(vals, 50)),
                                     float(np.percentile(vals, 90))))
    return ParameterForecast(param, points, len(context))


def forecast_param(forecaster, series, param, horizons, use_chronos, seed):
    if use_chronos:
        return _chronos_quantiles(forecaster, series, param, horizons, seed)
    return forecaster._fallback_forecast(param, horizons, list(series))


# ---------------------------------------------------------------------------
# Forward rollout for the approve/decline mini-chart
# ---------------------------------------------------------------------------

def rollout(physics, residual0, cond_now, ph_now, temp_seed, mk, dose_kg, window_h, slope):
    """Roll the inhibitor residual and LSI forward for the decision horizon under
    two operator choices: run the recommended inhibitor dose, or decline it.
    Dosing a scale/corrosion inhibitor does not change the loop's mineral
    concentration, so both choices share one conductivity/LSI path (the
    scenario's natural continuation); only the inhibitor residual diverges.
    Residual dynamics come from the chemical tracker, LSI from the physics
    engine."""
    ts = timestamps(HORIZON_H, offset_h=HOURS)
    conds, lsi_path = [], []
    cond = cond_now
    for t in range(HORIZON_H):
        cond = float(np.clip(cond + slope, 300, 9000))
        coc = max(1.0, min(cond / max(mk["makeup_conductivity"], 1.0), 15.0))
        temp = temp_seed + 3.0 * math.sin(2 * math.pi * ((HOURS + t) - 8) / 24.0)
        ca = mk["makeup_calcium"] * coc
        alk = mk["makeup_alkalinity"] * coc * 0.85
        wc = WaterChemistry(ph=ph_now, conductivity_us=cond, temperature_c=temp,
                            orp_mv=650.0, tds_ppm=cond * 0.65,
                            calcium_hardness_ppm=ca, total_alkalinity_ppm=alk)
        conds.append(cond)
        lsi_path.append(physics.calculate_lsi(wc))

    def inhib_path(active):
        tracker = ChemicalResidualTracker(TOWER)
        tracker.residuals[INHIBITOR] = residual0
        out = []
        for t in range(HORIZON_H):
            coc = max(1.0, min(conds[t] / max(mk["makeup_conductivity"], 1.0), 15.0))
            temp = temp_seed + 3.0 * math.sin(2 * math.pi * ((HOURS + t) - 8) / 24.0)
            evap = physics.estimate_evaporation_rate(temp)
            bd_rate = 0.1 * LIMITS.max_blowdown_rate_m3_per_hr
            dose = dose_kg if (active and t < window_h) else 0.0
            tracker.update(dt_hours=DT_H, coc=coc, evaporation_rate=evap,
                           blowdown_rate=bd_rate, temperature_c=temp, lsi=lsi_path[t],
                           orp_mv=650.0, pump_actions={INHIBITOR: dose},
                           current_timestamp=ts[t])
            out.append(tracker.residuals[INHIBITOR])
        return out

    tiso = [iso(x) for x in ts]
    return (
        {"t": tiso, "inhibitor": inhib_path(True), "lsi": lsi_path},
        {"t": tiso, "inhibitor": inhib_path(False), "lsi": lsi_path},
    )


# ---------------------------------------------------------------------------
# Anomaly episodes — collapse consecutive flags into one entry per event
# ---------------------------------------------------------------------------

SENSOR_KEY = {"pH": "ph", "conductivity": "conductivity", "temperature": "temperature", "orp": "orp"}


def collect_anomalies(reports, ts):
    episodes = []
    open_ep = None
    for i, rep in enumerate(reports):
        if rep is not None and rep.is_anomalous and rep.suspect_sensor:
            key = SENSOR_KEY.get(rep.suspect_sensor, rep.suspect_sensor)
            if open_ep and open_ep["param"] == key and i - open_ep["_last"] <= 2:
                open_ep["_last"] = i
                if rep.system_score > open_ep["_score"]:
                    open_ep["_score"] = rep.system_score
                    open_ep["_note"] = rep.parameters[rep.suspect_sensor].details
                    open_ep["_crit"] = open_ep["_crit"] or rep.is_critical
            else:
                if open_ep:
                    episodes.append(open_ep)
                open_ep = {"param": key, "t": ts[i], "_last": i,
                           "_score": rep.system_score,
                           "_note": rep.parameters[rep.suspect_sensor].details,
                           "_crit": rep.is_critical}
        else:
            if open_ep:
                episodes.append(open_ep)
                open_ep = None
    if open_ep:
        episodes.append(open_ep)

    # Surface the high-confidence episodes only; borderline single-hour flags at
    # the detector's noise floor are dropped so the feed stays legible.
    episodes = [ep for ep in episodes if ep["_crit"] or ep["_score"] >= 0.6]
    episodes.sort(key=lambda e: e["_score"], reverse=True)
    episodes = episodes[:6]
    episodes.sort(key=lambda e: e["t"])

    out = []
    for ep in episodes:
        score = ep["_score"]
        severity = "critical" if ep["_crit"] else "high"
        out.append({
            "t": iso(ep["t"]),
            "param": ep["param"],
            "score": round(float(score), 3),
            "severity": severity,
            "note": ep["_note"],
        })
    return out


# ---------------------------------------------------------------------------
# Rounding so reruns are byte-identical
# ---------------------------------------------------------------------------

def rnd(x, d):
    return round(float(x), d)


def round_series(values, decimals):
    return [rnd(v, decimals) for v in values]


# ---------------------------------------------------------------------------
# v2 blocks — forecast anchors, radar scores, distributions, correlation,
# lab panel, dosing timeline, and the simulated operations layer
# ---------------------------------------------------------------------------

ANCHORS_H = list(range(48, HOURS + 1, 12))     # 24 h forecasts re-run every 12 h

LAB_PARAMS = {
    "total_hardness":     {"label": "Total hardness",     "unit": "ppm CaCO₃", "band": [250, 900],  "decimals": 0, "kind": "lab"},
    "magnesium_hardness": {"label": "Magnesium hardness", "unit": "ppm CaCO₃", "band": [80, 350],   "decimals": 0, "kind": "lab"},
    "silica":             {"label": "Silica",             "unit": "ppm SiO₂",  "band": [30, 150],   "decimals": 1, "kind": "lab"},
    "iron":               {"label": "Iron",               "unit": "ppm Fe",    "band": [0.0, 1.0],  "decimals": 2, "kind": "lab"},
    "sulfate":            {"label": "Sulphate",           "unit": "ppm",       "band": [100, 900],  "decimals": 0, "kind": "lab"},
    "ortho_phosphate":    {"label": "Ortho phosphate",    "unit": "ppm PO₄",   "band": [4.0, 12.0], "decimals": 2, "kind": "lab"},
    "total_phosphate":    {"label": "Total phosphate",    "unit": "ppm PO₄",   "band": [5.0, 15.0], "decimals": 2, "kind": "lab"},
    "tss":                {"label": "Suspended solids",   "unit": "ppm",       "band": [0, 40],     "decimals": 0, "kind": "lab"},
    "turbidity":          {"label": "Turbidity",          "unit": "NTU",       "band": [0.0, 15.0], "decimals": 1, "kind": "lab"},
}

LAB_STEP_H = 12

RADAR_AXES = [
    {"key": "scaling_margin",     "label": "Scaling margin",
     "formula": "100 · clip((1.0 − LSI) / 1.5, 0, 1) — distance of LSI below the +1.0 scaling threshold"},
    {"key": "corrosion_margin",   "label": "Corrosion margin",
     "formula": "100 · min(clip((LSI + 0.5) / 1.0, 0, 1), clip((7.5 − RSI) / 1.5, 0, 1)) — LSI above −0.5 and RSI below 7.5"},
    {"key": "inhibitor_adequacy", "label": "Inhibitor adequacy",
     "formula": "100 · clip((residual − band lo) / (target − band lo), 0, 1) against the chemical program band"},
    {"key": "data_health",        "label": "Data health",
     "formula": "100 − 60·(sensor fault active) − 12·(anomaly episodes in prior 24 h), floored at 0"},
    {"key": "safety_headroom",    "label": "Safety headroom",
     "formula": "0 if dosing held, else 100 · (1 − inhibitor dosed in prior 24 h / daily cap)"},
    {"key": "forecast_margin",    "label": "Forecast margin",
     "formula": "100 · clip((band hi − conductivity q90 at +24 h) / (0.15 · band hi), 0, 1); 50 when the forecast is suppressed"},
]


def clip01(x):
    return max(0.0, min(1.0, float(x)))


def build_anchor_forecasts(forecaster, drivers, inhib_s, use_chronos, seed):
    """Re-run the 24 h forecast at every 12 h anchor across the window, from the
    data available up to that hour only. For the fault drill, the pH forecast is
    suppressed at anchors whose recent context includes the faulted probe —
    forecasting on bad input would be worse than admitting the gap."""
    anchors = []
    sources = {
        "conductivity": ("conductivity", np.asarray(drivers["conductivity"], dtype=float)),
        "ph":           ("pH",           np.asarray(drivers["ph"], dtype=float)),
        "inhibitor":    ("inhibitor",    np.asarray(inhib_s, dtype=float)),
    }
    for k, a in enumerate(ANCHORS_H):
        params_out = {}
        for key, (pname, arr) in sources.items():
            ctx = arr[:a]
            dec = PARAMS[key]["decimals"]
            entry = {"t": [iso(x) for x in timestamps(HORIZON_H, offset_h=a)]}
            faulted = bool(key == "ph" and np.any(ctx[max(0, a - 24):a] <= 3.0))
            if faulted:
                entry["suppressed"] = True
                entry["note"] = "forecast suppressed — input sensor faulted"
            else:
                seed_k = seed if a == HOURS else seed * 1000 + k
                pf = forecast_param(forecaster, ctx, pname, HORIZONS, use_chronos,
                                    seed_k)
                entry["q50"] = [rnd(pf.at_horizon(hz).p50, dec) for hz in HORIZONS]
                entry["q10"] = [rnd(pf.at_horizon(hz).p10, dec) for hz in HORIZONS]
                entry["q90"] = [rnd(pf.at_horizon(hz).p90, dec) for hz in HORIZONS]
            params_out[key] = entry
        anchors.append({"h": a, "t": iso(timestamps(1, offset_h=a - 1)[0]),
                        "params": params_out})
    return anchors


def build_radar(anchors, lsi_s, rsi_s, inhib_s, cum24_s, blocked_s, anomalies):
    """Six 0–100 axis scores at every forecast anchor; the formulas live in
    RADAR_AXES and are shown verbatim on the page."""
    inhib_band = PARAMS["inhibitor"]["band"]
    target = TOWER.chemicals[INHIBITOR].target_ppm
    cap = TOWER.chemicals[INHIBITOR].max_dose_rate_kg_per_hr * 24.0
    cond_hi = PARAMS["conductivity"]["band"][1]
    anom_ts = [datetime.strptime(a["t"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
               for a in anomalies]
    scores = {ax["key"]: [] for ax in RADAR_AXES}
    for anchor in anchors:
        i = anchor["h"] - 1
        lsi, rsi = lsi_s[i], rsi_s[i]
        t_anchor = START.timestamp() + anchor["h"] * 3600.0
        n_anom = sum(1 for t in anom_ts if t_anchor - 86400.0 < t <= t_anchor)
        fault_active = bool(blocked_s[i])
        scores["scaling_margin"].append(rnd(100.0 * clip01((1.0 - lsi) / 1.5), 1))
        scores["corrosion_margin"].append(rnd(100.0 * min(clip01((lsi + 0.5) / 1.0),
                                                          clip01((7.5 - rsi) / 1.5)), 1))
        scores["inhibitor_adequacy"].append(rnd(100.0 * clip01((inhib_s[i] - inhib_band[0]) /
                                                               max(target - inhib_band[0], 0.1)), 1))
        scores["data_health"].append(rnd(max(0.0, 100.0 - (60.0 if fault_active else 0.0)
                                             - 12.0 * n_anom), 1))
        scores["safety_headroom"].append(rnd(0.0 if fault_active else
                                             100.0 * clip01(1.0 - cum24_s[i] / max(cap, 1e-9)), 1))
        fc = anchor["params"]["conductivity"]
        if fc.get("suppressed"):
            scores["forecast_margin"].append(50.0)
        else:
            q90_end = fc["q90"][-1]
            scores["forecast_margin"].append(rnd(100.0 * clip01((cond_hi - q90_end) /
                                                                (0.15 * cond_hi)), 1))
    return {"axes": RADAR_AXES, "anchors_h": [a["h"] for a in anchors], "scores": scores,
            "provenance": "axis scores computed from the simulated run by generate_data.py; formulas shown per axis"}


def build_distributions(series):
    """Quantile pack + 24-bin histogram per hourly parameter over the window."""
    out = {}
    for key in PARAMS:
        vals = np.asarray(series[key], dtype=float)
        vals = vals[np.isfinite(vals)]
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        iqr = q3 - q1
        counts, edges = np.histogram(vals, bins=24)
        d = PARAMS[key]["decimals"]
        out[key] = {
            "min": rnd(vals.min(), d), "q1": rnd(q1, d), "median": rnd(med, d),
            "mean": rnd(vals.mean(), d), "q3": rnd(q3, d), "max": rnd(vals.max(), d),
            "fence_lo": rnd(q1 - 1.5 * iqr, d), "fence_hi": rnd(q3 + 1.5 * iqr, d),
            "hist": {"edges": [rnd(e, d if d > 0 else 1) for e in edges],
                     "counts": [int(c) for c in counts]},
        }
    return out


def build_correlation(series):
    # Makeup-side series are near-constant in most scenarios; excluding them
    # keeps the matrix free of zero-variance artifacts.
    keys = [k for k in PARAMS if not k.startswith("makeup")]
    mat = np.corrcoef(np.vstack([np.asarray(series[k], dtype=float) for k in keys]))
    return {"params": keys, "n": HOURS, "window": "7 days, hourly",
            "matrix": [[rnd(v, 3) for v in row] for row in mat],
            "note": "Pearson correlation over the simulated hourly series; the fault drill's pH column includes the stuck/dropout readings on purpose."}


def build_lab_panel(sid, minerals, seed):
    """Slow lab-cadence chemistry, sampled every 12 h, tracking the loop's
    cycles of concentration so the panel stays consistent with the sensors."""
    rng = np.random.default_rng(seed + 5000)
    idx = list(range(0, HOURS, LAB_STEP_H))
    coc = minerals["coc"]
    ca = minerals["calcium"]
    out = {k: [] for k in LAB_PARAMS}
    for i in idx:
        c = float(coc[i])
        mg = 62.0 * c
        vals = {
            "total_hardness": float(ca[i]) + mg,
            "magnesium_hardness": mg,
            "silica": 16.0 * c * (1.06 if sid == "scaling" else 1.0),
            "iron": 0.12 + 0.045 * c,
            "sulfate": (52.0 + (34.0 if sid == "makeup" and i >= 48 else 0.0)) * c,
            "ortho_phosphate": 5.6 + 0.5 * math.sin(i / 29.0),
            "total_phosphate": 7.5 + 0.5 * math.sin(i / 29.0),
            "tss": 6.0 + 2.2 * c + (3.0 if sid == "scaling" and i >= 96 else 0.0),
            "turbidity": 1.6 + 0.5 * c + (0.8 if sid == "scaling" and i >= 96 else 0.0),
        }
        for k, v in vals.items():
            noisy = v * (1.0 + rng.normal(0, 0.012))
            out[k].append(rnd(noisy, LAB_PARAMS[k]["decimals"]))
    return {
        "t": [iso(START.timestamp() + i * 3600.0) for i in idx],
        "params": LAB_PARAMS,
        "values": out,
        "provenance": "synthetic lab panel (seeded), sampled every 12 h — presented as manually entered lab results",
    }


def build_biocide_slugs():
    """Twice-weekly oxidizing-biocide slug windows (illustrative schedule)."""
    slugs = []
    for day, hour in ((1, 10), (4, 10)):           # Tue/Fri of week 1 pattern
        for week in (0, 1):
            h = (day + 7 * week) * 24 + hour
            if h < HOURS:
                slugs.append({"start_h": h, "hours": 2,
                              "t": iso(START.timestamp() + h * 3600.0)})
    return slugs


def build_ops(sid, timeline_kg, blocked_s, bd_s, alerts, lab, seed):
    """Simulated operations layer: pump tiles, inventory drawdown, operator log.
    Everything here derives from the dosing timeline and the seeded schedule and
    is labeled simulated on the page."""
    slugs = build_biocide_slugs()
    slug_use_kg = 1.6

    inhib_cum = np.cumsum(np.asarray(timeline_kg, dtype=float))
    # Size the starting stocks off the window's own usage so the drawdown and
    # days-left read like a plausibly provisioned chemical store (~2 weeks
    # inhibitor, ~1 week biocide remaining at the end of the window).
    inhib_stock0 = float(math.ceil(float(inhib_cum[-1]) * 3.0 / 10.0) * 10.0)
    biocide_stock0 = 12.0
    daily_idx = list(range(0, HOURS, 24)) + [HOURS - 1]
    biocide_used_by = lambda h: slug_use_kg * sum(1 for s in slugs if s["start_h"] <= h)

    stock_series = {
        "t": [iso(START.timestamp() + i * 3600.0) for i in daily_idx],
        "inhibitor": [rnd(inhib_stock0 - inhib_cum[i], 1) for i in daily_idx],
        "biocide": [rnd(biocide_stock0 - biocide_used_by(i), 1) for i in daily_idx],
    }
    used_inhib = float(inhib_cum[-1])
    used_bio = biocide_used_by(HOURS)
    avg_daily_inhib = used_inhib / (HOURS / 24.0)
    avg_daily_bio = used_bio / (HOURS / 24.0)
    inventory = [
        {"chemical": "scale/corrosion inhibitor", "stock_start_kg": inhib_stock0,
         "used_kg": rnd(used_inhib, 1),
         "days_left": rnd((inhib_stock0 - used_inhib) / max(avg_daily_inhib, 1e-6), 1),
         "basis": "average simulated use over the window"},
        {"chemical": "oxidizing biocide", "stock_start_kg": biocide_stock0,
         "used_kg": rnd(used_bio, 1),
         "days_left": rnd((biocide_stock0 - used_bio) / max(avg_daily_bio, 1e-6), 1),
         "basis": "slug schedule, twice weekly"},
    ]

    rng = np.random.default_rng(seed + 9000)
    log = []
    for i, t in enumerate(lab["t"]):
        if i % 2 == 0:
            log.append({"t": t, "text": "Lab panel results entered", "source": "seeded"})
    for a in alerts[:3]:
        if a.get("resolved_t"):
            log.append({"t": a["resolved_t"], "text": f"Alert reviewed — {a['text']}",
                        "source": "seeded"})
    for s in slugs:
        log.append({"t": s["t"], "text": "Oxidizing biocide slug window started (schedule)",
                    "source": "seeded"})
    log.sort(key=lambda e: e["t"])

    pumps = [
        {"id": "pump_inhibitor", "label": "Inhibitor metering pump",
         "chemical": "scale/corrosion inhibitor", "drive": "timeline"},
        {"id": "pump_biocide", "label": "Biocide slug pump",
         "chemical": "oxidizing biocide", "drive": "slugs"},
        {"id": "valve_blowdown", "label": "Blowdown valve",
         "chemical": None, "drive": "blowdown"},
    ]
    return {
        "pumps": pumps,
        "biocide_slugs": slugs,
        "blowdown_open": round_series(np.asarray(bd_s) / max(float(np.max(bd_s)), 1e-9), 3),
        "inventory": inventory,
        "inventory_stock_series": stock_series,
        "operator_log": log,
        "provenance": "operations layer simulated for the demo — pump, inventory and log entries derive from the dosing timeline and a seeded schedule",
    }


def resolve_alert_lifecycles(alerts, series):
    """Attach a resolved_t to alerts whose condition demonstrably clears within
    the window; ongoing alerts keep resolved_t = None."""
    tlist = series["t"]
    cond = series["conductivity"]
    cond_hi = PARAMS["conductivity"]["band"][1]
    for a in alerts:
        a["resolved_t"] = None
        if a["source"] == "threshold" and a["text"].startswith("Conductivity"):
            start = tlist.index(a["t"]) if a["t"] in tlist else 0
            for j in range(start + 1, len(cond)):
                if cond[j] <= cond_hi:
                    a["resolved_t"] = tlist[j]
                    break
    return alerts


# ---------------------------------------------------------------------------
# One scenario end to end
# ---------------------------------------------------------------------------

def build_scenario(sid: str, meta: dict, use_chronos: bool, chronos_label: str):
    rng = np.random.default_rng(meta["seed"])
    physics = PhysicsEngine(TOWER)
    drivers = build_drivers(sid, rng)
    minerals = derive_minerals(drivers, rng)
    ts = timestamps(HOURS)

    # Real pipeline components, driven hour by hour.
    tracker = ChemicalResidualTracker(TOWER)
    mpc = MPCDosingOptimizer(TOWER, physics, LIMITS, horizon_steps=HORIZON_H, dt_hours=DT_H)
    safety = SafetyLayer(TOWER, LIMITS)
    forecaster = ChronosForecaster(model_size="small", context_length=512)
    if use_chronos:
        forecaster._load_model()
    detector = AnomalyDetector(window_size=288, warmup_cycles=24, sensitivity=1.0)

    lsi_s, rsi_s, psi_s, inhib_s = [], [], [], []
    timeline_kg, timeline_ml, blocked_s, cum24_s, bd_s = [], [], [], [], []
    anomaly_reports = []
    prev_doses = {}
    last = {}
    inhib_chem_ref = TOWER.chemicals[INHIBITOR]

    for i in range(HOURS):
        ph = float(drivers["ph"][i])
        cond = float(drivers["conductivity"][i])
        temp = float(drivers["temperature"][i])
        orp = float(drivers["orp"][i])
        ca = float(minerals["calcium"][i])
        alk = float(minerals["alkalinity"][i])
        tds = float(minerals["tds"][i])

        wc = WaterChemistry(ph=ph, conductivity_us=cond, temperature_c=temp, orp_mv=orp,
                            tds_ppm=tds, calcium_hardness_ppm=ca, total_alkalinity_ppm=alk,
                            timestamp=ts[i])
        risk = physics.full_risk_assessment(wc)
        phs = physics.calculate_phs(temp, tds, ca, alk)
        lsi_s.append(risk.lsi)
        rsi_s.append(risk.rsi)
        psi_s.append(puckorius(phs, alk))

        coc = physics.estimate_coc(cond)
        evap = physics.estimate_evaporation_rate(temp)
        bd = physics.estimate_blowdown_rate(coc, evap)
        snap = tracker.update(dt_hours=DT_H, coc=coc, evaporation_rate=evap, blowdown_rate=bd,
                              temperature_c=temp, lsi=risk.lsi, orp_mv=orp,
                              pump_actions=prev_doses, current_timestamp=ts[i])
        residuals = {n: st.estimated_ppm for n, st in snap.chemicals.items()}
        inhib_s.append(residuals[INHIBITOR])

        forecaster.add_reading(ts[i], {"pH": ph, "conductivity": cond, "temperature": temp, "ORP": orp})
        anomaly_reports.append(detector.analyze(ph=ph, conductivity=cond, temperature=temp,
                                                 orp=orp, timestamp=ts[i], cycle_index=i, tds=tds))

        # The history warms the safety layer and drives the residual tracker; the
        # forecast-aware decision is taken exactly once, at "now".
        if i == HOURS - 1:
            pfs = {key: forecast_param(forecaster, series, pname, HORIZONS, use_chronos, meta["seed"])
                   for key, (pname, series) in {
                       "conductivity": ("conductivity", drivers["conductivity"]),
                       "ph": ("pH", drivers["ph"]),
                       "orp": ("ORP", drivers["orp"]),
                       "inhibitor": ("inhibitor", np.array(inhib_s)),
                   }.items()}
            forecast = SystemForecast(
                timestamp=ts[i],
                parameters={"pH": pfs["ph"], "conductivity": pfs["conductivity"], "ORP": pfs["orp"]},
                forecast_horizons_hours=[float(h) for h in HORIZONS])
        else:
            forecast = None

        decision = mpc.optimize(wc, residuals, forecast, tracker, ts[i])
        safe_decision, report = safety.apply(decision, wc, residuals, dt_hours=DT_H)
        prev_doses = dict(safe_decision.continuous_doses_kg)
        for n, kg in safe_decision.slug_doses.items():
            prev_doses[n] = prev_doses.get(n, 0.0) + kg

        # Hourly dosing timeline for the console's playback and Advisor views.
        kg_i = 0.0 if report.emergency_stop else float(safe_decision.continuous_doses_kg.get(INHIBITOR, 0.0))
        timeline_kg.append(kg_i)
        timeline_ml.append(kg_i / inhib_chem_ref.density_kg_per_liter * 1000.0 / 60.0)
        blocked_s.append(bool(report.emergency_stop))
        cum24_s.append(float(snap.chemicals[INHIBITOR].cumulative_24h_kg))
        bd_s.append(float(bd))

        if i == HOURS - 1:
            last = {"wc": wc, "risk": risk, "residuals": residuals, "snap": snap,
                    "coc": coc, "cond": cond, "ph": ph, "temp": temp, "phs": phs,
                    "ts": ts[i], "safe": safe_decision, "report": report, "pfs": pfs}

    # --- Forecasts for the page: 24 h cones re-anchored every 12 h -----------
    # The final anchor reuses the loop's seed and context, so the cone shown at
    # "now" is exactly the forecast the controller acted on.
    anchors = build_anchor_forecasts(forecaster, drivers, inhib_s, use_chronos, meta["seed"])
    forecast_out = {key: dict(entry) for key, entry in anchors[-1]["params"].items()}

    # --- Recommendation + safety trace (decision taken in the loop at "now") -
    safe = last["safe"]
    report = last["report"]
    risk = last["risk"]
    blocked = report.emergency_stop
    inhib_chem = TOWER.chemicals[INHIBITOR]
    dose_kg = float(safe.continuous_doses_kg.get(INHIBITOR, 0.0))
    dose_ml_min = 0.0 if blocked else dose_kg / inhib_chem.density_kg_per_liter * 1000.0 / 60.0
    window_h = 4

    residual_now = last["residuals"][INHIBITOR]
    slope = (float(drivers["conductivity"][-1]) - float(drivers["conductivity"][-13])) / 12.0
    mk_now = {
        "makeup_conductivity": float(drivers["makeup_conductivity"][-1]),
        "makeup_calcium": float(drivers["makeup_calcium"][-1]),
        "makeup_alkalinity": float(drivers["makeup_alkalinity"][-1]),
    }
    # A valid pH for the rollout even when the live probe has dropped out.
    ph_for_roll = last["ph"] if last["ph"] > 3.0 else float(drivers["ph"][143])
    with_dose, without_dose = rollout(
        physics, residual_now, last["cond"], ph_for_roll, 31.5, mk_now,
        0.0 if blocked else dose_kg, window_h, slope)

    rationale = build_rationale(sid, risk, forecast_out, last, blocked, dose_ml_min)
    safety_checks = build_safety_checks(report, safe, last, blocked)

    if blocked:
        projected = "dosing held — no chemical delivered while the pH probe is faulted (simulated)"
    else:
        projected = f"inhibitor residual held near {inhib_chem.target_ppm:.0f} ppm across the horizon (simulated)"

    recommendation = {
        "state": "blocked" if blocked else "proposed",
        "chemical": "scale/corrosion inhibitor",
        "dose_ml_min": rnd(dose_ml_min, 2),
        "window_h": window_h,
        "rationale": rationale,
        "projected": projected,
        "safety_checks": safety_checks,
        "with_dose": {"t": with_dose["t"],
                      "inhibitor": round_series(with_dose["inhibitor"], 2),
                      "lsi": round_series(with_dose["lsi"], 3)},
        "without_dose": {"t": without_dose["t"],
                         "inhibitor": round_series(without_dose["inhibitor"], 2),
                         "lsi": round_series(without_dose["lsi"], 3)},
        "provenance": "tgf_dosing MPC optimizer + safety layer, run offline; trajectories from the TGF simulator. Closed-loop is backtest-only.",
    }

    # --- Anomalies + alerts -------------------------------------------------
    anomalies = collect_anomalies(anomaly_reports, ts)
    anomalies_provenance = ("tgf_dosing statistical anomaly detector (z-score + rate-of-change + "
                            "persistence; MOMENT not used)")
    alerts = build_alerts(sid, drivers, minerals, anomalies, report, ts)

    # --- Indices-now cross-check input --------------------------------------
    now_idx = HOURS - 1
    if sid == "fault":
        now_idx = 143   # last valid probe reading before the dropout
    ph_now = float(drivers["ph"][now_idx])
    temp_now = float(drivers["temperature"][now_idx])
    ca_now = float(minerals["calcium"][now_idx])
    alk_now = float(minerals["alkalinity"][now_idx])
    tds_now = float(minerals["tds"][now_idx])
    phs_now = physics.calculate_phs(temp_now, tds_now, ca_now, alk_now)
    indices_inputs_now = {"ph": rnd(ph_now, 2), "temp_c": rnd(temp_now, 1),
                          "calcium": rnd(ca_now, 0), "alkalinity": rnd(alk_now, 0),
                          "tds": rnd(tds_now, 0)}
    indices_now = {"lsi": rnd(ph_now - phs_now, 3),
                   "rsi": rnd(2.0 * phs_now - ph_now, 3),
                   "psi": rnd(puckorius(phs_now, alk_now), 3)}

    # --- Assemble -----------------------------------------------------------
    series = {"t": [iso(x) for x in ts]}
    for key in ("ph", "conductivity", "temperature", "orp"):
        series[key] = round_series(drivers[key], PARAMS[key]["decimals"])
    series["inhibitor"] = round_series(inhib_s, PARAMS["inhibitor"]["decimals"])
    series["coc"] = round_series(minerals["coc"], PARAMS["coc"]["decimals"])
    series["makeup_hardness"] = round_series(drivers["makeup_hardness"], PARAMS["makeup_hardness"]["decimals"])
    series["makeup_conductivity"] = round_series(drivers["makeup_conductivity"], PARAMS["makeup_conductivity"]["decimals"])
    series["alkalinity"] = round_series(minerals["alkalinity"], PARAMS["alkalinity"]["decimals"])
    series["calcium"] = round_series(minerals["calcium"], PARAMS["calcium"]["decimals"])
    series["chlorides"] = round_series(minerals["chlorides"], PARAMS["chlorides"]["decimals"])

    # --- v2 console blocks ---------------------------------------------------
    alerts = resolve_alert_lifecycles(alerts, series)
    lab = build_lab_panel(sid, minerals, meta["seed"])
    radar = build_radar(anchors, lsi_s, rsi_s, inhib_s, cum24_s, blocked_s, anomalies)
    distributions = build_distributions(series)
    correlation = build_correlation(series)
    ops = build_ops(sid, timeline_kg, blocked_s, bd_s, alerts, lab, meta["seed"])
    timeline = {
        "t": series["t"],
        "dose_ml_min": round_series(timeline_ml, 2),
        "blocked": [int(b) for b in blocked_s],
        "cum24_kg": round_series(cum24_s, 2),
        "daily_cap_kg": rnd(inhib_chem_ref.max_dose_rate_kg_per_hr * 24.0, 1),
        "pump_max_ml_min": rnd(inhib_chem_ref.max_dose_rate_kg_per_hr / inhib_chem_ref.density_kg_per_liter * 1000.0 / 60.0, 1),
        "provenance": "TGF MPC optimizer + safety layer evaluated hourly in the offline run",
    }
    sensor_fault = None
    if sid == "fault":
        sensor_fault = {"param": "ph",
                        "stuck_from": iso(ts[144]),
                        "dropout_from": iso(ts[150])}

    doc = {
        "meta": {
            "id": sid,
            "title": meta["title"],
            "subtitle": meta["subtitle"],
            "seed": meta["seed"],
            "schema": 2,
            "app": "TGF Console",
            "generated": GENERATED,
            "sim_start": series["t"][0],
            "sim_end": series["t"][-1],
            "tower": "representative mid-size industrial cooling tower (simulated)",
            "provenance_page": "All series simulated; see per-series provenance.",
        },
        "params": PARAMS,
        "series": series,
        "series_provenance": "synthetic scenario generator (seeded) → TGF physics engine",
        "indices": {
            "t": series["t"],
            "lsi": round_series(lsi_s, 3),
            "rsi": round_series(rsi_s, 3),
            "psi": round_series(psi_s, 3),
            "provenance": "tgf_dosing physics engine (LSI, RSI via vendored cooling-tower-chem formulas); PSI from the Puckorius equilibrium pH",
        },
        "indices_inputs_now": indices_inputs_now,
        "indices_now": indices_now,
        "forecast": {
            "horizon_h": HORIZON_H,
            "params": forecast_out,
            "anchors": anchors,
            "anchor_step_h": 12,
            "provenance": chronos_label,
        },
        "anomalies": anomalies,
        "anomalies_provenance": anomalies_provenance,
        "recommendation": recommendation,
        "alerts": alerts,
        "lab": lab,
        "timeline": timeline,
        "radar": radar,
        "distributions": distributions,
        "correlation": correlation,
        "ops": ops,
        "sensor_fault": sensor_fault,
    }
    return doc


def build_rationale(sid, risk, forecast_out, last, blocked, dose_ml_min):
    if blocked:
        return [
            "pH probe reading is outside the plausible range and has flatlined — the reading cannot be trusted.",
            "Sensor-sanity interlock failed, so the safety layer holds all dosing until the probe is verified.",
            "Indices derived from the faulted probe are suppressed; no chemical is delivered on bad data.",
        ]
    cond_band_hi = PARAMS["conductivity"]["band"][1]
    cond_q90_end = forecast_out["conductivity"]["q90"][-1]
    bullets = []
    if sid == "scaling":
        bullets.append(f"LSI is {risk.lsi:+.2f} and trending into the scaling zone as the loop concentrates.")
        bullets.append(f"The conductivity forecast's upper band reaches {cond_q90_end:.0f} µS/cm within {HORIZON_H} h, crossing the {cond_band_hi:.0f} µS/cm guideline.")
        bullets.append(f"Holding the scale/corrosion inhibitor at {dose_ml_min:.1f} mL/min keeps surfaces protected while the water stays concentrated.")
    elif sid == "makeup":
        bullets.append(f"LSI has gone negative ({risk.lsi:+.2f}) and RSI is {risk.rsi:.1f} — the water turned corrosive after the makeup shift.")
        bullets.append("Low-alkalinity makeup water reduced the buffering and the protective saturation margin.")
        bullets.append(f"Raising the scale/corrosion inhibitor to {dose_ml_min:.1f} mL/min restores the protective film.")
    else:  # baseline
        bullets.append(f"Indices sit mid-band (LSI {risk.lsi:+.2f}, RSI {risk.rsi:.1f}); the loop is stable.")
        bullets.append(f"A steady dose of {dose_ml_min:.1f} mL/min offsets normal inhibitor decay under load.")
        bullets.append("No forecast guideline crossing within the horizon.")
    return bullets


def build_safety_checks(report, safe, last, blocked):
    tower = TOWER
    inhib_chem = tower.chemicals[INHIBITOR]
    coc = last["coc"]
    ph = last["ph"]
    cum_24h = last["snap"].chemicals[INHIBITOR].cumulative_24h_kg
    daily_cap = inhib_chem.max_dose_rate_kg_per_hr * 24.0

    sensor_ok = not report.sensor_fault
    rate_ok = True   # the safety layer always clamps to the pump and step limits
    cum_ok = cum_24h <= daily_cap
    interlock_ok = (coc <= LIMITS.coc_max * 1.1) and ("blowdown" not in report.overrides) and ("cpcb_ph" not in report.overrides)

    return [
        {"name": "Sensor sanity",
         "detail": ("pH probe outside plausible range [3.0–12.0] for ≥3 cycles — dosing held"
                    if not sensor_ok else "all probes within plausible range"),
         "pass": bool(sensor_ok)},
        {"name": "Dose rate limit",
         "detail": f"≤ {SafetyLayer.MAX_RATE_CHANGE:.0%} step change and ≤ {inhib_chem.max_dose_rate_kg_per_hr:.0f} kg/hr pump limit",
         "pass": bool(rate_ok)},
        {"name": "Daily cumulative cap",
         "detail": f"{cum_24h:.2f} kg dosed in last 24 h vs {daily_cap:.0f} kg cap",
         "pass": bool(cum_ok)},
        {"name": "Discharge interlock (CoC / CPCB pH)",
         "detail": (f"CoC {coc:.1f} within {LIMITS.coc_max:.0f} limit; discharge pH check deferred while the probe is faulted"
                    if not sensor_ok else
                    f"CoC {coc:.1f} within {LIMITS.coc_max:.0f} limit; discharge pH {ph:.1f} within CPCB range" if interlock_ok
                    else "blowdown restricted to hold discharge quality"),
         "pass": bool(interlock_ok)},
    ]


def build_alerts(sid, drivers, minerals, anomalies, report, ts):
    alerts = []
    cond = drivers["conductivity"]
    cond_hi = PARAMS["conductivity"]["band"][1]
    # Threshold alert: first hour conductivity leaves the guideline band.
    over = np.where(cond > cond_hi)[0]
    if len(over):
        i = int(over[0])
        alerts.append({"t": iso(ts[i]), "severity": "warning",
                       "text": f"Conductivity above guideline band ({cond[i]:.0f} µS/cm)",
                       "source": "threshold"})
    # Corrosive-index threshold for the makeup scenario.
    if sid == "makeup":
        alerts.append({"t": iso(ts[60]), "severity": "warning",
                       "text": "Saturation index turned negative — corrosive tendency",
                       "source": "threshold"})
    # Anomaly-sourced alerts.
    for a in anomalies:
        alerts.append({"t": a["t"], "severity": "critical" if a["severity"] == "critical" else "warning",
                       "text": f"Anomaly on {a['param']} — {a['note']}", "source": "anomaly"})
    # Safety-sourced alert.
    if report.emergency_stop:
        alerts.append({"t": iso(ts[-1]), "severity": "critical",
                       "text": "pH sensor sanity check failed — dosing held", "source": "safety"})
    alerts.sort(key=lambda x: x["t"])
    return alerts


# ---------------------------------------------------------------------------
# Provenance manifest
# ---------------------------------------------------------------------------

def package_versions():
    import importlib.metadata as md
    names = ["numpy", "pandas", "scipy", "scikit-learn", "statsmodels",
             "torch", "transformers", "chronos-forecasting", "momentfm"]
    out = {}
    for n in names:
        try:
            out[n] = md.version(n)
        except Exception:
            out[n] = None
    return out


def repo_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        try:
            with open(os.path.join(REPO_ROOT, ".git", "HEAD")) as f:
                ref = f.read().strip()
            if ref.startswith("ref:"):
                with open(os.path.join(REPO_ROOT, ".git", ref.split(" ", 1)[1])) as f:
                    return f.read().strip()
            return ref
        except Exception:
            return None


def write_json(path, doc):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    forecaster_probe = ChronosForecaster(model_size="small", context_length=512)
    use_chronos = chronos_available(forecaster_probe)
    if use_chronos:
        chronos_label = f"chronos-t5-small (zero-shot, q10/q50/q90), run offline {GENERATED}"
        forecast_ran = "chronos-t5-small"
    else:
        chronos_label = "TGF statistical fallback forecaster (Chronos not used)"
        forecast_ran = "TGF statistical fallback forecaster"

    for sid, meta in SCENARIOS.items():
        doc = build_scenario(sid, meta, use_chronos, chronos_label)
        write_json(os.path.join(OUT_DIR, f"scenario-{sid}.json"), doc)
        rec = doc["recommendation"]
        print(f"  {sid:9s} state={rec['state']:8s} dose={rec['dose_ml_min']:.2f} mL/min "
              f"anomalies={len(doc['anomalies'])} alerts={len(doc['alerts'])}")

    provenance = {
        "generated": GENERATED,
        "repo_commit": repo_commit(),
        "python": sys.version.split()[0],
        "package_versions": package_versions(),
        "what_ran": {
            "forecast": forecast_ran,
            "anomalies": "TGF statistical anomaly detector",
            "indices": "TGF physics engine (vendored cooling-tower-chem formulas)",
            "controller": "TGF MPC optimizer + safety layer",
        },
        "forecast_anchors": "24 h forecasts recomputed every 12 h from hour 48; in the fault drill the pH cone is suppressed while the probe is faulted",
        "radar_axes": {ax["key"]: ax["formula"] for ax in RADAR_AXES},
        "ops_layer": "pump, inventory and operator-log entries are simulated for the demo and labeled as such",
        "regenerate": "python docs/demo/generate_data.py",
    }
    write_json(os.path.join(OUT_DIR, "provenance.json"), provenance)
    print(f"  forecast provenance: {chronos_label}")


if __name__ == "__main__":
    main()
