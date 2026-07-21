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
    "ph":              {"label": "pH",                     "unit": "",           "band": [LIMITS.ph_min, LIMITS.ph_max],          "decimals": 2},
    "conductivity":    {"label": "Conductivity",           "unit": "µS/cm",      "band": [round(TOWER.makeup_conductivity_us * LIMITS.coc_min), round(TOWER.makeup_conductivity_us * LIMITS.coc_max)], "decimals": 0},
    "temperature":     {"label": "Temperature",            "unit": "°C",         "band": [20.0, LIMITS.temperature_max_c],        "decimals": 1},
    "orp":             {"label": "ORP",                    "unit": "mV",         "band": [LIMITS.orp_min_mv, LIMITS.orp_max_mv],  "decimals": 0},
    "inhibitor":       {"label": "Inhibitor residual",     "unit": "ppm",        "band": [TOWER.chemicals[INHIBITOR].min_ppm, TOWER.chemicals[INHIBITOR].max_ppm], "decimals": 1},
    "coc":             {"label": "Cycles of concentration","unit": "",           "band": [LIMITS.coc_min, LIMITS.coc_max],        "decimals": 2},
    "makeup_hardness": {"label": "Makeup hardness",        "unit": "ppm CaCO₃",  "band": [60, 180],                               "decimals": 0},
    "alkalinity":      {"label": "Alkalinity",             "unit": "ppm CaCO₃",  "band": [100, 500],                              "decimals": 0},
    "calcium":         {"label": "Calcium hardness",       "unit": "ppm CaCO₃",  "band": [150, 550],                              "decimals": 0},
    "chlorides":       {"label": "Chlorides",              "unit": "ppm",        "band": [80, 500],                               "decimals": 0},
}

SCENARIOS = {
    "baseline": {
        "title": "Baseline steady state",
        "subtitle": "Steady operation; indices mid-band; a small maintenance dose",
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
    ph = 7.75 + 0.06 * np.sin(2 * np.pi * (h - 10) / 24.0) + rng.normal(0, 0.008, HOURS)
    orp = 660.0 + 15.0 * np.sin(2 * np.pi * (h - 14) / 24.0) - 10.0 * np.abs(np.sin(np.pi * h / 84.0)) + rng.normal(0, 3, HOURS)
    conductivity = 1460.0 + 25.0 * np.sin(2 * np.pi * h / 24.0) + rng.normal(0, 7, HOURS)

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
        makeup_calcium = TOWER.makeup_calcium_ppm - 42.0 * shift
        makeup_alkalinity = TOWER.makeup_alkalinity_ppm - 58.0 * shift
        makeup_chlorides = 45.0 + 80.0 * shift
        makeup_conductivity = TOWER.makeup_conductivity_us - 40.0 * shift
        ph = ph - 0.32 * shift                          # weaker buffering pulls pH down

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
    anomaly_reports = []
    prev_doses = {}
    last = {}

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

        if i == HOURS - 1:
            last = {"wc": wc, "risk": risk, "residuals": residuals, "snap": snap,
                    "coc": coc, "cond": cond, "ph": ph, "temp": temp, "phs": phs,
                    "ts": ts[i], "safe": safe_decision, "report": report, "pfs": pfs}

    # --- Forecast section for the page (conductivity, pH, inhibitor) ---------
    pfs = last["pfs"]
    fts = [iso(x) for x in timestamps(HORIZON_H, offset_h=HOURS)]
    forecast_out = {}
    for key in ("conductivity", "ph", "inhibitor"):
        pf = pfs[key]
        dec = PARAMS[key]["decimals"]
        forecast_out[key] = {
            "t": fts,
            "q50": [rnd(pf.at_horizon(hz).p50, dec) for hz in HORIZONS],
            "q10": [rnd(pf.at_horizon(hz).p10, dec) for hz in HORIZONS],
            "q90": [rnd(pf.at_horizon(hz).p90, dec) for hz in HORIZONS],
        }

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
    series["alkalinity"] = round_series(minerals["alkalinity"], PARAMS["alkalinity"]["decimals"])
    series["calcium"] = round_series(minerals["calcium"], PARAMS["calcium"]["decimals"])
    series["chlorides"] = round_series(minerals["chlorides"], PARAMS["chlorides"]["decimals"])

    doc = {
        "meta": {
            "id": sid,
            "title": meta["title"],
            "subtitle": meta["subtitle"],
            "seed": meta["seed"],
            "generated": GENERATED,
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
            "provenance": chronos_label,
        },
        "anomalies": anomalies,
        "anomalies_provenance": anomalies_provenance,
        "recommendation": recommendation,
        "alerts": alerts,
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
        bullets.append(f"A small maintenance dose of {dose_ml_min:.1f} mL/min offsets normal inhibitor decay.")
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
         "detail": f"CoC {coc:.1f} within {LIMITS.coc_max:.0f} limit; discharge pH {ph:.1f} within CPCB range" if interlock_ok
                   else "blowdown restricted to hold discharge quality",
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
        "regenerate": "python docs/demo/generate_data.py",
    }
    write_json(os.path.join(OUT_DIR, "provenance.json"), provenance)
    print(f"  forecast provenance: {chronos_label}")


if __name__ == "__main__":
    main()
