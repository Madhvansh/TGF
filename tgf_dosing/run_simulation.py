"""
TGF Historical Simulation Runner
==================================
Runs the complete dosing control system on historical Parameters_5K.csv data.

This validates the entire pipeline:
1. Physics engine (LSI/RSI calculations)
2. Chemical residual tracking (mass balance)
3. Forecasting (statistical fallback if Chronos unavailable)
4. MPC optimization
5. Safety layer

Output: Performance report with cost analysis, risk metrics, dosing patterns.

Usage:
    cd /home/claude/tgf_dosing
    python run_simulation.py
"""
import sys
import os
import pandas as pd
import numpy as np
import json
import logging
import time
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.tower_config import AQUATECH_850_TPD, DEFAULT_LIMITS
from core.physics_engine import PhysicsEngine, WaterChemistry
from core.chemical_tracker import ChemicalResidualTracker
from core.chronos_forecaster import ChronosForecaster
from core.mpc_optimizer import MPCDosingOptimizer, DosingDecision
from core.safety_layer import SafetyLayer
from core.dosing_controller import DosingController, ControlCycleResult

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("TGF_Simulation")


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Load Parameters_5K.csv and prepare for simulation."""
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"  Raw samples: {len(df)}")
    
    # Required columns for simulation
    required = ['pH', 'TDS_ppm', 'Conductivity_uS_cm']
    
    # Check available columns
    available = {col: df[col].notna().sum() for col in df.columns}
    logger.info("  Column availability:")
    for col, count in sorted(available.items(), key=lambda x: -x[1]):
        logger.info(f"    {col}: {count} ({count/len(df)*100:.0f}%)")
    
    # We need pH at minimum. Fill others intelligently.
    df_sim = df.copy()
    
    # Create synthetic timestamp if Date column is empty/missing
    if 'Date' not in df.columns or df['Date'].isna().all():
        # Simulate 5-minute intervals
        start_ts = datetime(2024, 1, 1).timestamp()
        df_sim['timestamp'] = [start_ts + i * 300 for i in range(len(df))]
        logger.info("  Created synthetic timestamps (5-min intervals)")
    else:
        df_sim['Date'] = pd.to_datetime(df_sim['Date'], errors='coerce')
        df_sim['timestamp'] = df_sim['Date'].apply(
            lambda x: x.timestamp() if pd.notna(x) else None)
        # Fill missing timestamps
        if df_sim['timestamp'].isna().any():
            start_ts = datetime(2024, 1, 1).timestamp()
            df_sim['timestamp'] = df_sim['timestamp'].fillna(
                pd.Series([start_ts + i * 300 for i in range(len(df))]))
    
    # Fill missing sensor values
    # pH: forward fill (sensor reading persists)
    df_sim['pH'] = df_sim['pH'].ffill().fillna(7.8)
    
    # Conductivity: if missing, estimate from TDS
    if 'Conductivity_uS_cm' in df.columns:
        df_sim['Conductivity_uS_cm'] = df_sim['Conductivity_uS_cm'].fillna(
            df_sim['TDS_ppm'] / 0.65)
    df_sim['Conductivity_uS_cm'] = df_sim['Conductivity_uS_cm'].ffill().fillna(2000)
    
    # TDS: if missing, estimate from conductivity
    df_sim['TDS_ppm'] = df_sim['TDS_ppm'].fillna(
        df_sim['Conductivity_uS_cm'] * 0.65)
    df_sim['TDS_ppm'] = df_sim['TDS_ppm'].ffill().fillna(1300)
    # Ensure TDS is never 0
    df_sim['TDS_ppm'] = df_sim['TDS_ppm'].clip(lower=50.0)
    
    # Temperature: not in dataset, simulate seasonal + daily pattern
    if 'Temperature_C' not in df.columns or df_sim.get('Temperature_C', pd.Series()).isna().all():
        n = len(df_sim)
        # Seasonal: 25°C winter, 40°C summer (India)
        seasonal = 32.5 + 7.5 * np.sin(2 * np.pi * np.arange(n) / (365 * 288))
        # Daily: ±3°C
        daily = 3.0 * np.sin(2 * np.pi * np.arange(n) / 288)
        # Random noise
        noise = np.random.normal(0, 0.5, n)
        df_sim['Temperature_C'] = seasonal + daily + noise
        logger.info("  Generated synthetic temperature data (seasonal+daily)")
    
    # ORP: not in dataset, simulate based on biocide dosing patterns
    if 'ORP_mV' not in df.columns:
        n = len(df_sim)
        # Base ORP around 650 with random fluctuations
        base_orp = 650
        # Periodic drops (biocide depletion between doses)
        periodic = -50 * np.abs(np.sin(2 * np.pi * np.arange(n) / (15 * 288)))
        # Random
        noise = np.random.normal(0, 20, n)
        df_sim['ORP_mV'] = base_orp + periodic + noise
        df_sim['ORP_mV'] = df_sim['ORP_mV'].clip(300, 800)
        logger.info("  Generated synthetic ORP data")
    
    # Sort by timestamp
    df_sim = df_sim.sort_values('timestamp').reset_index(drop=True)
    
    # Keep columns we need
    sim_cols = ['timestamp', 'pH', 'Conductivity_uS_cm', 'TDS_ppm', 
                'Temperature_C', 'ORP_mV',
                'Total_Hardness_ppm', 'Calcium_Hardness_ppm', 
                'Total_Alkalinity_ppm', 'Chlorides_ppm']
    
    for col in sim_cols:
        if col not in df_sim.columns:
            df_sim[col] = np.nan
    
    df_sim = df_sim[sim_cols]
    logger.info(f"  Prepared {len(df_sim)} samples for simulation")
    
    return df_sim


def run_simulation(data: pd.DataFrame, 
                   max_cycles: int = None,
                   enable_forecasting: bool = True) -> dict:
    """
    Run the complete dosing control simulation.
    
    Args:
        data: Prepared DataFrame with sensor readings
        max_cycles: Limit number of cycles (None = all data)
        enable_forecasting: Enable Chronos forecasting
    
    Returns:
        Dict with simulation results and metrics
    """
    tower = AQUATECH_850_TPD
    
    # Initialize controller
    controller = DosingController(
        tower_config=tower,
        limits=DEFAULT_LIMITS,
        chronos_model_size="base",
        enable_forecasting=enable_forecasting,
    )
    
    n_cycles = min(len(data), max_cycles) if max_cycles else len(data)
    logger.info(f"\nStarting simulation: {n_cycles} cycles")
    logger.info(f"  Tower: {tower.name}")
    logger.info(f"  Volume: {tower.holding_volume_m3} m³")
    logger.info(f"  Chemicals: {len(tower.chemicals)} products")
    logger.info(f"  Forecasting: {'ON' if enable_forecasting else 'OFF'}")
    
    # Tracking arrays
    results = []
    lsi_values = []
    rsi_values = []
    risk_levels = []
    total_costs = []
    preemptive_count = 0
    safety_override_count = 0
    
    # Chemical tracking
    chemical_doses_total = {name: 0.0 for name in tower.chemicals}
    chemical_status_counts = {name: {"ADEQUATE": 0, "LOW": 0, "CRITICAL": 0, "OVERDOSED": 0}
                              for name in tower.chemicals}
    
    start_time = time.time()
    
    # Simulate lab calibration every 2016 cycles (~7 days at 5-min intervals)
    LAB_INTERVAL = 2016
    
    for i in range(n_cycles):
        row = data.iloc[i]
        
        # Run one control cycle
        try:
            result = controller.run_cycle(
                ph=float(row['pH']),
                conductivity=float(row['Conductivity_uS_cm']),
                temperature=float(row['Temperature_C']),
                orp=float(row['ORP_mV']),
                timestamp=float(row['timestamp']),
                tds=float(row['TDS_ppm']) if pd.notna(row.get('TDS_ppm')) else None,
                calcium_hardness=float(row['Calcium_Hardness_ppm']) if pd.notna(row.get('Calcium_Hardness_ppm')) else None,
                total_alkalinity=float(row['Total_Alkalinity_ppm']) if pd.notna(row.get('Total_Alkalinity_ppm')) else None,
                total_hardness=float(row['Total_Hardness_ppm']) if pd.notna(row.get('Total_Hardness_ppm')) else None,
            )
            
            # Track metrics
            lsi_values.append(result.risk_assessment.lsi)
            rsi_values.append(result.risk_assessment.rsi)
            risk_levels.append(result.risk_assessment.risk_level)
            total_costs.append(result.total_chemical_cost_inr)
            
            if result.safe_decision.preemptive:
                preemptive_count += 1
            
            if result.safety_report.overrides:
                safety_override_count += 1
            
            # Track per-chemical doses
            for name, kg in result.safe_decision.continuous_doses_kg.items():
                chemical_doses_total[name] = chemical_doses_total.get(name, 0) + kg
            for name, kg in result.safe_decision.slug_doses.items():
                chemical_doses_total[name] = chemical_doses_total.get(name, 0) + kg
            
            # Track chemical status
            for name, state in result.tracker_snapshot.chemicals.items():
                if name in chemical_status_counts:
                    chemical_status_counts[name][state.status] = (
                        chemical_status_counts[name].get(state.status, 0) + 1)
            
            results.append(result.summary_dict())
            
        except Exception as e:
            logger.error(f"Cycle {i} failed: {e}")
            import traceback
            traceback.print_exc()
            if i < 5:  # If early failures, abort
                raise
            continue
        
        # Simulate lab calibration
        if i > 0 and i % LAB_INTERVAL == 0:
            # Use actual measured values from data as "lab results"
            lab_results = {}
            if pd.notna(row.get('Calcium_Hardness_ppm')):
                lab_results['calcium_hardness'] = float(row['Calcium_Hardness_ppm'])
            if pd.notna(row.get('Total_Alkalinity_ppm')):
                lab_results['alkalinity'] = float(row['Total_Alkalinity_ppm'])
            if pd.notna(row.get('Total_Hardness_ppm')):
                lab_results['total_hardness'] = float(row['Total_Hardness_ppm'])
            
            if lab_results:
                controller.calibrate_from_lab(
                    lab_results=lab_results,
                    current_conductivity=float(row['Conductivity_uS_cm']),
                    timestamp=float(row['timestamp']),
                )
                logger.info(f"  Lab calibration applied at cycle {i}: {lab_results}")
        
        # Progress
        if (i + 1) % 500 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (n_cycles - i - 1) / max(rate, 0.1)
            logger.info(
                f"  Progress: {i+1}/{n_cycles} ({(i+1)/n_cycles*100:.0f}%) "
                f"Rate: {rate:.0f} cycles/sec  ETA: {eta:.0f}s"
            )
    
    elapsed = time.time() - start_time
    
    # ================================================================
    # BUILD REPORT
    # ================================================================
    lsi_arr = np.array(lsi_values)
    rsi_arr = np.array(rsi_values)
    cost_arr = np.array(total_costs)
    
    report = {
        "simulation": {
            "tower": tower.name,
            "total_cycles": n_cycles,
            "elapsed_seconds": round(elapsed, 1),
            "cycles_per_second": round(n_cycles / max(elapsed, 0.01), 1),
            "simulated_days": round(n_cycles * 5 / 60 / 24, 1),
        },
        "water_chemistry": {
            "LSI": {
                "mean": round(float(np.mean(lsi_arr)), 3),
                "std": round(float(np.std(lsi_arr)), 3),
                "min": round(float(np.min(lsi_arr)), 3),
                "max": round(float(np.max(lsi_arr)), 3),
                "pct_scaling": round(float(np.mean(lsi_arr > 1.5) * 100), 1),
                "pct_corrosive": round(float(np.mean(lsi_arr < -1.0) * 100), 1),
                "pct_optimal": round(float(np.mean((lsi_arr >= -0.5) & (lsi_arr <= 1.5)) * 100), 1),
            },
            "RSI": {
                "mean": round(float(np.mean(rsi_arr)), 3),
                "std": round(float(np.std(rsi_arr)), 3),
            },
        },
        "risk_distribution": {
            level: round(risk_levels.count(level) / max(len(risk_levels), 1) * 100, 1)
            for level in ["LOW", "MODERATE", "HIGH", "CRITICAL"]
        },
        "dosing": {
            "total_chemical_cost_inr": round(float(np.sum(cost_arr)), 0),
            "daily_avg_cost_inr": round(float(np.sum(cost_arr)) / max(n_cycles * 5 / 60 / 24, 0.01), 0),
            "preemptive_pct": round(preemptive_count / max(n_cycles, 1) * 100, 1),
            "safety_override_pct": round(safety_override_count / max(n_cycles, 1) * 100, 1),
            "per_chemical_kg": {
                name: round(kg, 2) for name, kg in chemical_doses_total.items()
            },
        },
        "chemical_adequacy": {
            name: {
                status: round(count / max(n_cycles, 1) * 100, 1)
                for status, count in counts.items()
            }
            for name, counts in chemical_status_counts.items()
        },
        "dashboard_final_state": controller.get_dashboard_state(),
    }
    
    return report, results


def print_report(report: dict):
    """Print formatted simulation report."""
    print("\n" + "="*70)
    print("  TGF PREDICTIVE DOSING SIMULATION REPORT")
    print("="*70)
    
    sim = report["simulation"]
    print(f"\n  Tower: {sim['tower']}")
    print(f"  Cycles: {sim['total_cycles']} ({sim['simulated_days']} simulated days)")
    print(f"  Runtime: {sim['elapsed_seconds']}s ({sim['cycles_per_second']} cycles/sec)")
    
    wc = report["water_chemistry"]
    print(f"\n  --- Water Chemistry ---")
    print(f"  LSI: {wc['LSI']['mean']:.2f} ± {wc['LSI']['std']:.2f} "
          f"[{wc['LSI']['min']:.2f} to {wc['LSI']['max']:.2f}]")
    print(f"    Scaling (LSI>1.5):    {wc['LSI']['pct_scaling']:.1f}%")
    print(f"    Corrosive (LSI<-1.0): {wc['LSI']['pct_corrosive']:.1f}%")
    print(f"    Optimal:              {wc['LSI']['pct_optimal']:.1f}%")
    print(f"  RSI: {wc['RSI']['mean']:.2f} ± {wc['RSI']['std']:.2f}")
    
    risk = report["risk_distribution"]
    print(f"\n  --- Risk Distribution ---")
    for level in ["LOW", "MODERATE", "HIGH", "CRITICAL"]:
        bar = "█" * int(risk.get(level, 0) / 2)
        print(f"    {level:10s}: {risk.get(level, 0):5.1f}% {bar}")
    
    dosing = report["dosing"]
    print(f"\n  --- Dosing Performance ---")
    print(f"  Total cost: ₹{dosing['total_chemical_cost_inr']:,.0f}")
    print(f"  Daily avg:  ₹{dosing['daily_avg_cost_inr']:,.0f}/day")
    print(f"  Preemptive: {dosing['preemptive_pct']:.1f}% of decisions")
    print(f"  Safety overrides: {dosing['safety_override_pct']:.1f}%")
    
    print(f"\n  --- Chemical Usage (total kg) ---")
    for name, kg in dosing["per_chemical_kg"].items():
        print(f"    {name:25s}: {kg:8.2f} kg")
    
    print(f"\n  --- Chemical Adequacy ---")
    for name, status in report["chemical_adequacy"].items():
        adequate = status.get("ADEQUATE", 0)
        low = status.get("LOW", 0)
        critical = status.get("CRITICAL", 0)
        over = status.get("OVERDOSED", 0)
        print(f"    {name:25s}: OK={adequate:5.1f}% LOW={low:5.1f}% "
              f"CRIT={critical:5.1f}% OVER={over:5.1f}%")
    
    print("\n" + "="*70)


def main():
    """Main simulation entry point."""
    # Find data file
    data_paths = [
        '/mnt/project/Parameters_5K.csv',
        '/mnt/user-data/uploads/Parameters_5K.csv',
        'Parameters_5K.csv',
    ]
    
    csv_path = None
    for p in data_paths:
        if os.path.exists(p):
            csv_path = p
            break
    
    if csv_path is None:
        logger.error("Parameters_5K.csv not found!")
        sys.exit(1)
    
    # Load data
    data = load_and_prepare_data(csv_path)
    
    # Run simulation (no Chronos - use statistical fallback)
    # Set max_cycles for testing; remove limit for full run
    max_cycles = int(sys.argv[1]) if len(sys.argv) > 1 else len(data)
    
    report, results = run_simulation(
        data, 
        max_cycles=max_cycles,
        enable_forecasting=True,  # Will use statistical fallback if Chronos unavailable
    )
    
    # Print report
    print_report(report)
    
    # Save results
    output_dir = '/mnt/user-data/outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save report JSON
    report_path = os.path.join(output_dir, 'simulation_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"\nReport saved to {report_path}")
    
    # Save cycle results CSV
    if results:
        results_df = pd.DataFrame(results)
        results_path = os.path.join(output_dir, 'simulation_cycles.csv')
        results_df.to_csv(results_path, index=False)
        logger.info(f"Cycle data saved to {results_path}")
    
    return report


if __name__ == "__main__":
    main()
