"""
TGF Backtesting Framework
===========================
Answers "Would this autonomous system have performed better than
human operators?" — the key investor question.

CRITICAL DESIGN NOTE:
The previous version compared TGF against (TGF_cost × 1.20) — which is
scientifically meaningless. The Wilcoxon test always shows p≈0 because
the difference is deterministic. Any investor who understands statistics
would catch this immediately.

This version computes a REAL manual dosing baseline:
- Manual operators check 2-3× per day (not every 5 min)
- They respond REACTIVELY to out-of-range values
- They use fixed dose rates (no optimization)
- They have no forecasting capability
- Chemical waste comes from delayed response + overshoot

Usage:
    python -m tgf_dosing.validation.backtester --data data/Parameters_5K.csv
"""
import os
import sys
import json
import logging
import argparse
import numpy as np
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ManualDosingSimulator:
    """
    Simulates what a human operator would dose given the same sensor readings.

    Manual operation characteristics:
    - Checks water 2-3× per day (every 4-8 hours, ~48-96 cycles at 5-min intervals)
    - Responds only to OUT-OF-RANGE values (reactive, not preemptive)
    - Uses fixed dose rates from chemical supplier recommendations
    - Cannot forecast — only sees current readings
    - Over-doses by ~15-30% as safety margin (industry norm)
    - Does NOT track 7 chemicals individually — uses blanket programs
    """

    def __init__(self, tower_config):
        self.tower = tower_config
        self.check_interval = 72  # cycles between operator checks (~6 hours)
        self.last_check_cycle = 0
        self.overshoot_factor = 1.25  # operators dose 25% above target as buffer
        self.current_doses = {}  # what operator last set (stays until next check)

        # Initialize with supplier-recommended fixed doses
        for name, chem in self.tower.chemicals.items():
            # Manual operators use a constant feed rate based on target ppm
            volume_liters = self.tower.holding_volume_m3 * 1000
            target_kg_per_cycle = (chem.target_ppm * volume_liters /
                                   (chem.active_fraction * 1e6)) * 0.001  # tiny maintenance
            self.current_doses[name] = target_kg_per_cycle * self.overshoot_factor

    def compute_cycle_cost(self, cycle_index: int, ph: float,
                           conductivity: float, temperature: float,
                           orp: float, coc: float, lsi: float) -> float:
        """
        Compute what manual dosing would cost for this cycle.

        Manual operators:
        - Only adjust doses at check intervals
        - React to current readings (not forecasts)
        - Use simple threshold rules
        """
        # Is this a "check" cycle? (operator looks at readings)
        if cycle_index - self.last_check_cycle >= self.check_interval:
            self.last_check_cycle = cycle_index
            self._operator_adjustment(ph, conductivity, orp, lsi, coc)

        # Calculate cost from current fixed doses
        total_cost = 0.0
        for name, kg in self.current_doses.items():
            if name in self.tower.chemicals:
                total_cost += kg * self.tower.chemicals[name].cost_per_kg
        return total_cost

    def _operator_adjustment(self, ph: float, conductivity: float,
                             orp: float, lsi: float, coc: float):
        """
        Simulate operator response to current readings.
        Operators use simple rules:
        - High LSI → increase scale inhibitor
        - Low ORP → slug biocide
        - High conductivity → increase blowdown (reduces all chemicals)
        """
        for name, chem in self.tower.chemicals.items():
            volume_liters = self.tower.holding_volume_m3 * 1000
            base_rate = (chem.target_ppm * volume_liters /
                         (chem.active_fraction * 1e6)) * 0.001

            multiplier = self.overshoot_factor  # baseline overshoot

            # Scaling response (delayed, reactive)
            if lsi > 1.5:
                if chem.function.value in ("scale_inhibitor", "scale_corrosion"):
                    multiplier *= 1.5  # operator cranks up inhibitor
            elif lsi > 2.0:
                if chem.function.value in ("scale_inhibitor", "scale_corrosion"):
                    multiplier *= 2.0  # panic dosing

            # Low ORP → biocide boost
            if orp < 400:
                if "biocide" in chem.function.value:
                    multiplier *= 1.8

            # High conductivity → everything gets diluted by blowdown
            if conductivity > 4000:
                multiplier *= 1.3  # compensate for blowdown dilution

            # pH out of range → adjuster boost
            if ph > 8.5 or ph < 7.0:
                if chem.function.value == "ph_adjuster":
                    multiplier *= 1.5

            self.current_doses[name] = base_rate * multiplier


class Backtester:
    """
    Walk-forward backtesting of TGF dosing decisions against
    realistic manual operation simulation.
    """

    def __init__(self, csv_path: str, tower_config=None):
        self.csv_path = csv_path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from config.tower_config import AQUATECH_850_TPD, DEFAULT_LIMITS
        from core.dosing_controller import DosingController

        self.tower = tower_config or AQUATECH_850_TPD

    def run(self, n_folds: int = 5) -> Dict:
        """Run walk-forward cross-validation backtest."""
        import pandas as pd

        logger.info(f"Loading data from {self.csv_path}")
        df = pd.read_csv(self.csv_path)

        rename_map = {
            'Total_Hardness_ppm': 'total_hardness',
            'Calcium_Hardness_ppm': 'calcium_hardness',
            'Total_Alkalinity_ppm': 'total_alkalinity',
            'Conductivity_uS_cm': 'conductivity',
            'TDS_ppm': 'tds',
        }
        df = df.rename(columns=rename_map)

        required = ['pH', 'conductivity']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        df['temperature'] = df.get('Temperature_C', 32.0)
        if isinstance(df['temperature'], (int, float)):
            df['temperature'] = 32.0
        df['temperature'] = df['temperature'].fillna(32.0)

        df['orp'] = df.get('ORP_mV', 400.0)
        if isinstance(df['orp'], (int, float)):
            df['orp'] = 400.0
        df['orp'] = df['orp'].fillna(400.0)

        df['tds'] = df.get('tds', np.nan)

        logger.info(f"Backtesting on {len(df)} rows with {n_folds} folds")

        fold_size = len(df) // (n_folds + 1)
        results_per_fold = []

        for fold in range(n_folds):
            train_end = fold_size * (fold + 2)
            test_start = train_end
            test_end = min(test_start + fold_size, len(df))
            if test_end <= test_start:
                break
            fold_result = self._run_fold(
                df.iloc[:train_end], df.iloc[test_start:test_end], fold + 1)
            results_per_fold.append(fold_result)

        return self._aggregate_results(results_per_fold)

    def _run_fold(self, df_train, df_test, fold_num: int) -> Dict:
        """Run one fold with TGF controller AND manual simulator in parallel."""
        from core.dosing_controller import DosingController
        from core.physics_engine import PhysicsEngine

        controller = DosingController(
            tower_config=self.tower, enable_forecasting=False)
        physics = PhysicsEngine(self.tower)
        manual_sim = ManualDosingSimulator(self.tower)

        tgf_costs = []
        manual_costs = []
        lsi_values = []
        risk_levels = []
        preemptive_count = 0

        for idx, row in df_test.iterrows():
            try:
                result = controller.run_cycle(
                    ph=float(row['pH']),
                    conductivity=float(row['conductivity']),
                    temperature=float(row.get('temperature', 32.0)),
                    orp=float(row.get('orp', 400.0)),
                    tds=float(row['tds']) if 'tds' in row and not np.isnan(
                        row.get('tds', np.nan)) else None,
                    calcium_hardness=float(row.get('calcium_hardness', np.nan))
                        if not np.isnan(row.get('calcium_hardness', np.nan)) else None,
                    total_alkalinity=float(row.get('total_alkalinity', np.nan))
                        if not np.isnan(row.get('total_alkalinity', np.nan)) else None,
                    total_hardness=float(row.get('total_hardness', np.nan))
                        if not np.isnan(row.get('total_hardness', np.nan)) else None,
                )

                tgf_cost = result.total_chemical_cost_inr
                tgf_costs.append(tgf_cost)

                coc = physics.estimate_coc(float(row['conductivity']))
                lsi = result.risk_assessment.lsi
                lsi_values.append(lsi)
                risk_levels.append(result.risk_assessment.risk_level)

                # Manual baseline: what would an operator have dosed?
                cycle_idx = len(tgf_costs)
                manual_cost = manual_sim.compute_cycle_cost(
                    cycle_index=cycle_idx,
                    ph=float(row['pH']),
                    conductivity=float(row['conductivity']),
                    temperature=float(row.get('temperature', 32.0)),
                    orp=float(row.get('orp', 400.0)),
                    coc=coc, lsi=lsi,
                )
                manual_costs.append(manual_cost)

                if result.safe_decision.preemptive:
                    preemptive_count += 1

            except Exception as e:
                logger.debug(f"Fold {fold_num} cycle error: {e}")
                continue

        if not tgf_costs:
            return {"fold": fold_num, "error": "No successful cycles"}

        tgf_arr = np.array(tgf_costs)
        manual_arr = np.array(manual_costs)
        lsi_arr = np.array(lsi_values)

        stats = self._compute_statistics(tgf_arr, manual_arr)

        savings_pct = float(
            (np.sum(manual_arr) - np.sum(tgf_arr)) /
            max(np.sum(manual_arr), 1) * 100
        )

        return {
            "fold": fold_num,
            "test_size": len(df_test),
            "successful_cycles": len(tgf_costs),
            "tgf_total_cost": float(np.sum(tgf_arr)),
            "manual_total_cost": float(np.sum(manual_arr)),
            "savings_pct": savings_pct,
            "tgf_mean_cycle_cost": float(np.mean(tgf_arr)),
            "manual_mean_cycle_cost": float(np.mean(manual_arr)),
            "lsi_in_range_pct": float(
                np.mean((lsi_arr >= -0.5) & (lsi_arr <= 1.5)) * 100),
            "lsi_mean": float(np.mean(lsi_arr)),
            "lsi_std": float(np.std(lsi_arr)),
            "preemptive_pct": float(
                preemptive_count / max(len(tgf_costs), 1) * 100),
            "risk_distribution": {
                level: int(risk_levels.count(level))
                for level in ["LOW", "MODERATE", "HIGH", "CRITICAL"]
            },
            **stats,
        }

    def _compute_statistics(self, tgf: np.ndarray,
                            manual: np.ndarray) -> Dict:
        """Compute statistical significance — now meaningful because
        tgf and manual are independently computed."""
        result = {}

        try:
            from scipy.stats import wilcoxon
            stat, p_value = wilcoxon(tgf, manual, alternative='less')
            result["wilcoxon_statistic"] = float(stat)
            result["wilcoxon_p_value"] = float(p_value)
            result["statistically_significant"] = p_value < 0.05
        except Exception as e:
            result["wilcoxon_error"] = str(e)

        # Cohen's d
        diff = manual - tgf
        pooled_std = np.sqrt((np.var(tgf) + np.var(manual)) / 2)
        if pooled_std > 0:
            result["cohens_d"] = round(float(np.mean(diff) / pooled_std), 3)
        else:
            result["cohens_d"] = 0.0

        # Bootstrap CI for savings percentage
        try:
            n_bootstrap = 1000
            savings = []
            n = len(tgf)
            rng = np.random.default_rng(42)
            for _ in range(n_bootstrap):
                idx = rng.integers(0, n, size=n)
                s = ((np.sum(manual[idx]) - np.sum(tgf[idx])) /
                     max(np.sum(manual[idx]), 1) * 100)
                savings.append(s)
            result["bootstrap_savings_ci_95"] = [
                round(float(np.percentile(savings, 2.5)), 2),
                round(float(np.percentile(savings, 97.5)), 2),
            ]
        except Exception:
            pass

        return result

    def _aggregate_results(self, fold_results: List[Dict]) -> Dict:
        valid = [f for f in fold_results if "error" not in f]
        if not valid:
            return {"error": "All folds failed", "folds": fold_results}

        avg_savings = np.mean([f["savings_pct"] for f in valid])
        avg_lsi_range = np.mean([f["lsi_in_range_pct"] for f in valid])
        total_tgf = sum(f["tgf_total_cost"] for f in valid)
        total_manual = sum(f["manual_total_cost"] for f in valid)
        sig_folds = sum(1 for f in valid
                        if f.get("statistically_significant", False))

        # Get bootstrap CI range across folds
        ci_low = np.mean([f.get("bootstrap_savings_ci_95", [0, 0])[0]
                          for f in valid])
        ci_high = np.mean([f.get("bootstrap_savings_ci_95", [0, 0])[1]
                           for f in valid])

        return {
            "summary": {
                "n_folds": len(valid),
                "avg_savings_pct": round(float(avg_savings), 2),
                "savings_ci_95": [round(ci_low, 2), round(ci_high, 2)],
                "avg_lsi_in_range_pct": round(float(avg_lsi_range), 2),
                "total_tgf_cost_inr": round(total_tgf, 0),
                "total_manual_cost_inr": round(total_manual, 0),
                "total_savings_inr": round(total_manual - total_tgf, 0),
                "folds_statistically_significant": sig_folds,
                "comparison_method": "Manual dosing simulation (6h check interval, 25% overshoot, reactive only)",
                "conclusion": (
                    f"TGF saves {avg_savings:.1f}% vs manual operation "
                    f"(95% CI: [{ci_low:.1f}%, {ci_high:.1f}%], "
                    f"p<0.05 in {sig_folds}/{len(valid)} folds)"
                ),
            },
            "folds": fold_results,
        }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="TGF Backtest Framework")
    parser.add_argument("--data", required=True, help="Path to Parameters_5K.csv")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')

    bt = Backtester(args.data)
    report = bt.run(n_folds=args.folds)

    if "summary" in report:
        s = report["summary"]
        print("\n" + "=" * 60)
        print("  TGF BACKTEST RESULTS")
        print("=" * 60)
        print(f"  {s['conclusion']}")
        print(f"  Method: {s['comparison_method']}")
        print(f"  Total savings: Rs.{s['total_savings_inr']:,.0f}")
        print(f"  LSI in-range:  {s['avg_lsi_in_range_pct']:.1f}%")
        print("=" * 60)

    output_path = args.output or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "outputs", "backtest_report.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {output_path}")


if __name__ == "__main__":
    main()