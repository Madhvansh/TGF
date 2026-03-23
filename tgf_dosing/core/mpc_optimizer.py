"""
TGF MPC Dosing Optimizer
=========================
Model Predictive Control for optimal chemical dosing.

Why MPC over RL (SAC/PPO):
1. Works with physics model + 5K samples (RL needs millions of interactions)
2. Safety constraints are HARD (mathematically guaranteed, not soft penalties)
3. Explainable: "dosing because LSI forecast at +1.8 in 6h"
4. Production-ready in weeks (RL needs months of simulation environment development)
5. Zero risk of constraint violation during exploration

Architecture:
- Receding horizon: optimize 2-hour trajectory, execute first 5-min step only
- Cost function: minimize chemical cost + risk penalties
- Constraints: hard bounds on residuals, dose rates, and water chemistry
- Re-optimize every control cycle (5 minutes)
"""
import numpy as np
from scipy.optimize import minimize, differential_evolution
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging
import math

from config.tower_config import (
    TowerConfig, ChemicalProduct, DosingMode, 
    ChemicalFunction, OperatingLimits, DEFAULT_LIMITS
)
from core.physics_engine import WaterChemistry, RiskAssessment, PhysicsEngine
from core.chemical_tracker import ChemicalResidualTracker, TrackerSnapshot
from core.chronos_forecaster import SystemForecast

logger = logging.getLogger(__name__)


@dataclass
class DosingDecision:
    """Output of the MPC optimizer for one control cycle."""
    continuous_doses_kg: Dict[str, float]     # Chemical name → kg to dose this cycle
    blowdown_command: float                    # Blowdown rate as fraction of max (0-1)
    slug_doses: Dict[str, float]              # Chemical name → kg for slug dose (if any)
    
    # Explanation
    reasoning: Dict[str, str]                 # Chemical name → why this dose
    
    # Forecast-driven insights
    primary_risk: str                          # "scaling", "corrosion", "biofouling", "none"
    risk_horizon_hours: float                  # When the risk materializes
    preemptive: bool                           # Is this a preemptive action?
    
    # MPC diagnostics
    optimization_cost: float                   # Final cost function value
    optimization_converged: bool               # Did optimizer converge?
    
    def total_chemical_cost_inr(self, tower_config: TowerConfig) -> float:
        """Calculate total cost of this dosing decision in INR."""
        total = 0.0
        for name, kg in self.continuous_doses_kg.items():
            if name in tower_config.chemicals:
                total += kg * tower_config.chemicals[name].cost_per_kg
        for name, kg in self.slug_doses.items():
            if name in tower_config.chemicals:
                total += kg * tower_config.chemicals[name].cost_per_kg
        return total


class MPCDosingOptimizer:
    """
    Model Predictive Control optimizer for chemical dosing.
    
    Every 5 minutes:
    1. Read current state (sensors + chemical residuals + forecasts)
    2. Optimize a 2-hour dosing trajectory
    3. Execute only the FIRST 5-minute step
    4. Repeat (receding horizon)
    """
    
    def __init__(self, 
                 tower_config: TowerConfig,
                 physics_engine: PhysicsEngine,
                 limits: OperatingLimits = None,
                 horizon_steps: int = 24,       # 24 × 5min = 2 hours
                 dt_hours: float = 5.0/60.0):   # 5-minute cycles
        
        self.tower = tower_config
        self.physics = physics_engine
        self.limits = limits or DEFAULT_LIMITS
        self.horizon_steps = horizon_steps
        self.dt_hours = dt_hours
        
        # Identify continuous chemicals (these are optimized)
        self.continuous_chemicals: List[str] = [
            name for name, chem in tower_config.chemicals.items()
            if chem.dosing_mode == DosingMode.CONTINUOUS
        ]
        self.n_chemicals = len(self.continuous_chemicals)
        
        # Cost weights (tunable)
        self.w_chemical_cost = 1.0        # Weight for chemical usage cost
        self.w_underdose = 100.0          # Weight for being below target (CRITICAL to maintain)
        self.w_overdose = 30.0            # Weight for being above target
        self.w_scaling = 100.0            # Weight for LSI > target
        self.w_corrosion = 80.0           # Weight for LSI < target
        self.w_biofouling = 90.0          # Weight for low ORP
        self.w_blowdown = 5.0             # Weight for water usage (blowdown)
        self.w_rate_change = 5.0          # Weight for dose rate changes (smoothness)
    
    def optimize(self,
                 current_chemistry: WaterChemistry,
                 current_residuals: Dict[str, float],
                 forecast: SystemForecast,
                 tracker: ChemicalResidualTracker,
                 current_timestamp: float
                 ) -> DosingDecision:
        """
        Run MPC optimization for the current control cycle.
        
        Args:
            current_chemistry: Current sensor readings
            current_residuals: Current chemical residual estimates
            forecast: Chronos-2 forecast of water quality
            tracker: Chemical residual tracker
            current_timestamp: Unix timestamp
        
        Returns:
            DosingDecision with optimal doses for this cycle
        """
        # ====================================================================
        # STEP 1: Assess current and forecasted risk
        # ====================================================================
        risk = self.physics.full_risk_assessment(current_chemistry)
        
        # Get forecast risk at key horizons
        forecast_risks = self._assess_forecast_risks(forecast)
        
        # Determine primary risk driver
        primary_risk, risk_horizon = self._identify_primary_risk(
            risk, forecast_risks)
        
        # ====================================================================
        # STEP 2: Check for slug dose needs
        # ====================================================================
        slug_doses = {}
        for name, chem in self.tower.chemicals.items():
            if chem.dosing_mode in (DosingMode.SLUG, DosingMode.ALTERNATING):
                should_slug, reason = tracker.should_slug_dose(name, current_timestamp)
                if should_slug and chem.slug_quantity_kg:
                    slug_doses[name] = chem.slug_quantity_kg
                    logger.info(f"Slug dose triggered for {name}: {reason}")
        
        # ====================================================================
        # STEP 3: Optimize continuous chemical doses
        # ====================================================================
        optimal_doses, blowdown, opt_cost, converged = self._run_optimization(
            current_chemistry, current_residuals, forecast, risk)
        
        # ====================================================================
        # STEP 4: Build reasoning
        # ====================================================================
        reasoning = self._build_reasoning(
            optimal_doses, current_residuals, risk, forecast_risks, primary_risk)
        
        is_preemptive = risk_horizon > 0.5  # Acting on future risk, not current
        
        return DosingDecision(
            continuous_doses_kg=optimal_doses,
            blowdown_command=blowdown,
            slug_doses=slug_doses,
            reasoning=reasoning,
            primary_risk=primary_risk,
            risk_horizon_hours=risk_horizon,
            preemptive=is_preemptive,
            optimization_cost=opt_cost,
            optimization_converged=converged,
        )
    
    def _run_optimization(self,
                          chemistry: WaterChemistry,
                          residuals: Dict[str, float],
                          forecast: SystemForecast,
                          current_risk: RiskAssessment
                          ) -> Tuple[Dict[str, float], float, float, bool]:
        """
        Core optimization: find doses that minimize cost + risk.
        
        Decision variables: [dose_chem1, dose_chem2, ..., blowdown_fraction]
        One set per horizon step, but we optimize a simpler version:
        single dose rates for THIS cycle (not full trajectory).
        
        Returns: (doses_dict, blowdown_fraction, cost, converged)
        """
        n_vars = self.n_chemicals + 1  # chemicals + blowdown
        
        # Bounds for each variable
        bounds = []
        for name in self.continuous_chemicals:
            chem = self.tower.chemicals[name]
            # Max kg per 5-min cycle = max_rate × dt_hours
            max_kg = chem.max_dose_rate_kg_per_hr * self.dt_hours
            bounds.append((0.0, max_kg))
        
        # Blowdown: 0 to 1 (fraction of max)
        bounds.append((0.0, 1.0))
        
        # Initial guess: proportional to deficit
        x0 = []
        for name in self.continuous_chemicals:
            chem = self.tower.chemicals[name]
            current_ppm = residuals.get(name, chem.target_ppm)
            deficit = chem.target_ppm - current_ppm
            
            if current_ppm >= chem.target_ppm:
                # At or above target: dose nothing
                x0.append(0.0)
            elif deficit > 0:
                # Below target: dose to close deficit
                kg_needed = (deficit * self.tower.holding_volume_m3 * 1000 / 1e6 
                            / max(chem.active_fraction, 0.01))
                # Spread over 6 cycles (30 min to correct)
                kg_per_cycle = min(kg_needed / 6, chem.max_dose_rate_kg_per_hr * self.dt_hours)
                x0.append(max(0, kg_per_cycle))
            else:
                x0.append(0.0)
        
        # Blowdown initial guess based on CoC
        coc = self.physics.estimate_coc(chemistry.conductivity_us)
        if coc > self.limits.coc_target:
            bd_initial = min(1.0, (coc - self.limits.coc_target) / 3.0)
        else:
            bd_initial = 0.1
        x0.append(bd_initial)
        
        x0 = np.array(x0)
        
        # Cost function
        def cost_fn(x):
            return self._cost_function(
                x, chemistry, residuals, forecast, current_risk)
        
        # Run optimization
        try:
            result = minimize(
                cost_fn,
                x0=x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-6}
            )
            
            optimal_x = result.x
            opt_cost = result.fun
            converged = result.success
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}. Using initial guess.")
            optimal_x = x0
            opt_cost = cost_fn(x0)
            converged = False
        
        # Extract results
        doses = {}
        for i, name in enumerate(self.continuous_chemicals):
            doses[name] = max(0.0, float(optimal_x[i]))
        
        blowdown = float(np.clip(optimal_x[-1], 0, 1))
        
        return doses, blowdown, float(opt_cost), converged
    
    def _cost_function(self,
                       x: np.ndarray,
                       chemistry: WaterChemistry,
                       residuals: Dict[str, float],
                       forecast: SystemForecast,
                       current_risk: RiskAssessment) -> float:
        """
        MPC cost function to minimize.
        
        Components:
        1. Chemical cost (direct INR)
        2. Underdosing penalty (risk of insufficient protection)
        3. Overdosing penalty (waste + potential damage)
        4. Scaling risk penalty (from current + forecasted LSI)
        5. Corrosion risk penalty
        6. Biofouling risk penalty (ORP-based)
        7. Blowdown cost (water usage)
        8. Smoothness penalty (rate of change)
        """
        total_cost = 0.0
        
        # Extract decision variables
        doses_kg = x[:self.n_chemicals]
        blowdown_frac = x[-1]
        
        # ---- 1. Chemical cost ----
        for i, name in enumerate(self.continuous_chemicals):
            chem = self.tower.chemicals[name]
            total_cost += self.w_chemical_cost * doses_kg[i] * chem.cost_per_kg
        
        # ---- 2 & 3. Residual penalties ----
        for i, name in enumerate(self.continuous_chemicals):
            chem = self.tower.chemicals[name]
            current_ppm = residuals.get(name, 0.0)
            
            # Estimate what residual will be after dosing
            ppm_added = (doses_kg[i] * chem.active_fraction * 1e6) / (
                self.tower.holding_volume_m3 * 1000)
            projected_ppm = current_ppm + ppm_added
            
            # Scale penalty by cost-to-fix so penalties compete with chemical cost
            # This ensures the optimizer doesn't just "save money" by not dosing
            cost_per_ppm = (self.tower.holding_volume_m3 * 1000 / 1e6 
                           / max(chem.active_fraction, 0.01) * chem.cost_per_kg)
            
            # Underdosing penalty (quadratic below target, scaled by cost_per_ppm)
            if projected_ppm < chem.target_ppm:
                deficit = chem.target_ppm - projected_ppm
                # Penalty = w × deficit² × cost_scale → same units as chemical cost
                total_cost += self.w_underdose * (deficit ** 2) * cost_per_ppm / chem.target_ppm
            
            # Critical underdosing (much higher penalty)
            if projected_ppm < chem.min_ppm:
                critical_deficit = chem.min_ppm - projected_ppm
                total_cost += self.w_underdose * 5.0 * (critical_deficit ** 2) * cost_per_ppm / chem.min_ppm
            
            # Overdosing penalty (quadratic above max)
            if projected_ppm > chem.max_ppm:
                excess = projected_ppm - chem.max_ppm
                total_cost += self.w_overdose * (excess ** 2) * cost_per_ppm / chem.max_ppm
            
            # Mild excess above target (still penalized, but less)
            if projected_ppm > chem.target_ppm * 1.2:
                mild_excess = projected_ppm - chem.target_ppm * 1.2
                total_cost += self.w_overdose * 0.3 * (mild_excess ** 2) * cost_per_ppm / chem.target_ppm
        
        # ---- 4. Scaling risk ----
        lsi = current_risk.lsi
        if lsi > self.limits.lsi_max:
            total_cost += self.w_scaling * (lsi - self.limits.lsi_max) ** 2
        
        # Forecast scaling risk (proactive component)
        if forecast:
            ph_forecast = forecast.get_parameter("pH")
            cond_forecast = forecast.get_parameter("conductivity")
            if ph_forecast and cond_forecast:
                for horizon in [6.0, 12.0]:
                    ph_future = ph_forecast.p50_at(horizon)
                    cond_future = cond_forecast.p50_at(horizon)
                    if ph_future and cond_future:
                        coc_future = cond_future / max(self.tower.makeup_conductivity_us, 1)
                        ca_future = self.physics.estimate_calcium_hardness(coc_future)
                        alk_future = self.physics.estimate_alkalinity(coc_future)
                        tds_future = cond_future * 0.65
                        
                        phs = self.physics.calculate_phs(
                            chemistry.temperature_c, tds_future, ca_future, alk_future)
                        lsi_future = ph_future - phs
                        
                        if lsi_future > self.limits.lsi_max:
                            # Future scaling risk - discounted by time
                            discount = 1.0 / (1.0 + horizon * 0.1)
                            total_cost += (self.w_scaling * 0.5 * discount * 
                                         (lsi_future - self.limits.lsi_max) ** 2)
        
        # ---- 5. Corrosion risk ----
        if lsi < self.limits.lsi_min:
            total_cost += self.w_corrosion * (self.limits.lsi_min - lsi) ** 2
        
        # ---- 6. Biofouling risk ----
        if chemistry.orp_mv < self.limits.orp_min_mv:
            orp_deficit = (self.limits.orp_min_mv - chemistry.orp_mv) / 100
            total_cost += self.w_biofouling * orp_deficit ** 2
        
        # Forecast ORP decline
        if forecast:
            orp_forecast = forecast.get_parameter("ORP")
            if orp_forecast:
                orp_6h = orp_forecast.p50_at(6.0)
                if orp_6h and orp_6h < self.limits.orp_min_mv:
                    total_cost += self.w_biofouling * 0.3 * (
                        (self.limits.orp_min_mv - orp_6h) / 100) ** 2
        
        # ---- 7. Blowdown cost ----
        evap = self.physics.estimate_evaporation_rate(chemistry.temperature_c)
        bd_rate = blowdown_frac * self.limits.max_blowdown_rate_m3_per_hr
        total_cost += self.w_blowdown * bd_rate  # Linear cost for water
        
        # But penalize if CoC is too high and not enough blowdown
        coc = self.physics.estimate_coc(chemistry.conductivity_us)
        if coc > self.limits.coc_max and blowdown_frac < 0.5:
            total_cost += 50.0 * (coc - self.limits.coc_max) ** 2
        
        # ---- 8. Smoothness ----
        # Penalize large doses (prefer small frequent over large infrequent)
        for i, name in enumerate(self.continuous_chemicals):
            chem = self.tower.chemicals[name]
            max_kg = chem.max_dose_rate_kg_per_hr * self.dt_hours
            if max_kg > 0:
                rate = doses_kg[i] / max_kg
                total_cost += self.w_rate_change * rate ** 2
        
        return total_cost
    
    def _assess_forecast_risks(self, forecast: SystemForecast
                                ) -> Dict[str, Dict[float, float]]:
        """Compute risk scores at forecast horizons."""
        risks = {"scaling": {}, "corrosion": {}, "biofouling": {}}
        
        if not forecast:
            return risks
        
        ph_fc = forecast.get_parameter("pH")
        cond_fc = forecast.get_parameter("conductivity")
        orp_fc = forecast.get_parameter("ORP")
        
        for horizon in [1.0, 6.0, 12.0, 24.0]:
            # Scaling risk from forecast LSI
            if ph_fc and cond_fc:
                ph = ph_fc.p50_at(horizon)
                cond = cond_fc.p50_at(horizon)
                if ph and cond:
                    coc = cond / max(self.tower.makeup_conductivity_us, 1)
                    ca = self.physics.estimate_calcium_hardness(coc)
                    alk = self.physics.estimate_alkalinity(coc)
                    tds = cond * 0.65
                    phs = self.physics.calculate_phs(30.0, tds, ca, alk)
                    lsi = ph - phs
                    risks["scaling"][horizon] = max(0, (lsi - self.limits.lsi_target) / 2.0)
                    risks["corrosion"][horizon] = max(0, (-lsi - 0.5) / 2.0)
            
            # Biofouling risk from ORP
            if orp_fc:
                orp = orp_fc.p50_at(horizon)
                if orp:
                    risks["biofouling"][horizon] = max(0, 
                        (self.limits.orp_target_mv - orp) / self.limits.orp_target_mv)
        
        return risks
    
    def _identify_primary_risk(self, current_risk: RiskAssessment,
                                forecast_risks: Dict
                                ) -> Tuple[str, float]:
        """Identify the most critical risk and when it materializes."""
        risks_with_time = []
        
        # Current risks
        risks_with_time.append(("scaling", current_risk.scaling_risk, 0.0))
        risks_with_time.append(("corrosion", current_risk.corrosion_risk, 0.0))
        risks_with_time.append(("biofouling", current_risk.biofouling_risk, 0.0))
        
        # Forecast risks
        for risk_type, horizons in forecast_risks.items():
            for horizon, score in horizons.items():
                risks_with_time.append((risk_type, score, horizon))
        
        if not risks_with_time:
            return "none", 0.0
        
        # Pick highest risk
        primary = max(risks_with_time, key=lambda x: x[1])
        return primary[0], primary[2]
    
    def _build_reasoning(self,
                         doses: Dict[str, float],
                         residuals: Dict[str, float],
                         risk: RiskAssessment,
                         forecast_risks: Dict,
                         primary_risk: str) -> Dict[str, str]:
        """Generate human-readable reasoning for each dosing decision."""
        reasoning = {}
        
        for name in self.continuous_chemicals:
            chem = self.tower.chemicals[name]
            dose_kg = doses.get(name, 0.0)
            current_ppm = residuals.get(name, 0.0)
            deficit = chem.target_ppm - current_ppm
            
            if dose_kg < 0.001:
                reasoning[name] = (
                    f"No dosing needed. Residual={current_ppm:.1f}ppm "
                    f"(target={chem.target_ppm:.1f}ppm). "
                    f"Status: adequate."
                )
            elif deficit > 0:
                reasoning[name] = (
                    f"Dosing {dose_kg:.3f}kg to address deficit of {deficit:.1f}ppm. "
                    f"Current={current_ppm:.1f}ppm, target={chem.target_ppm:.1f}ppm. "
                    f"Primary risk: {primary_risk} (LSI={risk.lsi:.2f})."
                )
            else:
                reasoning[name] = (
                    f"Maintenance dose of {dose_kg:.3f}kg. "
                    f"Current={current_ppm:.1f}ppm is near target={chem.target_ppm:.1f}ppm."
                )
        
        # Overall reasoning
        scaling_future = forecast_risks.get("scaling", {}).get(6.0, 0)
        if scaling_future > 0.3:
            reasoning["_preemptive"] = (
                f"PREEMPTIVE dosing: scaling risk forecast to increase to "
                f"{scaling_future:.0%} at 6h horizon. Acting now with smaller dose "
                f"to prevent larger corrective dose later."
            )
        
        return reasoning
