"""
TGF Dosing Controller
======================
Main integration class that orchestrates the complete control loop:

    Sensors → Forecaster → Physics Engine → Chemical Tracker →
    MPC Optimizer → Safety Layer → Actuator Commands

This is the top-level controller that runs every 5 minutes.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
import logging
import time
import json

from config.tower_config import (
    TowerConfig, OperatingLimits, DEFAULT_LIMITS, AQUATECH_850_TPD
)
from core.physics_engine import PhysicsEngine, WaterChemistry, RiskAssessment
from core.chemical_tracker import ChemicalResidualTracker, TrackerSnapshot
from core.chronos_forecaster import ChronosForecaster, SystemForecast
from core.mpc_optimizer import MPCDosingOptimizer, DosingDecision
from core.safety_layer import SafetyLayer, SafetyReport

logger = logging.getLogger(__name__)


@dataclass
class ControlCycleResult:
    """Complete result of one 5-minute control cycle."""
    timestamp: float
    cycle_number: int
    
    # Inputs
    chemistry: WaterChemistry
    
    # Components
    risk_assessment: RiskAssessment
    forecast: Optional[SystemForecast]
    tracker_snapshot: TrackerSnapshot
    
    # Decision
    mpc_decision: DosingDecision
    safe_decision: DosingDecision
    safety_report: SafetyReport
    
    # Summary
    total_chemical_cost_inr: float
    
    def summary_dict(self) -> dict:
        """Compact summary for logging/dashboard."""
        return {
            "cycle": self.cycle_number,
            "timestamp": self.timestamp,
            "pH": self.chemistry.ph,
            "conductivity": self.chemistry.conductivity_us,
            "temperature": self.chemistry.temperature_c,
            "ORP": self.chemistry.orp_mv,
            "LSI": self.risk_assessment.lsi,
            "RSI": self.risk_assessment.rsi,
            "risk_level": self.risk_assessment.risk_level,
            "primary_risk": self.safe_decision.primary_risk,
            "preemptive": self.safe_decision.preemptive,
            "doses": {k: round(v, 4) for k, v in self.safe_decision.continuous_doses_kg.items()},
            "blowdown": round(self.safe_decision.blowdown_command, 3),
            "slug_doses": {k: round(v, 1) for k, v in self.safe_decision.slug_doses.items()},
            "cost_inr": round(self.total_chemical_cost_inr, 2),
            "safety_overrides": len(self.safety_report.overrides),
            "emergency_stop": self.safety_report.emergency_stop,
        }


class DosingController:
    """
    Main TGF dosing controller.
    
    Usage:
        controller = DosingController(tower_config)
        
        # Every 5 minutes:
        result = controller.run_cycle(
            ph=7.82, conductivity=2450, temperature=32, orp=648
        )
        
        # Execute the dosing commands:
        for chemical, kg in result.safe_decision.continuous_doses_kg.items():
            pump.dose(chemical, kg)
        
        blowdown_valve.set(result.safe_decision.blowdown_command)
    """
    
    def __init__(self,
                 tower_config: TowerConfig = None,
                 limits: OperatingLimits = None,
                 chronos_model_size: str = "base",
                 enable_forecasting: bool = True):
        """
        Initialize all sub-components.
        
        Args:
            tower_config: Tower specifications and chemical program
            limits: Safe operating limits
            chronos_model_size: "tiny", "small", "base", "large"
            enable_forecasting: If False, skip Chronos (faster for testing)
        """
        self.tower = tower_config or AQUATECH_850_TPD
        self.limits = limits or DEFAULT_LIMITS
        
        # Initialize components
        self.physics = PhysicsEngine(self.tower)
        self.tracker = ChemicalResidualTracker(self.tower)
        self.safety = SafetyLayer(self.tower, self.limits)
        
        # MPC optimizer
        self.mpc = MPCDosingOptimizer(
            self.tower, self.physics, self.limits,
            horizon_steps=24,
            dt_hours=5.0/60.0
        )
        
        # Forecaster (optional)
        self.enable_forecasting = enable_forecasting
        if enable_forecasting:
            self.forecaster = ChronosForecaster(
                model_size=chronos_model_size,
                context_length=512,
            )
        else:
            self.forecaster = None
        
        # State tracking
        self.cycle_count = 0
        self.dt_hours = 5.0 / 60.0
        
        # History for analysis
        self.history: List[ControlCycleResult] = []
        self.max_history = 8640  # 30 days at 5-min intervals
        
        # Cumulative stats
        self.total_chemical_cost = 0.0
        self.total_water_saved_m3 = 0.0
        self.total_cycles_run = 0
        
        logger.info(f"DosingController initialized for tower: {self.tower.name}")
        logger.info(f"  Chemicals: {list(self.tower.chemicals.keys())}")
        logger.info(f"  Forecasting: {'enabled' if enable_forecasting else 'disabled'}")
    
    def run_cycle(self,
                  ph: float,
                  conductivity: float,
                  temperature: float,
                  orp: float,
                  timestamp: float = None,
                  # Optional measured values (override estimates)
                  tds: float = None,
                  calcium_hardness: float = None,
                  total_alkalinity: float = None,
                  total_hardness: float = None,
                  ) -> ControlCycleResult:
        """
        Execute one complete control cycle.
        
        This is THE main function called every 5 minutes.
        
        Args:
            ph: Current pH reading
            conductivity: Current conductivity (µS/cm)
            temperature: Current temperature (°C)
            orp: Current ORP (mV)
            timestamp: Unix timestamp (auto-generated if None)
            tds: Optional measured TDS
            calcium_hardness: Optional measured Ca hardness
            total_alkalinity: Optional measured alkalinity
            total_hardness: Optional measured total hardness
        
        Returns:
            ControlCycleResult with all decisions and reasoning
        """
        if timestamp is None:
            timestamp = time.time()
        
        self.cycle_count += 1
        
        # ================================================================
        # STEP 1: Build current water chemistry state
        # ================================================================
        chemistry = WaterChemistry(
            ph=ph,
            conductivity_us=conductivity,
            temperature_c=temperature,
            orp_mv=orp,
            tds_ppm=tds,
            calcium_hardness_ppm=calcium_hardness,
            total_alkalinity_ppm=total_alkalinity,
            total_hardness_ppm=total_hardness,
            timestamp=timestamp,
        )
        
        # ================================================================
        # STEP 2: Physics-based risk assessment
        # ================================================================
        risk = self.physics.full_risk_assessment(chemistry)
        
        # Compute water balance parameters
        coc = self.physics.estimate_coc(conductivity)
        evap_rate = self.physics.estimate_evaporation_rate(temperature)
        bd_rate = self.physics.estimate_blowdown_rate(coc, evap_rate)
        
        # ================================================================
        # STEP 3: Update chemical residual tracker
        # ================================================================
        # Use previous cycle's doses (tracker needs to know what was dosed LAST cycle)
        prev_doses = {}
        if self.history:
            prev_result = self.history[-1]
            prev_doses = prev_result.safe_decision.continuous_doses_kg
            # Add slug doses
            for name, kg in prev_result.safe_decision.slug_doses.items():
                prev_doses[name] = prev_doses.get(name, 0) + kg
        
        tracker_snapshot = self.tracker.update(
            dt_hours=self.dt_hours,
            coc=coc,
            evaporation_rate=evap_rate,
            blowdown_rate=bd_rate,
            temperature_c=temperature,
            lsi=risk.lsi,
            orp_mv=orp,
            pump_actions=prev_doses,
            current_timestamp=timestamp,
        )
        
        current_residuals = {
            name: state.estimated_ppm 
            for name, state in tracker_snapshot.chemicals.items()
        }
        
        # ================================================================
        # STEP 4: Update forecaster and generate forecasts
        # ================================================================
        forecast = None
        if self.forecaster is not None:
            self.forecaster.add_reading(timestamp, {
                "pH": ph,
                "conductivity": conductivity,
                "temperature": temperature,
                "ORP": orp,
            })
            
            if self.forecaster.has_enough_history(min_points=24):
                try:
                    forecast = self.forecaster.generate_forecast(
                        horizons_hours=[1.0, 6.0, 12.0, 24.0])
                except Exception as e:
                    logger.error(f"Forecast generation failed: {e}")
                    forecast = None
        
        # ================================================================
        # STEP 5: MPC optimization
        # ================================================================
        try:
            mpc_decision = self.mpc.optimize(
                current_chemistry=chemistry,
                current_residuals=current_residuals,
                forecast=forecast,
                tracker=self.tracker,
                current_timestamp=timestamp,
            )
        except Exception as e:
            logger.error(f"MPC optimization failed: {e}. Using zero doses.")
            mpc_decision = DosingDecision(
                continuous_doses_kg={name: 0.0 for name in self.tower.chemicals},
                blowdown_command=0.2,
                slug_doses={},
                reasoning={"_error": f"MPC failed: {e}"},
                primary_risk="unknown",
                risk_horizon_hours=0.0,
                preemptive=False,
                optimization_cost=float('inf'),
                optimization_converged=False,
            )
        
        # ================================================================
        # STEP 6: Safety layer
        # ================================================================
        safe_decision, safety_report = self.safety.apply(
            mpc_decision, chemistry, current_residuals, self.dt_hours)
        
        # ================================================================
        # STEP 7: Calculate costs and build result
        # ================================================================
        cycle_cost = safe_decision.total_chemical_cost_inr(self.tower)
        self.total_chemical_cost += cycle_cost
        self.total_cycles_run += 1
        
        result = ControlCycleResult(
            timestamp=timestamp,
            cycle_number=self.cycle_count,
            chemistry=chemistry,
            risk_assessment=risk,
            forecast=forecast,
            tracker_snapshot=tracker_snapshot,
            mpc_decision=mpc_decision,
            safe_decision=safe_decision,
            safety_report=safety_report,
            total_chemical_cost_inr=cycle_cost,
        )
        
        # Store history
        self.history.append(result)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        # Log summary
        if self.cycle_count % 12 == 0:  # Every hour
            self._log_hourly_summary()
        
        return result
    
    def calibrate_from_lab(self,
                           lab_results: Dict[str, float],
                           current_conductivity: float,
                           timestamp: float = None):
        """
        Update models from lab test results.
        
        Args:
            lab_results: Dict with measured values, e.g.:
                {"calcium_hardness": 450, "alkalinity": 120,
                 "AQUATREAT-2501": 11.5, "AQUATREAT-1196": 33.0}
            current_conductivity: Current conductivity for CoC calculation
            timestamp: Unix timestamp
        """
        ts = timestamp or time.time()
        
        # Calibrate physics engine (hardness/alkalinity estimates)
        self.physics.calibrate_from_lab(
            lab_calcium=lab_results.get("calcium_hardness"),
            lab_alkalinity=lab_results.get("alkalinity"),
            lab_hardness=lab_results.get("total_hardness"),
            current_conductivity=current_conductivity,
            timestamp=ts,
        )
        
        # Calibrate chemical tracker
        for name in self.tower.chemicals:
            if name in lab_results:
                self.tracker.calibrate_from_lab(name, lab_results[name])
        
        logger.info(f"Lab calibration applied: {lab_results}")
    
    def get_dashboard_state(self) -> dict:
        """Get current state for dashboard display."""
        if not self.history:
            return {"status": "NO_DATA"}
        
        latest = self.history[-1]
        
        return {
            "status": "RUNNING",
            "cycle": latest.cycle_number,
            "uptime_hours": (latest.cycle_number * self.dt_hours),
            "sensors": {
                "pH": latest.chemistry.ph,
                "conductivity": latest.chemistry.conductivity_us,
                "temperature": latest.chemistry.temperature_c,
                "ORP": latest.chemistry.orp_mv,
            },
            "indices": {
                "LSI": latest.risk_assessment.lsi,
                "RSI": latest.risk_assessment.rsi,
            },
            "risk": {
                "level": latest.risk_assessment.risk_level,
                "scaling": latest.risk_assessment.scaling_risk,
                "corrosion": latest.risk_assessment.corrosion_risk,
                "biofouling": latest.risk_assessment.biofouling_risk,
                "primary": latest.safe_decision.primary_risk,
                "preemptive": latest.safe_decision.preemptive,
            },
            "chemicals": {
                name: {
                    "estimated_ppm": state.estimated_ppm,
                    "target_ppm": state.target_ppm,
                    "status": state.status,
                    "confidence": state.confidence,
                }
                for name, state in latest.tracker_snapshot.chemicals.items()
            },
            "dosing": {
                name: round(kg, 4)
                for name, kg in latest.safe_decision.continuous_doses_kg.items()
            },
            "blowdown": latest.safe_decision.blowdown_command,
            "cost_today_inr": self._cost_last_24h(),
            "safety": {
                "overrides": len(latest.safety_report.overrides),
                "emergency_stop": latest.safety_report.emergency_stop,
                "pid_active": latest.safety_report.pid_active,
            },
            "water_balance": {
                "CoC": latest.tracker_snapshot.coc,
                "evaporation_m3_hr": latest.tracker_snapshot.evaporation_rate_m3_hr,
                "blowdown_m3_hr": latest.tracker_snapshot.blowdown_rate_m3_hr,
                "makeup_m3_hr": latest.tracker_snapshot.makeup_rate_m3_hr,
            }
        }
    
    def _cost_last_24h(self) -> float:
        """Sum chemical costs for the last 24 hours."""
        if not self.history:
            return 0.0
        cutoff = self.history[-1].timestamp - 86400
        return sum(r.total_chemical_cost_inr 
                   for r in self.history if r.timestamp > cutoff)
    
    def _log_hourly_summary(self):
        """Log a summary every hour."""
        if not self.history:
            return
        
        latest = self.history[-1]
        tracker_summary = self.tracker.summary()
        
        logger.info(
            f"\n{'='*60}\n"
            f"  HOURLY SUMMARY - Cycle {self.cycle_count}\n"
            f"{'='*60}\n"
            f"  pH={latest.chemistry.ph:.2f}  Cond={latest.chemistry.conductivity_us:.0f}  "
            f"Temp={latest.chemistry.temperature_c:.1f}°C  ORP={latest.chemistry.orp_mv:.0f}mV\n"
            f"  LSI={latest.risk_assessment.lsi:.2f}  RSI={latest.risk_assessment.rsi:.2f}  "
            f"Risk: {latest.risk_assessment.risk_level}\n"
            f"  CoC={latest.tracker_snapshot.coc:.1f}  "
            f"Evap={latest.tracker_snapshot.evaporation_rate_m3_hr:.1f}m³/hr  "
            f"BD={latest.tracker_snapshot.blowdown_rate_m3_hr:.1f}m³/hr\n"
            f"  Cost (last 24h): ₹{self._cost_last_24h():.0f}\n"
            f"{tracker_summary}\n"
            f"{'='*60}"
        )
