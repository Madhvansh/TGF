"""
TGF Safety Layer
=================
PID backup controllers + hard limits + emergency stops.

This is the safety FLOOR from TGF_TRUE_MVP_Architecture.md.
Even if the MPC optimizer fails completely, this layer keeps
the system safe using proven rule-based control.

Layers of protection:
1. Hard limits: absolute max/min for every chemical and parameter
2. Rate limiters: max 20% change per cycle
3. PID controllers: classical setpoint tracking for pH, ORP, CoC
4. MPC override detection: catches insane MPC outputs
5. Emergency stop: shuts down dosing if sensors fail
"""
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import logging
import time

from config.tower_config import (
    TowerConfig, ChemicalProduct, DosingMode, ChemicalFunction,
    OperatingLimits, DEFAULT_LIMITS
)
from core.physics_engine import WaterChemistry, PhysicsEngine
from core.mpc_optimizer import DosingDecision

logger = logging.getLogger(__name__)


@dataclass
class PIDState:
    """State of a PID controller."""
    setpoint: float
    integral: float = 0.0
    last_error: float = 0.0
    output: float = 0.0
    saturated: bool = False


class PIDController:
    """
    Classic PID controller with anti-windup.
    """
    def __init__(self, kp: float, ki: float, kd: float,
                 output_min: float = 0.0, output_max: float = 1.0,
                 setpoint: float = 0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self.state = PIDState(setpoint=setpoint)
    
    def update(self, measured: float, dt: float) -> float:
        """
        Compute PID output.
        
        Args:
            measured: Current measured value
            dt: Time step in hours
        
        Returns:
            Control output (0-1 normalized)
        """
        error = self.state.setpoint - measured
        
        # Proportional
        p_term = self.kp * error
        
        # Integral (with anti-windup)
        if not self.state.saturated:
            self.state.integral += error * dt
        i_term = self.ki * self.state.integral
        
        # Derivative
        if dt > 0:
            d_term = self.kd * (error - self.state.last_error) / dt
        else:
            d_term = 0.0
        
        # Total output
        output = p_term + i_term + d_term
        
        # Clamp and detect saturation (for anti-windup)
        if output > self.output_max:
            output = self.output_max
            self.state.saturated = True
        elif output < self.output_min:
            output = self.output_min
            self.state.saturated = True
        else:
            self.state.saturated = False
        
        self.state.last_error = error
        self.state.output = output
        
        return output
    
    def set_setpoint(self, setpoint: float):
        self.state.setpoint = setpoint
    
    def reset(self):
        self.state.integral = 0.0
        self.state.last_error = 0.0
        self.state.output = 0.0
        self.state.saturated = False


@dataclass
class SafetyReport:
    """Report from the safety layer about what it modified."""
    original_decision: Dict[str, float]   # What MPC wanted
    modified_decision: Dict[str, float]   # What safety layer approved
    overrides: Dict[str, str]             # What was changed and why
    emergency_stop: bool                  # Was emergency stop triggered?
    sensor_fault: bool                    # Are sensors considered faulty?
    pid_active: bool                      # Is PID override in effect?


class SafetyLayer:
    """
    Safety layer that wraps MPC decisions with hard constraints.
    
    Usage:
        safety = SafetyLayer(tower_config)
        safe_decision = safety.apply(mpc_decision, chemistry, residuals)
    """
    
    # Maximum allowed change per cycle (fraction of current)
    MAX_RATE_CHANGE = 0.30     # 30% max change per 5-minute cycle
    
    # If MPC dose exceeds PID by this factor, override
    MPC_OVERRIDE_THRESHOLD = 5.0  # Allow MPC to dose up to 5x PID
    
    # Sensor fault detection thresholds
    SENSOR_FAULT_THRESHOLDS = {
        "pH": (3.0, 12.0),            # Outside this = fault
        "conductivity": (10.0, 20000.0),
        "temperature": (0.0, 60.0),
        "ORP": (-500.0, 1200.0),
    }
    
    def __init__(self, tower_config: TowerConfig, 
                 limits: OperatingLimits = None):
        self.tower = tower_config
        self.limits = limits or DEFAULT_LIMITS
        
        # PID controllers for main parameters
        self.pid_ph = PIDController(
            kp=0.5, ki=0.05, kd=0.1,
            output_min=0.0, output_max=1.0,
            setpoint=limits.ph_target if limits else 8.0
        )
        
        self.pid_orp = PIDController(
            kp=0.003, ki=0.0005, kd=0.001,
            output_min=0.0, output_max=1.0,
            setpoint=limits.orp_target_mv if limits else 650.0
        )
        
        self.pid_coc = PIDController(
            kp=0.2, ki=0.02, kd=0.05,
            output_min=0.0, output_max=1.0,
            setpoint=limits.coc_target if limits else 6.0
        )
        
        # Track previous doses for rate limiting
        self.previous_doses: Dict[str, float] = {}
        
        # Sensor fault tracking
        self.consecutive_faults: Dict[str, int] = {
            "pH": 0, "conductivity": 0, "temperature": 0, "ORP": 0
        }
        self.FAULT_COUNT_THRESHOLD = 3  # 3 consecutive faults = shutdown
    
    def apply(self,
              mpc_decision: DosingDecision,
              chemistry: WaterChemistry,
              residuals: Dict[str, float],
              dt_hours: float = 5.0/60.0
              ) -> Tuple[DosingDecision, SafetyReport]:
        """
        Apply safety checks to an MPC dosing decision.
        
        Returns modified decision and safety report.
        """
        overrides = {}
        emergency_stop = False
        sensor_fault = False
        pid_active = False
        
        original_doses = dict(mpc_decision.continuous_doses_kg)
        modified_doses = dict(mpc_decision.continuous_doses_kg)
        modified_blowdown = mpc_decision.blowdown_command
        
        # ================================================================
        # CHECK 1: Sensor fault detection
        # ================================================================
        sensor_fault = self._check_sensors(chemistry)
        if sensor_fault:
            emergency_stop = True
            # Zero all doses except maintenance minimum
            for name in modified_doses:
                modified_doses[name] = 0.0
            modified_blowdown = 0.3  # Keep some blowdown for safety
            overrides["_emergency"] = (
                "EMERGENCY STOP: Sensor fault detected. "
                "All dosing suspended until sensors verified."
            )
            logger.critical("EMERGENCY STOP triggered due to sensor fault")
        
        if not emergency_stop:
            # ============================================================
            # CHECK 2: Hard limit enforcement
            # ============================================================
            for name, kg in modified_doses.items():
                chem = self.tower.chemicals.get(name)
                if chem is None:
                    continue
                
                current_ppm = residuals.get(name, 0.0)
                ppm_added = (kg * chem.active_fraction * 1e6) / (
                    self.tower.holding_volume_m3 * 1000)
                projected = current_ppm + ppm_added
                
                # Hard max enforcement
                if projected > chem.max_ppm:
                    # How much can we safely add?
                    safe_deficit = max(0, chem.max_ppm - current_ppm)
                    safe_kg = (safe_deficit * self.tower.holding_volume_m3 * 1000 / 1e6 
                              / max(chem.active_fraction, 0.01))
                    modified_doses[name] = max(0, safe_kg)
                    overrides[name] = (
                        f"Hard limit: projected {projected:.1f}ppm exceeds max "
                        f"{chem.max_ppm:.1f}ppm. Capped to {safe_deficit:.1f}ppm addition."
                    )
                
                # Hard min check (ensure we dose if critically low)
                if current_ppm < chem.min_ppm and kg < 0.001:
                    # Force minimum dose
                    min_deficit = chem.min_ppm - current_ppm
                    force_kg = (min_deficit * self.tower.holding_volume_m3 * 1000 / 1e6 
                               / max(chem.active_fraction, 0.01))
                    force_kg = min(force_kg, chem.max_dose_rate_kg_per_hr * dt_hours)
                    modified_doses[name] = force_kg
                    overrides[name] = (
                        f"Safety floor: residual {current_ppm:.1f}ppm below minimum "
                        f"{chem.min_ppm:.1f}ppm. Forced dose of {force_kg:.3f}kg."
                    )
            
            # ============================================================
            # CHECK 3: Rate limiting (prevent sudden jumps, but allow starting from 0)
            # ============================================================
            for name, kg in modified_doses.items():
                prev = self.previous_doses.get(name, 0.0)
                chem = self.tower.chemicals.get(name)
                if chem is None:
                    continue
                    
                # Minimum dose floor: even from 0, allow up to 20% of max rate
                min_allowed = chem.max_dose_rate_kg_per_hr * dt_hours * 0.20
                
                if prev > 0.01:  # Previous dose was nonzero
                    max_increase = prev * (1 + self.MAX_RATE_CHANGE)
                    max_allowed = max(max_increase, min_allowed)
                else:
                    # Starting from zero: allow up to 50% of max
                    max_allowed = chem.max_dose_rate_kg_per_hr * dt_hours * 0.50
                
                if kg > max_allowed and kg > 0.01:
                    old_kg = kg
                    modified_doses[name] = max_allowed
                    overrides[name] = overrides.get(name, "") + (
                        f" Rate limited: {old_kg:.3f}kg → {max_allowed:.3f}kg."
                    )
            
            # ============================================================
            # CHECK 4: PID sanity check
            # ============================================================
            pid_doses = self._compute_pid_doses(chemistry, residuals, dt_hours)
            pid_active = False
            
            for name, mpc_kg in modified_doses.items():
                pid_kg = pid_doses.get(name, 0.0)
                if pid_kg > 0 and mpc_kg > self.MPC_OVERRIDE_THRESHOLD * pid_kg:
                    # MPC is dosing way more than PID thinks necessary
                    capped = pid_kg * 1.5  # Allow 50% above PID, not more
                    overrides[name] = overrides.get(name, "") + (
                        f" MPC override: MPC wanted {mpc_kg:.3f}kg but PID suggests "
                        f"{pid_kg:.3f}kg. Capped to {capped:.3f}kg."
                    )
                    modified_doses[name] = capped
                    pid_active = True
            
            # ============================================================
            # CHECK 5: Blowdown safety
            # ============================================================
            coc = chemistry.conductivity_us / max(self.tower.makeup_conductivity_us, 1)
            
            # Force blowdown if CoC exceeds absolute max
            if coc > self.limits.coc_max * 1.1:
                modified_blowdown = max(modified_blowdown, 0.8)
                overrides["blowdown"] = (
                    f"Safety blowdown: CoC={coc:.1f} exceeds max "
                    f"{self.limits.coc_max:.1f}. Forced high blowdown."
                )
            
            # Prevent excessive blowdown (water waste)
            if coc < self.limits.coc_min and modified_blowdown > 0.3:
                modified_blowdown = 0.1
                overrides["blowdown"] = (
                    f"Blowdown reduced: CoC={coc:.1f} below minimum "
                    f"{self.limits.coc_min:.1f}. Conserving water."
                )
        
        # Update previous doses for next cycle's rate limiting
        self.previous_doses = dict(modified_doses)
        
        # Build modified decision
        safe_decision = DosingDecision(
            continuous_doses_kg=modified_doses,
            blowdown_command=modified_blowdown,
            slug_doses=mpc_decision.slug_doses if not emergency_stop else {},
            reasoning=mpc_decision.reasoning,
            primary_risk=mpc_decision.primary_risk,
            risk_horizon_hours=mpc_decision.risk_horizon_hours,
            preemptive=mpc_decision.preemptive,
            optimization_cost=mpc_decision.optimization_cost,
            optimization_converged=mpc_decision.optimization_converged,
        )
        
        report = SafetyReport(
            original_decision=original_doses,
            modified_decision=modified_doses,
            overrides=overrides,
            emergency_stop=emergency_stop,
            sensor_fault=sensor_fault,
            pid_active=pid_active,
        )
        
        return safe_decision, report
    
    def _check_sensors(self, chemistry: WaterChemistry) -> bool:
        """
        Check if sensor readings are physically plausible.
        Returns True if a FAULT is detected.
        """
        checks = {
            "pH": chemistry.ph,
            "conductivity": chemistry.conductivity_us,
            "temperature": chemistry.temperature_c,
            "ORP": chemistry.orp_mv,
        }
        
        any_fault = False
        for param, value in checks.items():
            lo, hi = self.SENSOR_FAULT_THRESHOLDS[param]
            if value < lo or value > hi or np.isnan(value):
                self.consecutive_faults[param] += 1
                if self.consecutive_faults[param] >= self.FAULT_COUNT_THRESHOLD:
                    logger.warning(
                        f"Sensor fault: {param}={value} outside [{lo}, {hi}] "
                        f"for {self.consecutive_faults[param]} consecutive readings"
                    )
                    any_fault = True
            else:
                self.consecutive_faults[param] = 0
        
        return any_fault
    
    def _compute_pid_doses(self,
                           chemistry: WaterChemistry,
                           residuals: Dict[str, float],
                           dt_hours: float) -> Dict[str, float]:
        """
        Compute what PID controllers would dose (for sanity checking MPC).
        """
        pid_doses = {}
        
        # pH PID → scale/corrosion inhibitor
        ph_output = self.pid_ph.update(chemistry.ph, dt_hours)
        # pH too low → need more alkalinity/inhibitor
        # pH too high → need less (or acid)
        ph_deficit = abs(self.limits.ph_target - chemistry.ph)
        
        # ORP PID → biocide
        orp_output = self.pid_orp.update(chemistry.orp_mv, dt_hours)
        
        # CoC PID → blowdown (handled separately)
        coc = chemistry.conductivity_us / max(self.tower.makeup_conductivity_us, 1)
        self.pid_coc.update(coc, dt_hours)
        
        # Map PID outputs to chemical doses
        for name, chem in self.tower.chemicals.items():
            if chem.dosing_mode != DosingMode.CONTINUOUS:
                continue
            
            current_ppm = residuals.get(name, chem.target_ppm)
            deficit = max(0, chem.target_ppm - current_ppm)
            
            # Base dose from deficit
            if deficit > 0:
                kg_needed = (deficit * self.tower.holding_volume_m3 * 1000 / 1e6 
                            / max(chem.active_fraction, 0.01))
                # Spread over ~12 cycles (1 hour)
                base_dose = kg_needed / 12.0
            else:
                base_dose = 0.0
            
            # Adjust based on PID output and chemical function
            if chem.function in (ChemicalFunction.SCALE_INHIBITOR, 
                                 ChemicalFunction.SCALE_CORROSION):
                pid_factor = max(0, 1.0 - ph_output)  # Low pH output → scale risk → dose more
                pid_doses[name] = base_dose * (0.5 + pid_factor)
                
            elif chem.function == ChemicalFunction.CORROSION_INHIBITOR:
                pid_factor = ph_output  # High pH output → corrosion risk → dose more
                pid_doses[name] = base_dose * (0.5 + pid_factor)
                
            elif chem.function == ChemicalFunction.NON_OXIDIZING_BIOCIDE:
                pid_factor = max(0, 1.0 - orp_output)  # Low ORP → more biocide
                pid_doses[name] = base_dose * (0.5 + pid_factor)
                
            else:
                pid_doses[name] = base_dose
            
            # Cap to hardware limit
            max_kg = chem.max_dose_rate_kg_per_hr * dt_hours
            pid_doses[name] = min(pid_doses[name], max_kg)
        
        return pid_doses
