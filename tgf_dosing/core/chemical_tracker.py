"""
TGF Chemical Residual Tracker
===============================
Physics-based mass balance model tracking estimated concentration
of every treatment chemical in the cooling tower system.

This is THE answer to "how much chemical is already in the system?"

For each chemical at each timestep:
  C(t) = C(t-1) + dosed - blowdown_loss - drift_loss - degradation - consumption + concentration_effect

Calibrated by:
  - ORP sensor feedback for biocides (real-time)
  - Weekly lab testing for scale/corrosion inhibitors
  - Bayesian updating of decay constants
"""
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
from collections import deque
import logging
import time

from config.tower_config import (
    TowerConfig, ChemicalProduct, DosingMode, ChemicalFunction, OperatingLimits
)

logger = logging.getLogger(__name__)


@dataclass
class ChemicalState:
    """State of a single chemical in the system."""
    name: str
    estimated_ppm: float           # Current estimated concentration
    target_ppm: float              # Desired concentration
    deficit_ppm: float             # target - current (positive = underdosed)
    last_dose_timestamp: float     # When last dosed
    cumulative_24h_kg: float       # Total dosed in last 24 hours
    confidence: float              # 0-1, how confident we are in the estimate
    status: str                    # "ADEQUATE", "LOW", "CRITICAL", "OVERDOSED"


@dataclass
class TrackerSnapshot:
    """Complete state of all chemicals at a point in time."""
    timestamp: float
    chemicals: Dict[str, ChemicalState]
    coc: float
    evaporation_rate_m3_hr: float
    blowdown_rate_m3_hr: float
    makeup_rate_m3_hr: float


class ChemicalResidualTracker:
    """
    Tracks estimated chemical concentrations using mass balance physics.
    
    Usage:
        tracker = ChemicalResidualTracker(tower_config, physics_engine)
        
        # Every control cycle (5 minutes):
        snapshot = tracker.update(
            dt_hours=5/60,
            sensor_data=current_readings,
            pump_actions={"AQUATREAT-2501": 0.5},  # kg dosed this cycle
            physics_engine=engine
        )
        
        # Weekly lab calibration:
        tracker.calibrate_from_lab("AQUATREAT-2501", measured_ppm=11.5)
    """
    
    R_GAS = 8.314e-3  # kJ/(mol·K)
    
    def __init__(self, tower_config: TowerConfig):
        self.tower = tower_config
        
        # Current estimated concentrations (ppm) for each chemical
        self.residuals: Dict[str, float] = {}
        
        # Confidence in each estimate (0-1), decays over time without calibration
        self.confidence: Dict[str, float] = {}
        
        # Dosing history for rate tracking
        self.dose_history: Dict[str, deque] = {}  # deque of (timestamp, kg)
        
        # Calibration state
        self.decay_constants: Dict[str, float] = {}  # Learned k_degradation
        self.consumption_constants: Dict[str, float] = {}
        
        # Last slug dose info for biocides
        self.last_slug_time: Dict[str, float] = {}
        self.last_slug_ppm: Dict[str, float] = {}
        
        # Initialize all chemicals
        for name, chem in tower_config.chemicals.items():
            if chem.dosing_mode == DosingMode.SLUG or chem.dosing_mode == DosingMode.ALTERNATING:
                # Slug chemicals start at some decayed residual
                self.residuals[name] = chem.min_ppm
            else:
                # Continuous chemicals: assume at target initially
                self.residuals[name] = chem.target_ppm
            
            self.confidence[name] = 0.5  # Start uncertain
            self.dose_history[name] = deque(maxlen=2880)  # 10 days at 5-min intervals
            
            # Initial decay constants from product spec
            self.decay_constants[name] = math.log(2) / chem.half_life_hours
            # Consumption is typically MUCH smaller than degradation
            # Start conservative; lab calibration will adjust
            self.consumption_constants[name] = 0.002  # Very small default
    
    def update(self,
               dt_hours: float,
               coc: float,
               evaporation_rate: float,
               blowdown_rate: float,
               temperature_c: float,
               lsi: float,
               orp_mv: float,
               pump_actions: Dict[str, float],  # chemical_name → kg dosed this cycle
               current_timestamp: float = None
               ) -> TrackerSnapshot:
        """
        Update all chemical residual estimates for one control cycle.
        
        Args:
            dt_hours: Time step in hours (typically 5/60 = 0.0833)
            coc: Current cycles of concentration
            evaporation_rate: m³/hr
            blowdown_rate: m³/hr (calculated or measured)
            temperature_c: Current water temperature
            lsi: Current Langelier Saturation Index
            orp_mv: Current ORP reading
            pump_actions: Dict of chemical_name → kg of product dosed this cycle
            current_timestamp: Unix timestamp
        
        Returns:
            TrackerSnapshot with all chemical states
        """
        if current_timestamp is None:
            current_timestamp = time.time()
        
        volume = self.tower.holding_volume_m3
        drift_rate = self.tower.drift_fraction * self.tower.circulation_rate_m3_per_hr
        makeup_rate = evaporation_rate + blowdown_rate + drift_rate
        
        chemical_states = {}
        
        for name, chem in self.tower.chemicals.items():
            C_prev = self.residuals[name]
            
            # ====== ADDITIONS ======
            
            # 1. Chemical dosed this cycle
            kg_dosed = pump_actions.get(name, 0.0)
            if kg_dosed > 0:
                # Convert kg of product to ppm of active ingredient
                # ppm = (kg_active × 1e6) / (volume_liters)
                kg_active = kg_dosed * chem.active_fraction
                ppm_added = (kg_active * 1e6) / (volume * 1000)  # volume in m³ → liters
                
                # Record dosing event
                self.dose_history[name].append((current_timestamp, kg_dosed))
                
                # Update slug tracking
                if chem.dosing_mode in (DosingMode.SLUG, DosingMode.ALTERNATING):
                    if kg_dosed > 10:  # Slug threshold
                        self.last_slug_time[name] = current_timestamp
                        self.last_slug_ppm[name] = C_prev + ppm_added
            else:
                ppm_added = 0.0
            
            # NOTE: NO concentration effect for treatment chemicals.
            # Evaporation removes pure water but makeup (with 0 ppm treatment
            # chemical) immediately replaces it. The net effect on treatment
            # chemical concentration from evaporation+makeup is ZERO.
            # Concentration effect only applies to MINERALS in makeup water
            # (Ca, Mg, Alk, etc.) which is handled by CoC estimation separately.
            
            # ====== LOSSES ======
            
            # 3. Blowdown loss (chemical leaves with water)
            if volume > 0:
                blowdown_loss = C_prev * (blowdown_rate / volume) * dt_hours
            else:
                blowdown_loss = 0.0
            
            # 4. Drift loss
            if volume > 0:
                drift_loss = C_prev * (drift_rate / volume) * dt_hours
            else:
                drift_loss = 0.0
            
            # 5. Chemical degradation (temperature-dependent)
            k_deg = self._temperature_adjusted_decay(
                self.decay_constants[name],
                chem.activation_energy_kj,
                temperature_c
            )
            degradation = C_prev * (1.0 - math.exp(-k_deg * dt_hours))
            
            # 6. Consumption by reactions
            consumption = self._calculate_consumption(
                name, chem, C_prev, lsi, orp_mv, temperature_c, dt_hours
            )
            
            # ====== MASS BALANCE ======
            C_new = (C_prev 
                     + ppm_added 
                     - blowdown_loss 
                     - drift_loss 
                     - degradation 
                     - consumption)
            
            # Physical constraints
            C_new = max(0.0, C_new)
            
            # Update state
            self.residuals[name] = C_new
            
            # Confidence decays slowly without calibration
            self.confidence[name] = max(0.1, self.confidence[name] - 0.0001 * dt_hours)
            
            # Biocide calibration from ORP (real-time)
            if chem.sensor_proxy == "ORP" and chem.function == ChemicalFunction.OXIDIZING_BIOCIDE:
                self._calibrate_biocide_from_orp(name, chem, orp_mv, temperature_c)
            
            # Determine status
            deficit = chem.target_ppm - C_new
            if C_new > chem.max_ppm:
                status = "OVERDOSED"
            elif C_new < chem.min_ppm:
                status = "CRITICAL"
            elif C_new < chem.target_ppm * 0.8:
                status = "LOW"
            else:
                status = "ADEQUATE"
            
            # Calculate 24h cumulative dose
            cutoff_24h = current_timestamp - 86400
            cum_24h = sum(kg for ts, kg in self.dose_history[name] if ts > cutoff_24h)
            
            chemical_states[name] = ChemicalState(
                name=name,
                estimated_ppm=C_new,
                target_ppm=chem.target_ppm,
                deficit_ppm=deficit,
                last_dose_timestamp=max(
                    (ts for ts, _ in self.dose_history[name]), default=0
                ) if self.dose_history[name] else 0,
                cumulative_24h_kg=cum_24h,
                confidence=self.confidence[name],
                status=status,
            )
        
        return TrackerSnapshot(
            timestamp=current_timestamp,
            chemicals=chemical_states,
            coc=coc,
            evaporation_rate_m3_hr=evaporation_rate,
            blowdown_rate_m3_hr=blowdown_rate,
            makeup_rate_m3_hr=makeup_rate,
        )
    
    def _temperature_adjusted_decay(self, k_ref: float, ea_kj: float,
                                     temperature_c: float,
                                     t_ref_c: float = 30.0) -> float:
        """
        Arrhenius-adjusted decay rate constant.
        k(T) = k_ref × exp(Ea/R × (1/T_ref - 1/T))
        """
        t_k = temperature_c + 273.15
        t_ref_k = t_ref_c + 273.15
        
        if ea_kj <= 0 or t_k <= 0:
            return k_ref
        
        exponent = (ea_kj / self.R_GAS) * (1.0/t_ref_k - 1.0/t_k)
        # Clamp to prevent overflow
        exponent = max(-20.0, min(20.0, exponent))
        
        return k_ref * math.exp(exponent)
    
    def _calculate_consumption(self,
                                name: str,
                                chem: ChemicalProduct,
                                concentration: float,
                                lsi: float,
                                orp_mv: float,
                                temperature_c: float,
                                dt_hours: float) -> float:
        """
        Calculate chemical consumption by reactions.
        
        Scale inhibitors: consumed when chelating Ca²⁺ (proportional to scaling tendency)
        Corrosion inhibitors: consumed forming protective films (proportional to fresh metal)
        Biocides: consumed killing microorganisms (proportional to microbial demand)
        """
        k_cons = self.consumption_constants[name]
        
        if chem.function == ChemicalFunction.SCALE_INHIBITOR:
            # More consumed when LSI is positive (scaling conditions)
            scaling_factor = max(0.0, lsi) * 0.5
            consumption = concentration * k_cons * (1.0 + scaling_factor) * dt_hours
            
        elif chem.function == ChemicalFunction.CORROSION_INHIBITOR:
            # More consumed when corrosion is active (low pH, negative LSI)
            corrosion_factor = max(0.0, -lsi) * 0.3
            consumption = concentration * k_cons * (1.0 + corrosion_factor) * dt_hours
            
        elif chem.function == ChemicalFunction.SCALE_CORROSION:
            # Combined: both scaling and corrosion consume it
            combined_factor = abs(lsi) * 0.3
            consumption = concentration * k_cons * (1.0 + combined_factor) * dt_hours
            
        elif chem.function in (ChemicalFunction.OXIDIZING_BIOCIDE, 
                                ChemicalFunction.NON_OXIDIZING_BIOCIDE):
            # Biocide demand: higher when ORP is low (more microbes)
            bio_demand = max(0.0, (600 - orp_mv) / 600) * 2.0
            # Temperature effect: microbes grow faster in warm water
            temp_factor = max(0.5, 1.0 + (temperature_c - 25) * 0.03)
            consumption = concentration * k_cons * (1.0 + bio_demand) * temp_factor * dt_hours
            
        elif chem.function == ChemicalFunction.DISPERSANT:
            # Dispersant consumed by particle suspension
            consumption = concentration * k_cons * dt_hours
            
        else:
            consumption = concentration * k_cons * dt_hours
        
        return max(0.0, min(consumption, concentration * 0.5))  # Can't consume more than 50% per step
    
    def _calibrate_biocide_from_orp(self, name: str, chem: ChemicalProduct,
                                     orp_mv: float, temperature_c: float):
        """
        Use ORP reading to calibrate oxidizing biocide residual estimate.
        
        ORP vs Free Chlorine (approximate relationship at pH 7.5-8.5):
        ORP 650 mV → ~0.5-1.0 ppm free Cl
        ORP 700 mV → ~1.0-2.0 ppm free Cl
        ORP 750 mV → ~2.0-5.0 ppm free Cl
        ORP < 550 mV → essentially 0 free Cl
        
        NOTE: This is an approximation. ORP is affected by pH, temperature,
        and other oxidants/reductants. But it's the best real-time proxy.
        """
        if orp_mv < 400:
            orp_implied_residual = 0.0
        elif orp_mv < 550:
            orp_implied_residual = 0.0
        elif orp_mv < 650:
            orp_implied_residual = (orp_mv - 550) / 200  # 0 to 0.5
        elif orp_mv < 750:
            orp_implied_residual = 0.5 + (orp_mv - 650) / 100  # 0.5 to 1.5
        else:
            orp_implied_residual = 1.5 + (orp_mv - 750) / 200  # 1.5+
        
        # Scale to the product's active fraction
        # If product is 15% active chlorine, and we estimate 1.0 ppm free Cl,
        # then product residual ≈ 1.0 / 0.15 = 6.67 ppm (of the product)
        if chem.active_fraction > 0:
            implied_product_ppm = orp_implied_residual / chem.active_fraction
        else:
            implied_product_ppm = orp_implied_residual
        
        # Blend: 70% model, 30% ORP-implied (ORP is noisy but real-time)
        current = self.residuals[name]
        blended = 0.7 * current + 0.3 * implied_product_ppm
        self.residuals[name] = blended
        
        # Increase confidence since we have sensor feedback
        self.confidence[name] = min(0.9, self.confidence[name] + 0.01)
    
    def calibrate_from_lab(self, chemical_name: str, measured_ppm: float):
        """
        Update residual estimate from lab test result.
        Also adjusts consumption constants via Bayesian updating.
        """
        if chemical_name not in self.residuals:
            logger.warning(f"Unknown chemical: {chemical_name}")
            return
        
        estimated = self.residuals[chemical_name]
        error = measured_ppm - estimated
        
        logger.info(f"Lab calibration for {chemical_name}: "
                    f"estimated={estimated:.1f}, measured={measured_ppm:.1f}, "
                    f"error={error:+.1f}")
        
        # Update residual to lab value
        self.residuals[chemical_name] = measured_ppm
        
        # Reset confidence high
        self.confidence[chemical_name] = 0.95
        
        # Adjust consumption constant if error is consistent
        # If we consistently overestimate → increase consumption constant
        # If we consistently underestimate → decrease consumption constant
        if estimated > 0 and measured_ppm > 0:
            ratio = measured_ppm / estimated
            # Gentle adjustment: move consumption constant 10% toward correction
            k_old = self.consumption_constants[chemical_name]
            if ratio < 0.9:  # Overestimating: need more consumption
                k_new = k_old * 1.1
            elif ratio > 1.1:  # Underestimating: need less consumption
                k_new = k_old * 0.9
            else:
                k_new = k_old
            
            self.consumption_constants[chemical_name] = max(0.001, min(0.1, k_new))
    
    def get_residual(self, chemical_name: str) -> float:
        """Get current estimated residual for a chemical."""
        return self.residuals.get(chemical_name, 0.0)
    
    def get_deficit(self, chemical_name: str) -> float:
        """Get deficit (positive = needs more) for a chemical."""
        if chemical_name not in self.tower.chemicals:
            return 0.0
        target = self.tower.chemicals[chemical_name].target_ppm
        current = self.residuals.get(chemical_name, 0.0)
        return target - current
    
    def get_all_deficits(self) -> Dict[str, float]:
        """Get deficits for all chemicals."""
        return {name: self.get_deficit(name) for name in self.tower.chemicals}
    
    def should_slug_dose(self, chemical_name: str, 
                         current_timestamp: float) -> Tuple[bool, str]:
        """
        Determine if a slug biocide dose is needed.
        
        Returns (should_dose, reason)
        """
        if chemical_name not in self.tower.chemicals:
            return False, "Unknown chemical"
        
        chem = self.tower.chemicals[chemical_name]
        if chem.dosing_mode not in (DosingMode.SLUG, DosingMode.ALTERNATING):
            return False, "Not a slug-dosed chemical"
        
        last_slug = self.last_slug_time.get(chemical_name, 0)
        days_since_slug = (current_timestamp - last_slug) / 86400 if last_slug > 0 else 999
        
        # Check if interval has elapsed
        if chem.slug_interval_days and days_since_slug >= chem.slug_interval_days:
            return True, f"Scheduled: {days_since_slug:.1f} days since last slug (interval={chem.slug_interval_days}d)"
        
        # Check if residual has dropped critically
        if self.residuals[chemical_name] < chem.min_ppm and days_since_slug > 3:
            return True, f"Residual critical: {self.residuals[chemical_name]:.1f} ppm (min={chem.min_ppm})"
        
        return False, f"Next slug in {max(0, chem.slug_interval_days - days_since_slug):.1f} days"
    
    def summary(self) -> str:
        """Human-readable summary of all chemical states."""
        lines = ["═══ Chemical Residual Tracker ═══"]
        for name, chem in self.tower.chemicals.items():
            ppm = self.residuals[name]
            conf = self.confidence[name]
            target = chem.target_ppm
            
            if ppm > chem.max_ppm:
                indicator = "⚠️  OVERDOSED"
            elif ppm < chem.min_ppm:
                indicator = "🔴 CRITICAL"
            elif ppm < target * 0.8:
                indicator = "🟡 LOW"
            else:
                indicator = "🟢 OK"
            
            lines.append(
                f"  {name:25s} │ {ppm:6.1f} ppm │ target: {target:5.1f} │ "
                f"[{chem.min_ppm:.0f}-{chem.max_ppm:.0f}] │ conf: {conf:.0%} │ {indicator}"
            )
        return "\n".join(lines)
