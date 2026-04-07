"""
TGF Physics Engine
===================
Langelier Saturation Index (LSI), Ryznar Stability Index (RSI),
and risk assessment from water chemistry parameters.

For MVP: Hardness and Alkalinity are estimated from CoC × makeup water quality
(physics-based), NOT from ML virtual sensors (which showed R²=0.37 on our data).
This is periodically calibrated with lab results.
"""
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


@dataclass
class WaterChemistry:
    """Complete water chemistry state at a point in time."""
    ph: float
    conductivity_us: float
    temperature_c: float
    orp_mv: float
    tds_ppm: Optional[float] = None
    calcium_hardness_ppm: Optional[float] = None      # as CaCO3
    total_hardness_ppm: Optional[float] = None         # as CaCO3
    total_alkalinity_ppm: Optional[float] = None       # as CaCO3
    chlorides_ppm: Optional[float] = None
    silica_ppm: Optional[float] = None
    iron_ppm: Optional[float] = None
    phosphate_ppm: Optional[float] = None
    turbidity_ntu: Optional[float] = None
    free_chlorine_ppm: Optional[float] = None
    timestamp: Optional[float] = None                   # Unix timestamp


@dataclass
class RiskAssessment:
    """Comprehensive risk scores for the cooling tower."""
    lsi: float                      # Langelier Saturation Index
    rsi: float                      # Ryznar Stability Index
    scaling_risk: float             # 0-1 score
    corrosion_risk: float           # 0-1 score
    biofouling_risk: float          # 0-1 score
    cascade_risk: float             # 0-1 composite
    overall_risk: float             # 0-1 weighted composite
    risk_level: str                 # "LOW", "MODERATE", "HIGH", "CRITICAL"
    details: Dict[str, str]         # Human-readable explanations


class PhysicsEngine:
    """
    Calculates water chemistry indices and risk assessments.
    
    LSI Calculation Strategy for MVP (No Hardness/Alkalinity Sensors):
    ================================================================
    1. PRIMARY: CoC-based estimation
       - Ca_circulating ≈ Ca_makeup × CoC
       - Alk_circulating ≈ Alk_makeup × CoC × correction_factor
       - CoC = conductivity_circulating / conductivity_makeup
       
    2. CALIBRATION: Weekly lab results update correction factors
       - If lab says Ca = 450 but CoC model says 480 → correction = 0.9375
       - Correction factors drift slowly, so weekly updates are sufficient
       
    3. WHY NOT ML VIRTUAL SENSORS:
       - RF/GB on Parameters_5K gave R²=0.37 for hardness, -0.30 for alkalinity
       - The paper's R²=0.93 was on surface water bodies, NOT cooling towers
       - Cooling tower water has treatment chemicals that confound correlations
       - CoC-based physics is more reliable than poor ML predictions
    """
    
    def __init__(self, tower_config):
        self.tower = tower_config
        
        # Calibration correction factors (updated by lab results)
        self.calcium_correction = 1.0      # Multiplied with CoC estimate
        self.alkalinity_correction = 0.85   # Alkalinity doesn't concentrate linearly
                                            # (CO2 exchange with atmosphere reduces it)
        self.hardness_correction = 1.0
        
        # Last lab calibration values
        self.last_lab_calcium = None
        self.last_lab_alkalinity = None
        self.last_lab_hardness = None
        self.last_calibration_timestamp = None
    
    # ========================================================================
    # ESTIMATED PARAMETERS (from CoC × makeup)
    # ========================================================================
    
    def estimate_coc(self, conductivity_circulating: float) -> float:
        """Calculate Cycles of Concentration from conductivity ratio."""
        if self.tower.makeup_conductivity_us <= 0:
            return 1.0
        coc = conductivity_circulating / self.tower.makeup_conductivity_us
        return max(1.0, min(coc, 15.0))  # Physical limits
    
    def estimate_tds(self, conductivity_us: float) -> float:
        """
        Estimate TDS from conductivity.
        TDS ≈ conductivity × conversion_factor
        Factor varies 0.5-0.7 depending on water composition.
        Using 0.65 as typical for cooling tower water.
        """
        tds = conductivity_us * 0.65
        return max(tds, 50.0)  # TDS can never be 0 in a cooling tower
    
    def estimate_calcium_hardness(self, coc: float, virtual_sensor=None,
                                    ph: float = None, conductivity: float = None,
                                    temperature: float = None, orp: float = None) -> float:
        """
        Estimate Ca hardness from CoC and makeup water quality.
        If a virtual_sensor is available and confident, uses ML prediction.
        """
        physics_value = self.tower.makeup_calcium_ppm * coc * self.calcium_correction
        if virtual_sensor and virtual_sensor.available and ph is not None:
            physics_th = self.tower.makeup_hardness_ppm * coc * self.hardness_correction
            physics_alk = self.tower.makeup_alkalinity_ppm * coc * self.alkalinity_correction
            preds, confidence = virtual_sensor.predict(
                ph, conductivity or 0, temperature or 32, orp or 400,
                coc, physics_th, physics_value, physics_alk)
            if confidence in ("GREEN", "AMBER"):
                return preds["calcium_hardness"]
        return physics_value

    def estimate_total_hardness(self, coc: float, virtual_sensor=None,
                                 ph: float = None, conductivity: float = None,
                                 temperature: float = None, orp: float = None) -> float:
        """Estimate total hardness from CoC and makeup water quality."""
        physics_value = self.tower.makeup_hardness_ppm * coc * self.hardness_correction
        if virtual_sensor and virtual_sensor.available and ph is not None:
            physics_ca = self.tower.makeup_calcium_ppm * coc * self.calcium_correction
            physics_alk = self.tower.makeup_alkalinity_ppm * coc * self.alkalinity_correction
            preds, confidence = virtual_sensor.predict(
                ph, conductivity or 0, temperature or 32, orp or 400,
                coc, physics_value, physics_ca, physics_alk)
            if confidence in ("GREEN", "AMBER"):
                return preds["total_hardness"]
        return physics_value

    def estimate_alkalinity(self, coc: float, virtual_sensor=None,
                             ph: float = None, conductivity: float = None,
                             temperature: float = None, orp: float = None) -> float:
        """
        Estimate alkalinity from CoC and makeup water quality.

        NOTE: Alkalinity does NOT concentrate linearly with CoC.
        CO2 exchange with atmosphere, acid dosing, and chemical
        treatments cause alkalinity to be 70-90% of theoretical.
        The correction factor handles this.
        """
        physics_value = self.tower.makeup_alkalinity_ppm * coc * self.alkalinity_correction
        if virtual_sensor and virtual_sensor.available and ph is not None:
            physics_th = self.tower.makeup_hardness_ppm * coc * self.hardness_correction
            physics_ca = self.tower.makeup_calcium_ppm * coc * self.calcium_correction
            preds, confidence = virtual_sensor.predict(
                ph, conductivity or 0, temperature or 32, orp or 400,
                coc, physics_th, physics_ca, physics_value)
            if confidence in ("GREEN", "AMBER"):
                return preds["total_alkalinity"]
        return physics_value
    
    def estimate_evaporation_rate(self, 
                                  temperature_c: float,
                                  circulation_rate: Optional[float] = None
                                  ) -> float:
        """
        Estimate evaporation rate in m³/hr.
        Evaporation ≈ Circulation × ΔT × Cp / Latent_Heat
        """
        circ = circulation_rate or self.tower.circulation_rate_m3_per_hr
        cp = 4.186  # kJ/(kg·°C)
        # Latent heat varies with temperature, ~2400 kJ/kg at 30°C
        latent_heat = 2501 - 2.361 * temperature_c  # kJ/kg
        evap = circ * 1000 * self.tower.temperature_delta_c * cp / (latent_heat * 1000)
        return evap  # m³/hr
    
    def estimate_blowdown_rate(self, coc: float, evaporation_rate: float) -> float:
        """
        Estimate blowdown rate from mass balance.
        Blowdown = Evaporation / (CoC - 1) - Drift
        """
        if coc <= 1.01:
            return evaporation_rate * 10  # If CoC ≈ 1, massive blowdown
        drift = self.tower.drift_fraction * self.tower.circulation_rate_m3_per_hr
        blowdown = evaporation_rate / (coc - 1) - drift
        return max(0.0, blowdown)
    
    def estimate_makeup_rate(self, evaporation_rate: float, 
                             blowdown_rate: float) -> float:
        """Makeup = Evaporation + Blowdown + Drift"""
        drift = self.tower.drift_fraction * self.tower.circulation_rate_m3_per_hr
        return evaporation_rate + blowdown_rate + drift
    
    # ========================================================================
    # WATER CHEMISTRY COMPLETION
    # ========================================================================
    
    def complete_chemistry(self, chemistry: WaterChemistry,
                           virtual_sensor=None) -> WaterChemistry:
        """
        Fill in missing parameters using physics-based estimation.
        Returns a new WaterChemistry with all fields populated.
        
        If a virtual_sensor is available (or stored on self), uses ML-corrected
        predictions for hardness/alkalinity instead of pure CoC scaling.
        """
        coc = self.estimate_coc(chemistry.conductivity_us)
        
        # Use stored virtual sensor if none passed explicitly
        vs = virtual_sensor or getattr(self, '_virtual_sensor', None)
        
        result = WaterChemistry(
            ph=chemistry.ph,
            conductivity_us=chemistry.conductivity_us,
            temperature_c=chemistry.temperature_c,
            orp_mv=chemistry.orp_mv,
            timestamp=chemistry.timestamp,
        )
        
        # TDS
        result.tds_ppm = chemistry.tds_ppm or self.estimate_tds(chemistry.conductivity_us)
        
        # Hardness (prefer measured → virtual sensor → physics estimation)
        result.calcium_hardness_ppm = (chemistry.calcium_hardness_ppm or 
                                       self.estimate_calcium_hardness(
                                           coc, virtual_sensor=vs,
                                           ph=chemistry.ph,
                                           conductivity=chemistry.conductivity_us,
                                           temperature=chemistry.temperature_c,
                                           orp=chemistry.orp_mv))
        result.total_hardness_ppm = (chemistry.total_hardness_ppm or 
                                     self.estimate_total_hardness(
                                         coc, virtual_sensor=vs,
                                         ph=chemistry.ph,
                                         conductivity=chemistry.conductivity_us,
                                         temperature=chemistry.temperature_c,
                                         orp=chemistry.orp_mv))
        
        # Alkalinity
        result.total_alkalinity_ppm = (chemistry.total_alkalinity_ppm or 
                                       self.estimate_alkalinity(
                                           coc, virtual_sensor=vs,
                                           ph=chemistry.ph,
                                           conductivity=chemistry.conductivity_us,
                                           temperature=chemistry.temperature_c,
                                           orp=chemistry.orp_mv))
        
        # Pass through measured values
        result.chlorides_ppm = chemistry.chlorides_ppm
        result.silica_ppm = chemistry.silica_ppm
        result.iron_ppm = chemistry.iron_ppm
        result.phosphate_ppm = chemistry.phosphate_ppm
        result.turbidity_ntu = chemistry.turbidity_ntu
        result.free_chlorine_ppm = chemistry.free_chlorine_ppm
        
        return result
    
    # ========================================================================
    # LSI / RSI CALCULATION
    # ========================================================================
    
    def calculate_phs(self, 
                      temperature_c: float,
                      tds_ppm: float,
                      calcium_hardness_ppm: float,
                      total_alkalinity_ppm: float) -> float:
        """
        Calculate pH of saturation (pHs) for LSI.
        
        pHs = (9.3 + A + B) - (C + D)
        
        where:
        A = (log10(TDS) - 1) / 10
        B = -13.12 × log10(T_kelvin) + 34.55
        C = log10(Ca as CaCO3) - 0.4
        D = log10(Alkalinity as CaCO3)
        """
        # Validate inputs
        if calcium_hardness_ppm <= 0 or total_alkalinity_ppm <= 0 or tds_ppm <= 0:
            logger.warning("Invalid inputs for pHs calculation: "
                         f"Ca={calcium_hardness_ppm}, Alk={total_alkalinity_ppm}, TDS={tds_ppm}")
            return 7.0  # Neutral fallback
        
        temp_k = temperature_c + 273.15
        
        # A: TDS factor
        A = (math.log10(max(tds_ppm, 1.0)) - 1.0) / 10.0
        
        # B: Temperature factor
        B = -13.12 * math.log10(temp_k) + 34.55
        
        # C: Calcium factor
        C = math.log10(max(calcium_hardness_ppm, 1.0)) - 0.4
        
        # D: Alkalinity factor
        D = math.log10(max(total_alkalinity_ppm, 1.0))
        
        phs = (9.3 + A + B) - (C + D)
        return phs
    
    def calculate_lsi(self, chemistry: WaterChemistry) -> float:
        """
        Langelier Saturation Index = pH - pHs
        
        LSI > 0: Scale-forming (CaCO3 precipitation tendency)
        LSI = 0: Balanced
        LSI < 0: Corrosive (CaCO3 dissolution tendency)
        
        Typical targets: 0.0 to +1.0 (slight scaling for metal protection)
        """
        completed = self.complete_chemistry(chemistry)
        
        phs = self.calculate_phs(
            completed.temperature_c,
            completed.tds_ppm,
            completed.calcium_hardness_ppm,
            completed.total_alkalinity_ppm
        )
        
        lsi = completed.ph - phs
        return max(-5.0, min(5.0, lsi))  # Physical bound: LSI beyond ±5 is nonsensical
    
    def calculate_rsi(self, chemistry: WaterChemistry) -> float:
        """
        Ryznar Stability Index = 2 × pHs - pH
        
        RSI < 6.0: Heavy scale formation
        RSI 6.0-7.0: Slight scale
        RSI 7.0-7.5: Slight corrosion
        RSI > 7.5: Significant corrosion
        
        Target: 6.0-7.0
        """
        completed = self.complete_chemistry(chemistry)
        
        phs = self.calculate_phs(
            completed.temperature_c,
            completed.tds_ppm,
            completed.calcium_hardness_ppm,
            completed.total_alkalinity_ppm
        )
        
        rsi = 2.0 * phs - completed.ph
        return max(2.0, min(14.0, rsi))  # Physical bound
    
    # ========================================================================
    # RISK ASSESSMENT
    # ========================================================================
    
    def assess_scaling_risk(self, lsi: float, rsi: float, 
                            temperature_c: float) -> Tuple[float, str]:
        """
        Scaling risk score (0-1) based on LSI, RSI, and temperature.
        Higher temperature increases crystallization kinetics.
        """
        # LSI component (primary)
        if lsi <= 0:
            lsi_risk = 0.0
        elif lsi <= 0.5:
            lsi_risk = lsi * 0.4       # 0 to 0.2
        elif lsi <= 1.0:
            lsi_risk = 0.2 + (lsi - 0.5) * 0.6   # 0.2 to 0.5
        elif lsi <= 2.0:
            lsi_risk = 0.5 + (lsi - 1.0) * 0.3   # 0.5 to 0.8
        else:
            lsi_risk = min(1.0, 0.8 + (lsi - 2.0) * 0.2)
        
        # Temperature amplifier (scale forms faster at higher T)
        temp_factor = 1.0 + max(0, (temperature_c - 35)) * 0.02
        
        risk = min(1.0, lsi_risk * temp_factor)
        
        if risk < 0.2:
            detail = f"Low scaling risk (LSI={lsi:.2f}, RSI={rsi:.2f})"
        elif risk < 0.5:
            detail = f"Moderate scaling tendency (LSI={lsi:.2f}). Monitor Ca and alkalinity"
        elif risk < 0.8:
            detail = f"HIGH scaling risk (LSI={lsi:.2f}). Scale inhibitor dosing critical"
        else:
            detail = f"CRITICAL scaling (LSI={lsi:.2f}). Immediate blowdown + inhibitor boost needed"
        
        return risk, detail
    
    def assess_corrosion_risk(self, lsi: float, ph: float, orp_mv: float,
                               conductivity_us: float) -> Tuple[float, str]:
        """
        Corrosion risk score (0-1).
        Risk increases with: negative LSI, low pH, high ORP, high conductivity.
        """
        risk = 0.0
        details = []
        
        # LSI component (negative LSI → corrosive)
        if lsi < -1.0:
            risk += 0.4
            details.append(f"Aggressive water (LSI={lsi:.2f})")
        elif lsi < 0:
            risk += abs(lsi) * 0.2
        
        # pH component
        if ph < 7.0:
            risk += (7.0 - ph) * 0.3
            details.append(f"Low pH ({ph:.1f}) accelerates corrosion")
        
        # High ORP can indicate excess oxidizer → corrosion
        if orp_mv > 750:
            risk += (orp_mv - 750) / 500
            details.append(f"High ORP ({orp_mv:.0f}mV) - excess oxidizer")
        
        # High conductivity → more conductive electrolyte → faster corrosion
        if conductivity_us > 3000:
            risk += (conductivity_us - 3000) / 10000
        
        risk = min(1.0, risk)
        detail = "; ".join(details) if details else f"Low corrosion risk (LSI={lsi:.2f}, pH={ph:.1f})"
        
        return risk, detail
    
    def assess_biofouling_risk(self, orp_mv: float, temperature_c: float,
                                ph: float) -> Tuple[float, str]:
        """
        Biofouling risk score (0-1).
        Risk increases with: low ORP, warm temperature, neutral pH.
        """
        risk = 0.0
        details = []
        
        # ORP component (primary indicator of biocide efficacy)
        if orp_mv < 350:
            risk += 0.6
            details.append(f"Very low ORP ({orp_mv:.0f}mV) - no biocide protection")
        elif orp_mv < 500:
            risk += 0.3 + (500 - orp_mv) / 500
            details.append(f"Low ORP ({orp_mv:.0f}mV) - biocide depleted")
        elif orp_mv < 600:
            risk += (600 - orp_mv) / 1000
        
        # Temperature component (microbes thrive at 25-40°C)
        if 25 <= temperature_c <= 40:
            temp_risk = 0.1 + 0.1 * (1.0 - abs(temperature_c - 32.5) / 7.5)
            risk += temp_risk
        
        # pH near neutral → optimal for most microbes
        if 6.5 <= ph <= 8.5:
            risk += 0.05
        
        risk = min(1.0, risk)
        detail = "; ".join(details) if details else "Adequate biocide protection"
        
        return risk, detail
    
    def assess_cascade_risk(self, scaling_risk: float, corrosion_risk: float,
                            biofouling_risk: float) -> Tuple[float, str]:
        """
        Cascade failure risk: when one problem triggers others.
        
        Classic cascade: biofilm → under-deposit corrosion → 
                        metal release → more scaling → more biofilm
        """
        # Cascade happens when MULTIPLE risks are elevated
        risks = [scaling_risk, corrosion_risk, biofouling_risk]
        elevated = sum(1 for r in risks if r > 0.3)
        
        if elevated >= 3:
            cascade = max(risks) * 1.3  # Amplified
            detail = "CRITICAL: Multiple elevated risks - cascade failure likely"
        elif elevated >= 2:
            cascade = max(risks) * 1.1
            detail = "WARNING: Two elevated risks - monitor for cascade development"
        else:
            cascade = max(risks) * 0.8
            detail = "Single or no elevated risks"
        
        return min(1.0, cascade), detail
    
    def full_risk_assessment(self, chemistry: WaterChemistry) -> RiskAssessment:
        """Complete risk assessment from current water chemistry."""
        completed = self.complete_chemistry(chemistry)
        
        lsi = self.calculate_lsi(chemistry)
        rsi = self.calculate_rsi(chemistry)
        
        scaling_risk, scaling_detail = self.assess_scaling_risk(
            lsi, rsi, completed.temperature_c)
        corrosion_risk, corrosion_detail = self.assess_corrosion_risk(
            lsi, completed.ph, completed.orp_mv, completed.conductivity_us)
        biofouling_risk, biofouling_detail = self.assess_biofouling_risk(
            completed.orp_mv, completed.temperature_c, completed.ph)
        cascade_risk, cascade_detail = self.assess_cascade_risk(
            scaling_risk, corrosion_risk, biofouling_risk)
        
        # Weighted overall risk
        overall = (scaling_risk * 0.35 + corrosion_risk * 0.25 + 
                   biofouling_risk * 0.25 + cascade_risk * 0.15)
        
        if overall < 0.2:
            level = "LOW"
        elif overall < 0.4:
            level = "MODERATE"
        elif overall < 0.7:
            level = "HIGH"
        else:
            level = "CRITICAL"
        
        return RiskAssessment(
            lsi=lsi, rsi=rsi,
            scaling_risk=scaling_risk,
            corrosion_risk=corrosion_risk,
            biofouling_risk=biofouling_risk,
            cascade_risk=cascade_risk,
            overall_risk=overall,
            risk_level=level,
            details={
                "scaling": scaling_detail,
                "corrosion": corrosion_detail,
                "biofouling": biofouling_detail,
                "cascade": cascade_detail,
                "lsi_interpretation": self._interpret_lsi(lsi),
                "rsi_interpretation": self._interpret_rsi(rsi),
            }
        )
    
    # ========================================================================
    # LAB CALIBRATION
    # ========================================================================
    
    def calibrate_from_lab(self,
                           lab_calcium: Optional[float] = None,
                           lab_alkalinity: Optional[float] = None,
                           lab_hardness: Optional[float] = None,
                           current_conductivity: float = 2000.0,
                           timestamp: Optional[float] = None):
        """
        Update correction factors from lab test results.
        Call this weekly when lab results come in.
        """
        coc = self.estimate_coc(current_conductivity)
        
        if lab_calcium is not None and lab_calcium > 0:
            theoretical_ca = self.tower.makeup_calcium_ppm * coc
            if theoretical_ca > 0:
                self.calcium_correction = lab_calcium / theoretical_ca
                self.last_lab_calcium = lab_calcium
                logger.info(f"Calcium correction updated: {self.calcium_correction:.3f} "
                          f"(lab={lab_calcium:.0f}, theoretical={theoretical_ca:.0f})")
        
        if lab_alkalinity is not None and lab_alkalinity > 0:
            theoretical_alk = self.tower.makeup_alkalinity_ppm * coc
            if theoretical_alk > 0:
                self.alkalinity_correction = lab_alkalinity / theoretical_alk
                self.last_lab_alkalinity = lab_alkalinity
                logger.info(f"Alkalinity correction updated: {self.alkalinity_correction:.3f} "
                          f"(lab={lab_alkalinity:.0f}, theoretical={theoretical_alk:.0f})")
        
        if lab_hardness is not None and lab_hardness > 0:
            theoretical_h = self.tower.makeup_hardness_ppm * coc
            if theoretical_h > 0:
                self.hardness_correction = lab_hardness / theoretical_h
                self.last_lab_hardness = lab_hardness
        
        self.last_calibration_timestamp = timestamp
    
    # ========================================================================
    # HELPERS
    # ========================================================================
    
    @staticmethod
    def _interpret_lsi(lsi: float) -> str:
        if lsi < -2.0:
            return "Severely corrosive"
        elif lsi < -0.5:
            return "Moderately corrosive"
        elif lsi < 0:
            return "Mildly corrosive"
        elif lsi < 0.5:
            return "Balanced to slightly scale-forming (IDEAL)"
        elif lsi < 1.5:
            return "Scale-forming"
        else:
            return "Severely scale-forming"
    
    @staticmethod
    def _interpret_rsi(rsi: float) -> str:
        if rsi < 5.5:
            return "Heavy scale"
        elif rsi < 6.2:
            return "Scale-forming"
        elif rsi < 6.8:
            return "Slight scale (IDEAL)"
        elif rsi < 7.5:
            return "Slight corrosion"
        elif rsi < 8.5:
            return "Significant corrosion"
        else:
            return "Severe corrosion"