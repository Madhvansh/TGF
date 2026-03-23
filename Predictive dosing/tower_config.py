"""
TGF Cooling Tower Configuration
================================
Tower specifications and chemical programs from _All_CT_working_2025_(Revised).xlsx
"""
from dataclasses import dataclass, field
from typing import Dict, Optional
from enum import Enum


class DosingMode(Enum):
    CONTINUOUS = "continuous"      # Daily proportional dosing
    SLUG = "slug"                 # Periodic shock dosing
    ALTERNATING = "alternating"   # Alternating between products


class ChemicalFunction(Enum):
    SCALE_INHIBITOR = "scale_inhibitor"
    CORROSION_INHIBITOR = "corrosion_inhibitor"
    SCALE_CORROSION = "scale_corrosion"          # Combined
    OXIDIZING_BIOCIDE = "oxidizing_biocide"
    NON_OXIDIZING_BIOCIDE = "non_oxidizing_biocide"
    DISPERSANT = "dispersant"
    PH_ADJUSTER = "ph_adjuster"


@dataclass
class ChemicalProduct:
    """Definition of a treatment chemical product."""
    name: str
    function: ChemicalFunction
    dosing_mode: DosingMode
    target_ppm: float                    # Target concentration in system
    min_ppm: float                       # Minimum safe concentration
    max_ppm: float                       # Maximum safe concentration (overdose limit)
    active_fraction: float               # Active ingredient fraction (0-1)
    density_kg_per_liter: float          # Product density
    half_life_hours: float               # Chemical degradation half-life at 30°C
    activation_energy_kj: float          # For temperature adjustment of decay rate
    cost_per_kg: float                   # INR per kg of product
    slug_quantity_kg: Optional[float] = None    # For slug dosing: kg per dose
    slug_interval_days: Optional[float] = None  # For slug dosing: days between doses
    sensor_proxy: Optional[str] = None          # Which sensor tracks this (e.g., 'ORP')
    max_dose_rate_kg_per_hr: float = 5.0        # Hardware pump limit


@dataclass
class TowerConfig:
    """Cooling tower physical specifications."""
    name: str
    holding_volume_m3: float           # System water volume
    circulation_rate_m3_per_hr: float  # Recirculation rate
    temperature_delta_c: float         # Delta T across tower
    design_coc: float                  # Design cycles of concentration
    drift_fraction: float              # Drift loss as fraction of circulation
    makeup_conductivity_us: float      # Makeup water conductivity (µS/cm)
    makeup_hardness_ppm: float         # Makeup water total hardness (ppm as CaCO3)
    makeup_calcium_ppm: float          # Makeup water calcium hardness (ppm as CaCO3)
    makeup_alkalinity_ppm: float       # Makeup water total alkalinity (ppm as CaCO3)
    makeup_tds_ppm: float              # Makeup water TDS
    makeup_ph: float                   # Makeup water pH
    chemicals: Dict[str, ChemicalProduct] = field(default_factory=dict)


# ============================================================================
# AQUATECH CHEMICAL PROGRAM - 850 TPD Tower
# ============================================================================

AQUATECH_850_TPD = TowerConfig(
    name="850_TPD_Main",
    holding_volume_m3=3000.0,
    circulation_rate_m3_per_hr=14000.0,
    temperature_delta_c=5.0,
    design_coc=6.0,
    drift_fraction=0.00002,          # 0.002% of circulation
    makeup_conductivity_us=400.0,     # Typical Indian surface water
    makeup_hardness_ppm=120.0,
    makeup_calcium_ppm=80.0,
    makeup_alkalinity_ppm=90.0,
    makeup_tds_ppm=300.0,
    makeup_ph=7.4,
    chemicals={
        "AQUATREAT-2501": ChemicalProduct(
            name="AQUATREAT-2501",
            function=ChemicalFunction.SCALE_CORROSION,
            dosing_mode=DosingMode.CONTINUOUS,
            target_ppm=12.0,
            min_ppm=8.0,
            max_ppm=18.0,
            active_fraction=0.30,            # ~30% active (phosphonate-based)
            density_kg_per_liter=1.15,
            half_life_hours=720.0,           # ~30 days (very stable)
            activation_energy_kj=50.0,
            cost_per_kg=180.0,
            sensor_proxy=None,               # No direct sensor
            max_dose_rate_kg_per_hr=10.0,    # Sized for 3000m³ at high blowdown     # Realistic for 3000 m³ system
        ),
        "AQUATREAT-1196": ChemicalProduct(
            name="AQUATREAT-1196",
            function=ChemicalFunction.SCALE_INHIBITOR,
            dosing_mode=DosingMode.CONTINUOUS,
            target_ppm=35.0,
            min_ppm=25.0,
            max_ppm=50.0,
            active_fraction=0.25,
            density_kg_per_liter=1.20,
            half_life_hours=600.0,           # ~25 days
            activation_energy_kj=45.0,
            cost_per_kg=220.0,
            sensor_proxy=None,
            max_dose_rate_kg_per_hr=25.0,   # Sized for 3000m³ at high blowdown
        ),
        "AQUATREAT-2150": ChemicalProduct(
            name="AQUATREAT-2150",
            function=ChemicalFunction.CORROSION_INHIBITOR,
            dosing_mode=DosingMode.CONTINUOUS,
            target_ppm=5.0,
            min_ppm=3.0,
            max_ppm=8.0,
            active_fraction=0.35,
            density_kg_per_liter=1.10,
            half_life_hours=336.0,           # ~14 days
            activation_energy_kj=55.0,
            cost_per_kg=250.0,
            sensor_proxy=None,
            max_dose_rate_kg_per_hr=3.0,
        ),
        "AQUATREAT-3331": ChemicalProduct(
            name="AQUATREAT-3331",
            function=ChemicalFunction.OXIDIZING_BIOCIDE,
            dosing_mode=DosingMode.SLUG,
            target_ppm=35.0,                 # Slug target (peak)
            min_ppm=0.5,                     # Minimum residual between slugs
            max_ppm=50.0,
            active_fraction=0.15,            # ~15% active chlorine
            density_kg_per_liter=1.25,
            half_life_hours=4.0,             # Degrades FAST (2-6 hrs)
            activation_energy_kj=80.0,       # Very temperature sensitive
            cost_per_kg=85.0,
            slug_quantity_kg=120.0,
            slug_interval_days=15.0,
            sensor_proxy="ORP",              # ORP tracks this directly
            max_dose_rate_kg_per_hr=120.0,   # Slug: all at once
        ),
        "AQUATREAT-399": ChemicalProduct(
            name="AQUATREAT-399",
            function=ChemicalFunction.OXIDIZING_BIOCIDE,
            dosing_mode=DosingMode.ALTERNATING,  # Alternates with 3331
            target_ppm=30.0,
            min_ppm=0.5,
            max_ppm=45.0,
            active_fraction=0.12,
            density_kg_per_liter=1.20,
            half_life_hours=5.0,
            activation_energy_kj=75.0,
            cost_per_kg=95.0,
            slug_quantity_kg=120.0,
            slug_interval_days=15.0,         # Alternates: 3331 → 399 → 3331
            sensor_proxy="ORP",
            max_dose_rate_kg_per_hr=120.0,
        ),
        "AQUATREAT-4612": ChemicalProduct(
            name="AQUATREAT-4612",
            function=ChemicalFunction.NON_OXIDIZING_BIOCIDE,
            dosing_mode=DosingMode.CONTINUOUS,
            target_ppm=4.0,
            min_ppm=2.0,
            max_ppm=8.0,
            active_fraction=0.20,
            density_kg_per_liter=1.10,
            half_life_hours=48.0,            # ~2 days
            activation_energy_kj=60.0,
            cost_per_kg=320.0,
            sensor_proxy=None,
            max_dose_rate_kg_per_hr=3.0,
        ),
        "AQUATREAT-6625": ChemicalProduct(
            name="AQUATREAT-6625",
            function=ChemicalFunction.DISPERSANT,
            dosing_mode=DosingMode.CONTINUOUS,
            target_ppm=4.0,
            min_ppm=2.0,
            max_ppm=8.0,
            active_fraction=0.25,
            density_kg_per_liter=1.08,
            half_life_hours=480.0,           # ~20 days
            activation_energy_kj=40.0,
            cost_per_kg=200.0,
            sensor_proxy=None,
            max_dose_rate_kg_per_hr=3.0,
        ),
    }
)


# ============================================================================
# AQUATECH CHEMICAL PROGRAM - 650 TPD Tower
# ============================================================================

AQUATECH_650_TPD = TowerConfig(
    name="650_TPD",
    holding_volume_m3=2000.0,
    circulation_rate_m3_per_hr=6000.0,
    temperature_delta_c=4.0,
    design_coc=6.0,
    drift_fraction=0.00002,
    makeup_conductivity_us=400.0,
    makeup_hardness_ppm=120.0,
    makeup_calcium_ppm=80.0,
    makeup_alkalinity_ppm=90.0,
    makeup_tds_ppm=300.0,
    makeup_ph=7.4,
    chemicals={
        "AQUATREAT-2501": ChemicalProduct(
            name="AQUATREAT-2501",
            function=ChemicalFunction.SCALE_CORROSION,
            dosing_mode=DosingMode.CONTINUOUS,
            target_ppm=12.0, min_ppm=8.0, max_ppm=18.0,
            active_fraction=0.30, density_kg_per_liter=1.15,
            half_life_hours=720.0, activation_energy_kj=50.0,
            cost_per_kg=180.0, max_dose_rate_kg_per_hr=2.0,
        ),
        "AQUATREAT-1196": ChemicalProduct(
            name="AQUATREAT-1196",
            function=ChemicalFunction.SCALE_INHIBITOR,
            dosing_mode=DosingMode.CONTINUOUS,
            target_ppm=60.0, min_ppm=40.0, max_ppm=80.0,
            active_fraction=0.25, density_kg_per_liter=1.20,
            half_life_hours=600.0, activation_energy_kj=45.0,
            cost_per_kg=220.0, max_dose_rate_kg_per_hr=5.0,
        ),
        "AQUATREAT-2150": ChemicalProduct(
            name="AQUATREAT-2150",
            function=ChemicalFunction.CORROSION_INHIBITOR,
            dosing_mode=DosingMode.CONTINUOUS,
            target_ppm=12.0, min_ppm=8.0, max_ppm=18.0,
            active_fraction=0.35, density_kg_per_liter=1.10,
            half_life_hours=336.0, activation_energy_kj=55.0,
            cost_per_kg=250.0, max_dose_rate_kg_per_hr=1.0,
        ),
        "AQUATREAT-4612": ChemicalProduct(
            name="AQUATREAT-4612",
            function=ChemicalFunction.NON_OXIDIZING_BIOCIDE,
            dosing_mode=DosingMode.CONTINUOUS,
            target_ppm=6.0, min_ppm=3.0, max_ppm=10.0,
            active_fraction=0.20, density_kg_per_liter=1.10,
            half_life_hours=48.0, activation_energy_kj=60.0,
            cost_per_kg=320.0, max_dose_rate_kg_per_hr=1.0,
        ),
        "AQUATREAT-3331": ChemicalProduct(
            name="AQUATREAT-3331",
            function=ChemicalFunction.OXIDIZING_BIOCIDE,
            dosing_mode=DosingMode.SLUG,
            target_ppm=35.0, min_ppm=0.5, max_ppm=50.0,
            active_fraction=0.15, density_kg_per_liter=1.25,
            half_life_hours=4.0, activation_energy_kj=80.0,
            cost_per_kg=85.0,
            slug_quantity_kg=100.0, slug_interval_days=15.0,
            sensor_proxy="ORP", max_dose_rate_kg_per_hr=100.0,
        ),
    }
)


# ============================================================================
# SAFE OPERATING LIMITS
# ============================================================================

@dataclass
class OperatingLimits:
    """Safe operating ranges for the cooling tower."""
    ph_min: float = 7.0
    ph_max: float = 9.0
    ph_target: float = 8.0
    
    conductivity_max_us: float = 5000.0
    
    coc_min: float = 3.0
    coc_max: float = 8.0
    coc_target: float = 6.0
    
    orp_min_mv: float = 350.0       # Below this: severe biofouling risk
    orp_target_mv: float = 650.0
    orp_max_mv: float = 800.0       # Above this: corrosion from excess oxidizer
    
    lsi_min: float = -1.0           # Below: corrosive
    lsi_max: float = 1.5            # Above: scaling
    lsi_target: float = 0.5         # Slight scaling tendency (protective)
    
    rsi_min: float = 5.0            # Below: heavy scaling
    rsi_max: float = 8.0            # Above: corrosive
    rsi_target: float = 6.5
    
    temperature_max_c: float = 45.0
    tds_max_ppm: float = 5000.0
    
    max_blowdown_rate_m3_per_hr: float = 30.0  # Hardware limit


DEFAULT_LIMITS = OperatingLimits()
