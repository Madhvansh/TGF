"""
TGF Data Ingestion Pipeline
=============================
Simulates real-time sensor feeds from historical CSV data (Parameters_5K.csv).

For MVP: The dataset IS the sensor array. This module reads the CSV and streams
it as if 4 physical sensors (pH, Conductivity, Temperature, ORP) are producing
readings every 5 minutes.

Key capabilities:
1. Streaming mode: yields one reading at a time with configurable speed
2. Realistic sensor simulation: adds noise, dropouts, drift
3. Lab result extraction: pulls hardness/alkalinity as periodic lab data
4. Handles the messy reality of our dataset (missing values, inconsistent dates)
5. Provides a clean SensorReading interface to the rest of the system

Usage:
    ingestion = DataIngestionPipeline("Parameters_5K.csv")
    
    for reading in ingestion.stream(speed_multiplier=100):
        result = controller.run_cycle(
            ph=reading.ph, conductivity=reading.conductivity,
            temperature=reading.temperature, orp=reading.orp,
            timestamp=reading.timestamp,
            tds=reading.tds,
            calcium_hardness=reading.calcium_hardness,
            total_alkalinity=reading.total_alkalinity,
        )
"""
import pandas as pd
import numpy as np
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Generator, List, Dict, Tuple
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class IngestionMode(Enum):
    BATCH = "batch"              # Process all at once (for simulation)
    STREAMING = "streaming"      # Yield one at a time with timing
    REALTIME = "realtime"        # Match real 5-minute intervals


@dataclass
class SensorReading:
    """Single point-in-time sensor reading. This is the universal interface
    between data ingestion and the control system."""
    timestamp: float              # Unix timestamp
    cycle_index: int              # Sequential index
    
    # === PRIMARY SENSORS (always available) ===
    ph: float                     # pH sensor
    conductivity: float           # Conductivity sensor (µS/cm)
    temperature: float            # Temperature sensor (°C)
    orp: float                    # ORP sensor (mV)
    
    # === DERIVED / SECONDARY (may be None) ===
    tds: Optional[float] = None                # TDS (from sensor or estimated)
    
    # === LAB DATA (available periodically) ===
    calcium_hardness: Optional[float] = None   # From lab test
    total_hardness: Optional[float] = None     # From lab test
    total_alkalinity: Optional[float] = None   # From lab test
    magnesium_hardness: Optional[float] = None
    chlorides: Optional[float] = None
    phosphate: Optional[float] = None
    sulphates: Optional[float] = None
    silica: Optional[float] = None
    iron: Optional[float] = None
    turbidity: Optional[float] = None
    free_chlorine: Optional[float] = None
    suspended_solids: Optional[float] = None
    cycles_of_concentration: Optional[float] = None
    
    # === METADATA ===
    source_sheet: Optional[str] = None
    original_date: Optional[str] = None
    has_lab_data: bool = False     # Whether this reading includes lab results
    sensor_quality: str = "GOOD"  # "GOOD", "DEGRADED", "FAULT"


@dataclass 
class IngestionStats:
    """Statistics about the ingestion pipeline."""
    total_rows: int = 0
    rows_processed: int = 0
    rows_skipped: int = 0
    sensor_faults_injected: int = 0
    lab_readings_count: int = 0
    missing_ph: int = 0
    missing_conductivity: int = 0
    missing_temperature: int = 0
    missing_orp: int = 0
    date_range: str = ""
    source_sheets: List[str] = field(default_factory=list)


class DataIngestionPipeline:
    """
    Reads Parameters_5K.csv and streams it as realistic sensor data.
    
    The dataset has 5,614 rows with 18 columns. The 4 sensor columns
    (pH, TDS→Conductivity, Temperature, ORP) form our primary inputs.
    Everything else is "lab data" available periodically.
    
    Reality simulation features:
    - Sensor noise injection (±0.5% random noise on each reading)
    - Sensor dropout simulation (configurable % of readings marked as fault)
    - Temperature synthesis (dataset may lack temperature - seasonal model)
    - ORP synthesis (dataset may lack ORP - biocide decay model)
    - Lab data gating (hardness/alkalinity only available every N readings)
    """
    
    # Sensor noise levels (as fraction of reading)
    NOISE_LEVELS = {
        "pH": 0.005,            # ±0.04 pH units at pH 8
        "conductivity": 0.02,   # ±2% typical for industrial sensors
        "temperature": 0.01,    # ±0.3°C
        "orp": 0.03,            # ±20 mV typical
        "tds": 0.03,
    }
    
    # Valid physical ranges
    VALID_RANGES = {
        "pH": (4.0, 12.0),
        "conductivity": (50.0, 15000.0),
        "temperature": (5.0, 55.0),
        "orp": (100.0, 900.0),
        "tds": (50.0, 10000.0),
        "calcium_hardness": (10.0, 3000.0),
        "total_hardness": (10.0, 5000.0),
        "total_alkalinity": (10.0, 2000.0),
    }
    
    def __init__(self, 
                 csv_path: str,
                 cycle_interval_seconds: float = 300.0,  # 5 minutes
                 add_sensor_noise: bool = True,
                 sensor_dropout_rate: float = 0.0,  # 0-1, fraction of readings with faults
                 lab_interval_cycles: int = 2016,    # ~7 days at 5-min intervals
                 random_seed: int = 42):
        """
        Args:
            csv_path: Path to Parameters_5K.csv
            cycle_interval_seconds: Simulated time between readings (300s = 5 min)
            add_sensor_noise: Add realistic sensor noise
            sensor_dropout_rate: Fraction of readings with simulated sensor faults
            lab_interval_cycles: How often lab data is "available"
            random_seed: For reproducibility
        """
        self.csv_path = Path(csv_path)
        self.cycle_interval_seconds = cycle_interval_seconds
        self.add_noise = add_sensor_noise
        self.dropout_rate = sensor_dropout_rate
        self.lab_interval = lab_interval_cycles
        self.rng = np.random.RandomState(random_seed)
        
        self.stats = IngestionStats()
        self._df: Optional[pd.DataFrame] = None
        self._prepared: Optional[pd.DataFrame] = None
        
        # Load and prepare on init
        self._load_and_prepare()
    
    def _load_and_prepare(self):
        """Load CSV and prepare the simulation dataset."""
        logger.info(f"Loading data from {self.csv_path}")
        
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.csv_path}")
        
        self._df = pd.read_csv(self.csv_path)
        self.stats.total_rows = len(self._df)
        logger.info(f"  Loaded {self.stats.total_rows} rows, {len(self._df.columns)} columns")
        
        # Log column availability
        avail = {col: self._df[col].notna().sum() for col in self._df.columns}
        logger.info("  Column availability:")
        for col, count in sorted(avail.items(), key=lambda x: -x[1]):
            pct = count / self.stats.total_rows * 100
            logger.info(f"    {col}: {count} ({pct:.0f}%)")
        
        # Track source sheets
        if 'Source_Sheet' in self._df.columns:
            self.stats.source_sheets = self._df['Source_Sheet'].dropna().unique().tolist()
        
        # Prepare the simulation DataFrame
        self._prepared = self._prepare_simulation_data(self._df)
        logger.info(f"  Prepared {len(self._prepared)} samples for streaming")
    
    def _prepare_simulation_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw CSV into clean simulation-ready DataFrame.
        
        Strategy:
        - pH: forward-fill, then default 7.8
        - Conductivity: from column or estimated from TDS (TDS/0.65)
        - TDS: from column or estimated from Conductivity (Cond*0.65)
        - Temperature: synthesize seasonal+daily pattern (not in dataset)
        - ORP: synthesize biocide decay pattern (not in dataset)
        - Lab params: keep as-is (NaN where not measured)
        """
        sim = df.copy()
        n = len(sim)
        
        # ============================================================
        # TIMESTAMPS: Create realistic timeline
        # ============================================================
        if 'Date' in sim.columns and sim['Date'].notna().any():
            sim['Date_parsed'] = pd.to_datetime(sim['Date'], errors='coerce')
            valid_dates = sim['Date_parsed'].dropna()
            
            if len(valid_dates) > 0:
                date_min = valid_dates.min()
                date_max = valid_dates.max()
                self.stats.date_range = f"{date_min.date()} to {date_max.date()}"
                logger.info(f"  Date range in data: {self.stats.date_range}")
                
                # For rows with dates, use them. For gaps, interpolate.
                sim['timestamp'] = sim['Date_parsed'].apply(
                    lambda x: x.timestamp() if pd.notna(x) else np.nan)
        
        # If timestamps are still missing/empty, generate 5-min intervals
        if 'timestamp' not in sim.columns or sim['timestamp'].isna().all():
            start = datetime(2024, 1, 1).timestamp()
            sim['timestamp'] = [start + i * self.cycle_interval_seconds for i in range(n)]
            logger.info("  Generated synthetic timestamps (5-min intervals starting 2024-01-01)")
        else:
            # Fill gaps with interpolation
            sim['timestamp'] = sim['timestamp'].interpolate(method='linear')
            # Fill any remaining NaN at edges
            start = datetime(2024, 1, 1).timestamp()
            sim['timestamp'] = sim['timestamp'].fillna(
                pd.Series([start + i * self.cycle_interval_seconds for i in range(n)]))
        
        # ============================================================
        # PRIMARY SENSORS
        # ============================================================
        
        # pH: forward-fill gaps, then default
        sim['pH'] = sim['pH'].ffill().bfill().fillna(7.8)
        self.stats.missing_ph = df['pH'].isna().sum()
        
        # Conductivity & TDS: estimate from each other if missing
        has_cond = 'Conductivity_uS_cm' in sim.columns
        has_tds = 'TDS_ppm' in sim.columns
        
        if has_cond and has_tds:
            # Fill Conductivity from TDS where missing
            sim['Conductivity_uS_cm'] = sim['Conductivity_uS_cm'].fillna(
                sim['TDS_ppm'] / 0.65)
            # Fill TDS from Conductivity where missing
            sim['TDS_ppm'] = sim['TDS_ppm'].fillna(
                sim['Conductivity_uS_cm'] * 0.65)
        elif has_tds and not has_cond:
            sim['Conductivity_uS_cm'] = sim['TDS_ppm'] / 0.65
        elif has_cond and not has_tds:
            sim['TDS_ppm'] = sim['Conductivity_uS_cm'] * 0.65
        
        # Forward fill and default
        sim['Conductivity_uS_cm'] = sim['Conductivity_uS_cm'].ffill().bfill().fillna(2000.0)
        sim['TDS_ppm'] = sim['TDS_ppm'].ffill().bfill().fillna(1300.0)
        
        # Clip to physical minimums
        sim['Conductivity_uS_cm'] = sim['Conductivity_uS_cm'].clip(lower=50.0)
        sim['TDS_ppm'] = sim['TDS_ppm'].clip(lower=50.0)
        self.stats.missing_conductivity = (
            df.get('Conductivity_uS_cm', pd.Series(dtype=float)).isna().sum())
        
        # Temperature: synthesize if not in dataset
        if 'Temperature_C' not in sim.columns or sim.get('Temperature_C', pd.Series()).isna().all():
            sim['Temperature_C'] = self._synthesize_temperature(n)
            self.stats.missing_temperature = n
            logger.info("  Synthesized temperature data (Indian seasonal + diurnal cycle)")
        else:
            sim['Temperature_C'] = sim['Temperature_C'].ffill().bfill().fillna(32.0)
        
        # ORP: synthesize if not in dataset
        if 'ORP_mV' not in sim.columns or sim.get('ORP_mV', pd.Series()).isna().all():
            sim['ORP_mV'] = self._synthesize_orp(n)
            self.stats.missing_orp = n
            logger.info("  Synthesized ORP data (biocide dosing cycle pattern)")
        else:
            sim['ORP_mV'] = sim['ORP_mV'].ffill().bfill().fillna(650.0)
        
        # ============================================================
        # LAB PARAMETERS (keep as NaN where not measured)
        # ============================================================
        lab_cols = {
            'Calcium_Hardness_ppm': 'calcium_hardness',
            'Total_Hardness_ppm': 'total_hardness',
            'Total_Alkalinity_ppm': 'total_alkalinity',
            'Magnesium_Hardness_ppm': 'magnesium_hardness',
            'Chlorides_ppm': 'chlorides',
            'Phosphate_ppm': 'phosphate',
            'Sulphates_ppm': 'sulphates',
            'Silica_ppm': 'silica',
            'Iron_ppm': 'iron',
            'Turbidity_NTU': 'turbidity',
            'Free_Residual_Chlorine_ppm': 'free_chlorine',
            'Suspended_Solids_ppm': 'suspended_solids',
            'Cycles_of_Concentration': 'cycles_of_concentration',
        }
        
        for csv_col, internal_name in lab_cols.items():
            if csv_col in sim.columns:
                sim[internal_name] = sim[csv_col]
            else:
                sim[internal_name] = np.nan
        
        # Count lab readings
        lab_available = (sim['calcium_hardness'].notna() | 
                        sim['total_hardness'].notna() |
                        sim['total_alkalinity'].notna())
        self.stats.lab_readings_count = lab_available.sum()
        
        # Sort by timestamp
        sim = sim.sort_values('timestamp').reset_index(drop=True)
        
        return sim
    
    def _synthesize_temperature(self, n: int) -> np.ndarray:
        """
        Generate realistic cooling tower water temperature for Indian climate.
        
        Model:
        - Base: 32.5°C (annual average)
        - Seasonal swing: ±7.5°C (25°C winter Jan, 40°C peak May-Jun)
        - Diurnal swing: ±3°C (coolest 5am, hottest 3pm)
        - Random noise: σ=0.5°C (sensor + process noise)
        - Occasional process upsets: brief 2-5°C spikes
        """
        t = np.arange(n)
        
        # Seasonal: 365 days × 288 readings/day = 105,120 readings/year
        readings_per_day = 86400 / self.cycle_interval_seconds
        seasonal = 32.5 + 7.5 * np.sin(2 * np.pi * t / (365 * readings_per_day) - np.pi/3)
        
        # Diurnal: 288 readings/day cycle
        diurnal = 3.0 * np.sin(2 * np.pi * t / readings_per_day - np.pi/3)
        
        # Process noise
        noise = self.rng.normal(0, 0.5, n)
        
        # Occasional process upsets (1% chance per reading)
        upsets = np.zeros(n)
        upset_mask = self.rng.random(n) < 0.01
        upsets[upset_mask] = self.rng.uniform(2, 5, upset_mask.sum())
        
        temp = seasonal + diurnal + noise + upsets
        return np.clip(temp, 15.0, 50.0)
    
    def _synthesize_orp(self, n: int) -> np.ndarray:
        """
        Generate realistic ORP pattern for cooling tower.
        
        Model:
        - Biocide slug every ~15 days: sharp spike to 750+ mV, then exponential decay
        - Between slugs: gradual decline from ~650 to ~400 mV
        - Random noise: σ=20 mV
        - Correlation with temperature (higher temp → faster biocide decay)
        """
        readings_per_day = 86400 / self.cycle_interval_seconds
        slug_interval_readings = int(15 * readings_per_day)  # ~15 days
        
        orp = np.zeros(n)
        current_orp = 650.0
        last_slug = 0
        
        for i in range(n):
            # Check for slug dose
            if i - last_slug >= slug_interval_readings or (i == 0):
                current_orp = 750 + self.rng.normal(0, 15)
                last_slug = i
            else:
                # Exponential decay
                hours_since_slug = (i - last_slug) * self.cycle_interval_seconds / 3600
                decay_rate = 0.008  # per hour
                current_orp = 400 + (750 - 400) * np.exp(-decay_rate * hours_since_slug)
            
            # Add noise
            orp[i] = current_orp + self.rng.normal(0, 20)
        
        return np.clip(orp, 150, 850)
    
    def _add_sensor_noise(self, reading: SensorReading) -> SensorReading:
        """Add realistic sensor noise to a reading."""
        if not self.add_noise:
            return reading
        
        reading.ph += self.rng.normal(0, reading.ph * self.NOISE_LEVELS["pH"])
        reading.conductivity += self.rng.normal(
            0, reading.conductivity * self.NOISE_LEVELS["conductivity"])
        reading.temperature += self.rng.normal(
            0, max(1.0, reading.temperature) * self.NOISE_LEVELS["temperature"])
        reading.orp += self.rng.normal(0, max(10.0, reading.orp * self.NOISE_LEVELS["orp"]))
        
        if reading.tds is not None:
            reading.tds += self.rng.normal(0, reading.tds * self.NOISE_LEVELS["tds"])
        
        # Clip to physical ranges
        reading.ph = np.clip(reading.ph, *self.VALID_RANGES["pH"])
        reading.conductivity = np.clip(reading.conductivity, *self.VALID_RANGES["conductivity"])
        reading.temperature = np.clip(reading.temperature, *self.VALID_RANGES["temperature"])
        reading.orp = np.clip(reading.orp, *self.VALID_RANGES["orp"])
        
        return reading
    
    def _simulate_sensor_dropout(self, reading: SensorReading) -> SensorReading:
        """Simulate sensor faults/dropouts."""
        if self.dropout_rate <= 0:
            return reading
        
        if self.rng.random() < self.dropout_rate:
            # Pick a random sensor to fault
            fault_sensor = self.rng.choice(["pH", "conductivity", "temperature", "orp"])
            
            if fault_sensor == "pH":
                reading.ph = self.rng.choice([0.0, 14.0, np.nan])
            elif fault_sensor == "conductivity":
                reading.conductivity = self.rng.choice([0.0, 99999.0])
            elif fault_sensor == "temperature":
                reading.temperature = self.rng.choice([-10.0, 99.0])
            elif fault_sensor == "orp":
                reading.orp = self.rng.choice([-999.0, 9999.0])
            
            reading.sensor_quality = "FAULT"
            self.stats.sensor_faults_injected += 1
        
        return reading
    
    def _row_to_reading(self, row: pd.Series, index: int) -> SensorReading:
        """Convert a DataFrame row to a SensorReading."""
        
        # Determine if this is a "lab data" cycle
        has_lab = False
        ca_hard = None
        tot_hard = None
        tot_alk = None
        
        # Lab data availability: either from actual dataset values OR at regular intervals
        if pd.notna(row.get('calcium_hardness')):
            ca_hard = float(row['calcium_hardness'])
            has_lab = True
        if pd.notna(row.get('total_hardness')):
            tot_hard = float(row['total_hardness'])
            has_lab = True
        if pd.notna(row.get('total_alkalinity')):
            tot_alk = float(row['total_alkalinity'])
            has_lab = True
        
        reading = SensorReading(
            timestamp=float(row['timestamp']),
            cycle_index=index,
            
            # Primary sensors
            ph=float(row['pH']),
            conductivity=float(row['Conductivity_uS_cm']),
            temperature=float(row['Temperature_C']),
            orp=float(row['ORP_mV']),
            
            # Secondary
            tds=float(row['TDS_ppm']) if pd.notna(row.get('TDS_ppm')) else None,
            
            # Lab data
            calcium_hardness=ca_hard,
            total_hardness=tot_hard,
            total_alkalinity=tot_alk,
            magnesium_hardness=float(row['magnesium_hardness']) if pd.notna(row.get('magnesium_hardness')) else None,
            chlorides=float(row['chlorides']) if pd.notna(row.get('chlorides')) else None,
            phosphate=float(row['phosphate']) if pd.notna(row.get('phosphate')) else None,
            sulphates=float(row['sulphates']) if pd.notna(row.get('sulphates')) else None,
            silica=float(row['silica']) if pd.notna(row.get('silica')) else None,
            iron=float(row['iron']) if pd.notna(row.get('iron')) else None,
            turbidity=float(row['turbidity']) if pd.notna(row.get('turbidity')) else None,
            free_chlorine=float(row['free_chlorine']) if pd.notna(row.get('free_chlorine')) else None,
            suspended_solids=float(row['suspended_solids']) if pd.notna(row.get('suspended_solids')) else None,
            cycles_of_concentration=float(row['cycles_of_concentration']) if pd.notna(row.get('cycles_of_concentration')) else None,
            
            # Metadata
            source_sheet=str(row.get('Source_Sheet', '')) if pd.notna(row.get('Source_Sheet')) else None,
            original_date=str(row.get('Date', '')) if pd.notna(row.get('Date')) else None,
            has_lab_data=has_lab,
        )
        
        return reading
    
    # ========================================================================
    # PUBLIC STREAMING INTERFACE
    # ========================================================================
    
    def stream(self, 
               speed_multiplier: float = 0.0,
               max_readings: Optional[int] = None,
               start_index: int = 0
               ) -> Generator[SensorReading, None, None]:
        """
        Stream sensor readings one at a time.
        
        This is THE primary interface. The rest of the system calls this
        to get sensor data as if it were coming from real sensors.
        
        Args:
            speed_multiplier: How fast to simulate.
                0 = no delay (batch mode, as fast as possible)
                1 = real-time (5 min between readings)
                100 = 100x speed (3 seconds between readings)
                1000 = 1000x speed (0.3 seconds between readings)
            max_readings: Stop after this many readings (None = all)
            start_index: Start from this row index
        
        Yields:
            SensorReading objects, one per simulated 5-minute cycle
        """
        if self._prepared is None:
            raise RuntimeError("Data not prepared. Call _load_and_prepare() first.")
        
        n_total = len(self._prepared)
        end_index = min(n_total, start_index + max_readings) if max_readings else n_total
        
        delay = (self.cycle_interval_seconds / speed_multiplier 
                 if speed_multiplier > 0 else 0.0)
        
        logger.info(f"Streaming {end_index - start_index} readings "
                     f"(speed={speed_multiplier}x, delay={delay:.2f}s)")
        
        for i in range(start_index, end_index):
            row = self._prepared.iloc[i]
            reading = self._row_to_reading(row, i)
            
            # Add realism
            reading = self._add_sensor_noise(reading)
            reading = self._simulate_sensor_dropout(reading)
            
            self.stats.rows_processed += 1
            
            yield reading
            
            # Simulated delay
            if delay > 0:
                time.sleep(delay)
    
    def get_batch(self, 
                  max_readings: Optional[int] = None,
                  start_index: int = 0) -> List[SensorReading]:
        """Get all readings as a list (for batch processing)."""
        return list(self.stream(
            speed_multiplier=0, max_readings=max_readings, start_index=start_index))
    
    def get_reading_at(self, index: int) -> SensorReading:
        """Get a specific reading by index."""
        if self._prepared is None or index >= len(self._prepared):
            raise IndexError(f"Index {index} out of range (max {len(self._prepared) - 1})")
        row = self._prepared.iloc[index]
        return self._row_to_reading(row, index)
    
    @property
    def total_readings(self) -> int:
        """Total number of available readings."""
        return len(self._prepared) if self._prepared is not None else 0
    
    @property  
    def estimated_duration_days(self) -> float:
        """Estimated real-world duration the dataset covers."""
        return self.total_readings * self.cycle_interval_seconds / 86400
    
    def get_stats(self) -> dict:
        """Get ingestion statistics."""
        return {
            "total_rows": self.stats.total_rows,
            "rows_processed": self.stats.rows_processed,
            "rows_skipped": self.stats.rows_skipped,
            "sensor_faults_injected": self.stats.sensor_faults_injected,
            "lab_readings_available": self.stats.lab_readings_count,
            "missing_primary_sensors": {
                "pH": self.stats.missing_ph,
                "conductivity": self.stats.missing_conductivity,
                "temperature_synthesized": self.stats.missing_temperature,
                "orp_synthesized": self.stats.missing_orp,
            },
            "estimated_duration_days": round(self.estimated_duration_days, 1),
            "date_range": self.stats.date_range,
            "source_sheets": self.stats.source_sheets[:10],
        }
    
    def get_lab_calibration_data(self, reading: SensorReading) -> Optional[Dict[str, float]]:
        """
        Extract lab calibration data from a reading, if available.
        Returns dict suitable for DosingController.calibrate_from_lab()
        """
        if not reading.has_lab_data:
            return None
        
        lab = {}
        if reading.calcium_hardness is not None and reading.calcium_hardness > 0:
            lab['calcium_hardness'] = reading.calcium_hardness
        if reading.total_alkalinity is not None and reading.total_alkalinity > 0:
            lab['alkalinity'] = reading.total_alkalinity
        if reading.total_hardness is not None and reading.total_hardness > 0:
            lab['total_hardness'] = reading.total_hardness
        
        return lab if lab else None
