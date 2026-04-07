"""
MOMENT Anomaly Detector for TGF Real-time Pipeline
====================================================
Extracted from moment_2026_46_latest.py (2,439 lines) for production inference.

Architecture: MOMENT-1-large (385M params, 24 layers, D=1024)
Paper: "MOMENT: A Family of Open Time-series Foundation Models" (ICML 2024)

This module provides a clean interface for the AnomalyDetector in
infrastructure/anomaly_detector.py to use MOMENT reconstruction-based
anomaly detection.
"""

import math
import logging
from collections import deque
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger("MOMENT-Detector")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class MomentInferenceConfig:
    seq_len: int = 512
    patch_len: int = 8
    num_patches: int = 64
    d_model: int = 1024
    num_heads: int = 16
    num_layers: int = 24
    d_ff: int = 4096
    dropout: float = 0.1
    pot_quantile: float = 0.95
    pot_window: int = 100
    device: str = "auto"

    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# WATER QUALITY DOMAIN
# ============================================================================

class WaterQualityDomain:
    PARAMETER_SPEC = {
        "pH":                         ("pH",                7.5,   8.0,   True),
        "Turbidity_NTU":              ("Turbidity",         0.0,   20.0,  False),
        "Free_Residual_Chlorine_ppm": ("FRC",               0.2,   0.5,   True),
        "TDS_ppm":                    ("TDS",               0.0,   2100.0,True),
        "Total_Hardness_ppm":         ("Total Hardness",    0.0,   1200.0,True),
        "Calcium_Hardness_ppm":       ("Calcium Hardness",  0.0,   800.0, True),
        "Magnesium_Hardness_ppm":     ("Magnesium Hardness",0.0,   400.0, False),
        "Chlorides_ppm":              ("Chlorides",         0.0,   500.0, True),
        "Phosphate_ppm":              ("Phosphate",         6.0,   8.0,   True),
        "Total_Alkalinity_ppm":       ("Total Alkalinity",  0.0,   200.0, True),
        "Sulphates_ppm":              ("Sulphates",         0.0,   1000.0,True),
        "Silica_ppm":                 ("Silica",            0.0,   180.0, True),
        "Iron_ppm":                   ("Iron",              0.0,   2.0,   True),
        "Suspended_Solids_ppm":       ("Suspended Solids",  0.0,   50.0,  False),
        "Conductivity_uS_cm":        ("Conductivity",      0.0,   3000.0,True),
    }

    # Sensor name mapping (AnomalyDetector uses short names)
    SENSOR_TO_COLUMN = {
        "ph": "pH",
        "conductivity": "Conductivity_uS_cm",
        "temperature": None,  # Not a water quality parameter
        "orp": None,          # Not a water quality parameter
        "tds": "TDS_ppm",
    }

    CASCADE_CHAINS = {
        "scaling": ["Calcium_Hardness_ppm", "Total_Alkalinity_ppm", "pH",
                     "Silica_ppm", "Total_Hardness_ppm"],
        "corrosion": ["pH", "Chlorides_ppm", "Iron_ppm", "Conductivity_uS_cm"],
        "biofouling": ["Free_Residual_Chlorine_ppm", "Phosphate_ppm", "Turbidity_NTU"],
    }

    @classmethod
    def check_parameter(cls, col_name: str, value: float) -> float:
        """Return severity score 0-1 for a parameter value."""
        if col_name not in cls.PARAMETER_SPEC:
            return 0.0
        _, pmin, pmax, is_critical = cls.PARAMETER_SPEC[col_name]
        if np.isnan(value):
            return 0.0
        if value < pmin:
            deviation = (pmin - value) / (pmin + 1e-8)
        elif value > pmax:
            deviation = (value - pmax) / (pmax + 1e-8)
        else:
            return 0.0
        score = min(deviation, 1.0)
        if is_critical:
            score = min(score * 1.5, 1.0)
        return score

    @classmethod
    def detect_cascade_risk(cls, anomalous_params: List[str]) -> List[Dict]:
        risks = []
        for chain_name, triggers in cls.CASCADE_CHAINS.items():
            triggered = [p for p in triggers if p in anomalous_params]
            if len(triggered) >= 2:
                risk_level = len(triggered) / len(triggers)
                risks.append({
                    "chain": chain_name,
                    "risk_score": round(risk_level, 3),
                    "triggered_params": triggered,
                })
        return sorted(risks, key=lambda x: x["risk_score"], reverse=True)


# ============================================================================
# MODEL ARCHITECTURE (inference-only, matches moment_2026_46_latest.py)
# ============================================================================

class RevIN(nn.Module):
    def __init__(self, num_features=1, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x, mask, mode="norm"):
        if mode == "norm":
            mask_sum = mask.sum(dim=-1, keepdim=True).clamp(min=1)
            self.mean = (x * mask).sum(dim=-1, keepdim=True) / mask_sum
            self.std = torch.sqrt(
                ((x - self.mean) ** 2 * mask).sum(dim=-1, keepdim=True) / mask_sum
            )
            x = (x - self.mean) / (self.std + self.eps)
            x = x * self.affine_weight + self.affine_bias
        else:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            x = x * (self.std + self.eps) + self.mean
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, patch_len, d_model):
        super().__init__()
        self.proj = nn.Linear(patch_len, d_model)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, patches, patch_mask):
        embeddings = self.proj(patches)
        mask_expanded = patch_mask.unsqueeze(-1)
        mask_tokens = self.mask_token.expand_as(embeddings)
        return embeddings * mask_expanded + mask_tokens * (1 - mask_expanded)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MOMENT(nn.Module):
    """MOMENT reconstruction model (channel-independent)."""
    def __init__(self, config: MomentInferenceConfig):
        super().__init__()
        self.config = config
        self.revin = RevIN(num_features=1)
        self.patch_embedding = PatchEmbedding(config.patch_len, config.d_model)
        self.pos_encoding = PositionalEncoding(
            config.d_model, max_len=config.num_patches + 1, dropout=config.dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model, nhead=config.num_heads,
            dim_feedforward=config.d_ff, dropout=config.dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers,
            enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(config.d_model, elementwise_affine=False)
        self.reconstruction_head = nn.Sequential(
            nn.Dropout(config.dropout), nn.Linear(config.d_model, config.patch_len))

    def forward(self, timeseries, input_mask):
        # timeseries: (B, 1, seq_len), input_mask: (B, 1, seq_len)
        x_norm = self.revin(timeseries, input_mask, "norm")
        patches = x_norm.squeeze(1).unfold(1, self.config.patch_len, self.config.patch_len)
        mask = input_mask.squeeze(1).unfold(1, self.config.patch_len, self.config.patch_len)
        patch_mask = (mask.sum(dim=-1) == self.config.patch_len).float()
        embeddings = self.patch_embedding(patches, patch_mask)
        embeddings = self.pos_encoding(embeddings)
        encoded = self.output_norm(self.encoder(embeddings))
        recon_patches = self.reconstruction_head(encoded)
        B, N, P = recon_patches.shape
        recon_norm = recon_patches.reshape(B, 1, N * P)
        reconstruction = self.revin(recon_norm, input_mask, "denorm")
        anomaly_scores = ((timeseries - reconstruction) ** 2).squeeze(1)
        return reconstruction, anomaly_scores


# ============================================================================
# POT THRESHOLD
# ============================================================================

class POTThreshold:
    def __init__(self, q=0.95, window_size=100):
        self.q = q
        self.window_size = window_size
        self.threshold = None
        self.recent_scores = []

    def fit(self, scores):
        flat = np.array(scores).flatten()
        flat = flat[~np.isnan(flat)]
        if len(flat) > 0:
            self.threshold = float(np.quantile(flat, self.q))
        return self.threshold

    def update(self, new_score):
        self.recent_scores.append(float(new_score))
        if len(self.recent_scores) > self.window_size:
            self.recent_scores = self.recent_scores[-self.window_size:]
        if len(self.recent_scores) > 10 and self.threshold is not None:
            recent = np.array(self.recent_scores)
            median = float(np.median(recent))
            mad = float(np.mean(np.abs(recent - median)))
            new_threshold = median + 2.5 * mad
            self.threshold = 0.95 * self.threshold + 0.05 * new_threshold

    def is_anomalous(self, score):
        if self.threshold is None:
            return False
        self.update(score)
        return score > self.threshold


# ============================================================================
# MAIN WRAPPER: MomentAnomalyDetector
# ============================================================================

class MomentAnomalyDetector:
    """
    Production wrapper for MOMENT anomaly detection.

    Designed to integrate with AnomalyDetector in infrastructure/anomaly_detector.py.
    Accumulates sensor readings, runs MOMENT inference when enough data is available,
    and returns anomaly scores.

    Usage:
        detector = MomentAnomalyDetector("checkpoints/moment_tgf_model.pt")
        # Each cycle:
        score, channel_scores = detector.anomaly_score(history_dict)
    """

    # Parameters we can use from AnomalyDetector's history
    MONITORED_PARAMS = ["ph", "conductivity", "temperature", "orp"]

    def __init__(self, checkpoint_path: str = None, device: str = "auto",
                 config: MomentInferenceConfig = None):
        self.config = config or MomentInferenceConfig(device=device)
        self.device = self.config.device
        self.model = None
        self.scaler = RobustScaler()
        self.global_threshold = POTThreshold(self.config.pot_quantile,
                                              self.config.pot_window)
        self.channel_thresholds = {}
        self._fitted = False
        self._buffer = deque(maxlen=self.config.seq_len)
        self._score_history = deque(maxlen=500)

        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        else:
            self._init_fresh_model()

    def _init_fresh_model(self):
        """Initialize a fresh model (for when no checkpoint is available)."""
        # Use smaller config for fresh model (not pretrained)
        small_config = MomentInferenceConfig(
            seq_len=64, patch_len=8, num_patches=8,
            d_model=256, num_heads=8, num_layers=4,
            d_ff=512, dropout=0.1, device=self.device
        )
        self.config = small_config
        self._buffer = deque(maxlen=small_config.seq_len)
        self.model = MOMENT(small_config).to(self.device)
        self.model.eval()
        logger.info("Initialized fresh MOMENT model (no pretrained weights)")

    def _load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

                # Try to infer config from state dict
                if 'config' in checkpoint:
                    cfg = checkpoint['config']
                    if hasattr(cfg, 'seq_len'):
                        self.config = MomentInferenceConfig(
                            seq_len=cfg.seq_len, patch_len=cfg.patch_len,
                            num_patches=cfg.num_patches, d_model=cfg.d_model,
                            num_heads=cfg.num_heads, num_layers=cfg.num_layers,
                            d_ff=cfg.d_ff, dropout=cfg.dropout, device=self.device)
                        self._buffer = deque(maxlen=cfg.seq_len)

                # Load scaler if available
                if 'scaler' in checkpoint:
                    self.scaler = checkpoint['scaler']
                    self._fitted = True

                # Load thresholds if available
                if 'threshold' in checkpoint:
                    self.global_threshold.threshold = checkpoint['threshold']
                if 'channel_thresholds' in checkpoint:
                    self.channel_thresholds = checkpoint['channel_thresholds']

                self.model = MOMENT(self.config).to(self.device)
                self.model.load_state_dict(state_dict, strict=False)
            else:
                # Direct model object
                self.model = checkpoint.to(self.device)

            self.model.eval()
            logger.info(f"Loaded MOMENT checkpoint from {path}")

        except Exception as e:
            logger.warning(f"Failed to load checkpoint {path}: {e}")
            logger.info("Falling back to fresh model")
            self._init_fresh_model()

    def is_ready(self) -> bool:
        """Whether enough data has accumulated for inference."""
        return len(self._buffer) >= self.config.seq_len

    def add_reading(self, values: Dict[str, float]):
        """Add a sensor reading to the internal buffer."""
        # Convert to feature vector using available params
        features = []
        for param in self.MONITORED_PARAMS:
            val = values.get(param, 0.0)
            if val is None or np.isnan(val):
                val = 0.0
            features.append(val)
        self._buffer.append(features)

    def anomaly_score(self, history: Dict[str, deque]) -> Tuple[float, Dict[str, float]]:
        """
        Compute anomaly score from rolling sensor history.

        Args:
            history: Dict mapping parameter names to deque of recent values
                     (maintained by AnomalyDetector)

        Returns:
            (global_score, {param: score}) -- scores in [0, 1]
        """
        if self.model is None:
            return 0.0, {}

        # Build time series from history
        params = [p for p in self.MONITORED_PARAMS if p in history]
        if not params:
            return 0.0, {}

        min_len = min(len(history[p]) for p in params)
        if min_len < self.config.seq_len:
            return 0.0, {}

        channel_scores = {}
        all_scores = []

        with torch.no_grad():
            for param in params:
                values = list(history[param])[-self.config.seq_len:]
                arr = np.array(values, dtype=np.float32)

                # Handle NaN/inf
                mask_arr = np.isfinite(arr).astype(np.float32)
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

                # Normalize
                mean = arr[mask_arr > 0].mean() if mask_arr.sum() > 0 else 0
                std = arr[mask_arr > 0].std() if mask_arr.sum() > 0 else 1
                if std < 1e-8:
                    std = 1.0

                ts = torch.FloatTensor(arr).unsqueeze(0).unsqueeze(0).to(self.device)
                mask = torch.FloatTensor(mask_arr).unsqueeze(0).unsqueeze(0).to(self.device)

                _, scores = self.model(ts, mask)
                score = float(scores.mean().cpu())

                # Normalize score to 0-1 range
                all_scores.append(score)
                channel_scores[param] = score

        if not all_scores:
            return 0.0, channel_scores

        # Global score: mean of channel scores
        raw_global = float(np.mean(all_scores))

        # Use POT threshold for calibration
        if self.global_threshold.threshold is not None:
            # Normalize relative to threshold
            normalized = raw_global / (self.global_threshold.threshold + 1e-8)
            global_score = min(normalized, 1.0)
        else:
            # Accumulate scores for threshold calibration
            self._score_history.append(raw_global)
            if len(self._score_history) >= 50:
                self.global_threshold.fit(list(self._score_history))
                global_score = raw_global / (self.global_threshold.threshold + 1e-8)
                global_score = min(global_score, 1.0)
            else:
                global_score = 0.0  # Not enough history yet

        # Normalize channel scores similarly
        for param in channel_scores:
            raw = channel_scores[param]
            if self.global_threshold.threshold:
                channel_scores[param] = min(raw / (self.global_threshold.threshold + 1e-8), 1.0)

        return global_score, channel_scores
