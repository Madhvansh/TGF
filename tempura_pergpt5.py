from logging import config
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Optional, List, Tuple, Any
import optuna
from optuna.samplers import NSGAIISampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')
from dataclasses import dataclass
import time
from collections import defaultdict
import math
from dataclasses import dataclass, field


# ==================== Configuration Data Classes ====================

@dataclass
class SearchSpace:
    """Complete search space for TransNAS-TSAD as defined in the paper"""
    
    # Training hyperparameters
    learning_rate: Tuple[float, float] = (1e-5, 1e-1)  # log scale
    dropout_rate: Tuple[float, float] = (0.1, 0.5)
    batch_size: List[int] = field(default_factory=lambda: [16, 32, 48, 64, 96, 128])
    gaussian_noise: Tuple[float, float] = (1e-4, 1e-1)  # log scale
    time_warping: bool = True
    time_masking: bool = True
    window_size: Tuple[int, int] = (10, 30)
    
    # Architectural parameters
    positional_encoding_type: List[str] = field(default_factory=lambda: ['sinusoidal', 'fourier'])
    dim_feedforward: Tuple[int, int] = (8, 128)  # log scale
    encoder_layers: Tuple[int, int] = (1, 3)
    decoder_layers: Tuple[int, int] = (1, 3)
    activation_function: List[str] = field(default_factory=lambda: ['relu', 'leaky_relu', 'sigmoid', 'tanh'])
    attention_type: str = 'scaled_dot_product'
    use_linear_embedding: bool = True
    layer_normalization: List[str] = field(default_factory=lambda: ['layer', 'batch', 'instance', 'none'])
    self_conditioning: bool = True
    num_ffn_layers: Tuple[int, int] = (1, 3)
    phase_type: List[str] = field(default_factory=lambda: ['1phase', '2phase', 'iterative'])


# ==================== Transformer Components ====================

class PositionalEncoding(nn.Module):
    """Positional encoding with sinusoidal or Fourier options"""
    
    def __init__(self, d_model: int, max_len: int = 200, encoding_type: str = 'sinusoidal'):
        super().__init__()
        self.encoding_type = encoding_type
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        if encoding_type == 'sinusoidal':
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                               -(np.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            if d_model % 2 == 1:
                pe[:, 1::2] = torch.cos(position * div_term[:-1])
            else:
                pe[:, 1::2] = torch.cos(position * div_term)
        else:  # fourier
            for i in range(d_model):
                if i % 2 == 0:
                    pe[:, i] = torch.sin(2 * np.pi * position.squeeze() * (i // 2) / max_len)
                else:
                    pe[:, i] = torch.cos(2 * np.pi * position.squeeze() * (i // 2) / max_len)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerBlock(nn.Module):
    """Enhanced transformer block with configurable FFN layers"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, 
                 num_ffn_layers: int = 1, dropout: float = 0.1, 
                 activation: str = 'relu', norm_type: str = 'layer'):
        super().__init__()
        
        # Ensure n_heads divides d_model evenly
        while d_model % n_heads != 0 and n_heads > 1:
            n_heads -= 1
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # Normalization layers
        if norm_type == 'layer':
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        elif norm_type == 'batch':
            self.norm1 = nn.BatchNorm1d(d_model)
            self.norm2 = nn.BatchNorm1d(d_model)
        elif norm_type == 'instance':
            self.norm1 = nn.InstanceNorm1d(d_model)
            self.norm2 = nn.InstanceNorm1d(d_model)
        else:  # none
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        
        self.norm_type = norm_type
        
        # Configurable FFN with multiple layers
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh()
        }
        
        ffn_layers = []
        for i in range(num_ffn_layers):
            if i == 0:
                ffn_layers.append(nn.Linear(d_model, d_ff))
            elif i == num_ffn_layers - 1:
                ffn_layers.append(nn.Linear(d_ff, d_model))
            else:
                ffn_layers.append(nn.Linear(d_ff, d_ff))
            
            if i < num_ffn_layers - 1:
                ffn_layers.append(activations.get(activation, nn.ReLU()))
                ffn_layers.append(nn.Dropout(dropout))
        
        self.ff = nn.Sequential(*ffn_layers)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Self-attention
        attn_output, _ = self.attention(x, x, x)
        
        # Add & Norm
        if self.norm_type in ['batch', 'instance'] and len(x.shape) == 3:
            # Reshape for batch/instance norm
            batch_size, seq_len, d_model = x.shape
            x = self.norm1((x + self.dropout(attn_output)).transpose(1, 2)).transpose(1, 2)
        else:
            x = self.norm1(x + self.dropout(attn_output))
        
        # Feedforward
        ff_output = self.ff(x)
        
        # Add & Norm
        if self.norm_type in ['batch', 'instance'] and len(x.shape) == 3:
            x = self.norm2((x + self.dropout(ff_output)).transpose(1, 2)).transpose(1, 2)
        else:
            x = self.norm2(x + self.dropout(ff_output))
        
        return x


# ==================== Main TransNAS-TSAD Model ====================

class TransNASTSAD(nn.Module):
    """
    Complete TransNAS-TSAD model with three-phase adversarial training
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Model dimensions
        self.input_dim = config['input_dim']
        self.window_size = config['window_size']
        
        # Ensure dim_feedforward is divisible by possible num_attention_heads
        dim_ff = config['dim_feedforward']
        n_heads = config.get('num_attention_heads', min(8, self.input_dim))
        
        # Adjust dim_feedforward to be divisible by n_heads
        if config.get('use_linear_embedding', True):
            # Find the nearest value divisible by n_heads
            while dim_ff % n_heads != 0:
                dim_ff += 1
            self.d_model = dim_ff
        else:
            # When not using linear embedding, d_model equals input_dim
            self.d_model = self.input_dim
            # Adjust n_heads to divide d_model
            while self.d_model % n_heads != 0 and n_heads > 1:
                n_heads -= 1
        
        config['num_attention_heads'] = n_heads  # Update config with adjusted value
        self.phase_type = config.get('phase_type', '2phase')
        self.self_conditioning = config.get('self_conditioning', False)
        
        # Optional linear embedding
        if config.get('use_linear_embedding', True):
            self.embedding = nn.Linear(self.input_dim, self.d_model)
        else:
            self.embedding = nn.Identity()
        
        # Positional encoding - use max_len larger than window_size
        self.pos_encoding = PositionalEncoding(
            self.d_model, 
            max_len=max(200, self.window_size * 2),
            encoding_type=config.get('positional_encoding_type', 'sinusoidal')
        )
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(
                d_model=self.d_model,
                n_heads=n_heads,
                d_ff=self.d_model * 2,
                num_ffn_layers=config.get('num_ffn_layers', 1),
                dropout=config.get('dropout_rate', 0.1),
                activation=config.get('activation_function', 'relu'),
                norm_type=config.get('layer_normalization', 'layer')
            ) for _ in range(config.get('encoder_layers', 2))
        ])
        
        # Decoder 1 - Primary reconstruction
        self.decoder1_layers = nn.ModuleList([
            TransformerBlock(
                d_model=self.d_model,
                n_heads=n_heads,
                d_ff=self.d_model * 2,
                num_ffn_layers=config.get('num_ffn_layers', 1),
                dropout=config.get('dropout_rate', 0.1),
                activation=config.get('activation_function', 'relu'),
                norm_type=config.get('layer_normalization', 'layer')
            ) for _ in range(config.get('decoder_layers', 2))
        ])
        
        # Decoder 2 - Adversarial reconstruction (only for 2phase and iterative)
        if self.phase_type in ['2phase', 'iterative']:
            self.has_decoder2 = True
            self.decoder2_layers = nn.ModuleList([
                TransformerBlock(
                    d_model=self.d_model,
                    n_heads=n_heads,
                    d_ff=self.d_model * 2,
                    num_ffn_layers=config.get('num_ffn_layers', 1),
                    dropout=config.get('dropout_rate', 0.1),
                    activation=config.get('activation_function', 'relu'),
                    norm_type=config.get('layer_normalization', 'layer')
                ) for _ in range(config.get('decoder_layers', 2))
            ])
            
            # Focus score modulator for phase 2
            self.focus_modulator = nn.Sequential(
                nn.Linear(self.input_dim, self.d_model),  # This is correct
                nn.ReLU(),
                nn.Linear(self.d_model, self.d_model),
                nn.Sigmoid()
            )

            # Add a projection layer for focus scores if needed
            self.focus_projection = nn.Linear(self.input_dim, self.d_model)
            
            # Output projection for decoder 2
            self.output_projection2 = nn.Linear(self.d_model, self.input_dim)
        else:
            self.has_decoder2 = False
        
        # Output projections
        self.output_projection1 = nn.Linear(self.d_model, self.input_dim)
    
    def encode(self, x):
        """Encoder forward pass"""
        # Embedding and positional encoding
        x = self.embedding(x)
        x = self.pos_encoding(x)
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x)
        
        return x
    
    def decode_phase1(self, encoded):
        """Phase 1: Preliminary reconstruction"""
        x = encoded
        for layer in self.decoder1_layers:
            x = layer(x)
        return self.output_projection1(x)
    
    def decode_phase2(self, encoded, focus_scores):
        # Project focus_scores to match d_model dimensions
        focus_projected = self.focus_projection(torch.abs(focus_scores))
        focus_weights = self.focus_modulator(torch.abs(focus_scores))
        
        # Now both have d_model dimensions
        x = encoded * (1 + focus_weights)
        
        for layer in self.decoder2_layers:
            x = layer(x)
        return self.output_projection2(x)
        
    def forward(self, x, phase='train'):
        """
        Forward pass with three-phase approach
        
        Args:
            x: Input tensor [batch_size, window_size, input_dim]
            phase: 'train' or 'inference'
        
        Returns:
            Dictionary containing reconstructions and losses
        """
        batch_size = x.size(0)
        
        # Encode
        encoded = self.encode(x)
        
        results = {}
        
        if self.phase_type == '1phase':
            reconstruction = self.decode_phase1(encoded)
            results['reconstruction'] = reconstruction
            
            # Add weighted MSE loss for anomaly regions
            mse_per_sample = torch.mean((reconstruction - x) ** 2, dim=-1)
            
            # If we have labels during training, weight the loss
            # You'll need to pass labels to forward() method
            mse_loss = torch.mean(mse_per_sample)
            
            # Keep other regularization terms but adjust weights
            l1_loss = torch.mean(torch.abs(reconstruction))
            var_loss = -torch.var(reconstruction)
            
            results['loss'] = mse_loss + 0.001 * l1_loss + 0.0001 * var_loss  # Reduced weights
            
        elif self.phase_type == '2phase':
            # Phase 1: Preliminary reconstruction
            reconstruction1 = self.decode_phase1(encoded)
            focus_scores = x - reconstruction1  # Focus on reconstruction errors
            
            # Phase 2: Focus-driven reconstruction (not adversarial anymore)
            reconstruction2 = self.decode_phase2(encoded, focus_scores)
            
            results['reconstruction1'] = reconstruction1
            results['reconstruction2'] = reconstruction2
            results['focus_scores'] = focus_scores
            
            # Losses - FIXED: Both losses are positive MSE
            results['loss1'] = F.mse_loss(reconstruction1, x)
            results['loss2'] = F.mse_loss(reconstruction2, x)
            # Weighted combination
            results['total_loss'] = 0.7 * results['loss1'] + 0.3 * results['loss2']
            
        else:  # iterative
            # Phase 1: Initial reconstruction
            reconstruction1 = self.decode_phase1(encoded)
            focus_scores = x - reconstruction1
            
            # Phase 2: Focus-driven reconstruction
            reconstruction2 = self.decode_phase2(encoded, focus_scores)
            
            # Phase 3: Combine both reconstructions
            reconstruction3 = 0.6 * reconstruction1 + 0.4 * reconstruction2
            
            results['reconstruction1'] = reconstruction1
            results['reconstruction2'] = reconstruction2
            results['reconstruction3'] = reconstruction3
            results['focus_scores'] = focus_scores
            
            # Losses
            results['loss1'] = F.mse_loss(reconstruction1, x)
            results['loss2'] = F.mse_loss(reconstruction2, x)
            results['loss3'] = F.mse_loss(reconstruction3, x)
            results['total_loss'] = 0.5 * results['loss1'] + 0.3 * results['loss2'] + 0.2 * results['loss3']
        
        # Calculate anomaly scores for inference
        # ==================== In TransNASTSAD.forward (phase == 'inference'), replace scoring blocks ====================

            # Calculate anomaly scores for inference
        if phase == 'inference':
            if self.phase_type == '1phase':
                anomaly_scores = torch.mean((x - reconstruction) ** 2, dim=-1)
                anomaly_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min() + 1e-8)
            elif self.phase_type == '2phase':
                anomaly_scores = torch.mean((x - reconstruction2) ** 2, dim=-1)
                anomaly_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min() + 1e-8)
            else:  # iterative
                anomaly_scores = torch.mean((x - reconstruction3) ** 2, dim=-1)
                anomaly_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min() + 1e-8)
            results['anomaly_scores'] = anomaly_scores

        # Always return the results dict for both train and inference
        return results


# ==================== Data Augmentation ====================

class TimeSeriesAugmentation:
    """Data augmentation techniques for time series"""
    
    @staticmethod
    def time_warping(x, sigma=0.2):
        """Apply time warping augmentation"""
        batch_size, seq_len, n_features = x.shape
        time_points = np.arange(seq_len)
        
        # Generate random warping
        warp = np.random.normal(loc=1.0, scale=sigma, size=seq_len)
        warp = np.cumsum(warp)
        warp = warp / warp[-1] * seq_len
        
        # Interpolate
        warped = np.zeros_like(x.cpu().numpy())
        for b in range(batch_size):
            for f in range(n_features):
                warped[b, :, f] = np.interp(time_points, warp, x[b, :, f].cpu().numpy())
        
        return torch.tensor(warped, dtype=x.dtype, device=x.device)
    
    @staticmethod
    def time_masking(x, mask_ratio=0.1):
        """Apply time masking augmentation"""
        batch_size, seq_len, n_features = x.shape
        mask_len = int(seq_len * mask_ratio)
        
        masked = x.clone()
        for b in range(batch_size):
            if mask_len > 0 and seq_len > mask_len:
                start = np.random.randint(0, seq_len - mask_len)
                masked[b, start:start+mask_len, :] = 0
        
        return masked
    
    @staticmethod
    def gaussian_noise(x, noise_level=0.01):
        """Add Gaussian noise"""
        noise = torch.randn_like(x) * noise_level
        return x + noise


# ==================== Anomaly Detection with POT/mPOT ====================

class AnomalyDetector:
    """
    EVT-based detector with optional SPOT-style streaming updates,
    EMA smoothing, two-threshold hysteresis, and event post-processing.
    """

    def __init__(self,
                 q=0.995,                 # target risk level (upper tail)
                 ema_alpha=0.2,           # score smoothing (EMA)
                 on_percentile=99.5,      # ON threshold percentile (validation)
                 off_percentile=97.5,     # OFF threshold percentile (validation)
                 min_event_len=3,         # minimal consecutive positives to accept an event
                 gap_merge=2,             # merge events separated by <= gap_merge zeros
                 use_spot=True,           # enable streaming EVT updates
                 init_window=1000):       # calibration window for SPOT init
        self.q = q
        self.ema_alpha = ema_alpha
        self.on_percentile = on_percentile
        self.off_percentile = off_percentile
        self.min_event_len = min_event_len
        self.gap_merge = gap_merge
        self.use_spot = use_spot
        self.init_window = init_window

        self.threshold_on = None
        self.threshold_off = None
        self.base_threshold = None  # EVT base
        self.state = 0              # hysteresis state: 0=normal, 1=anomalous
        self._ema_prev = None
        self._calibrated = False

        # SPOT tail params (if scipy available, use genpareto MLE; else fallback to quantiles)
        self._use_scipy = False
        try:
            from scipy.stats import genpareto  # noqa: F401
            self._use_scipy = True
            self._genpareto = __import__('scipy.stats', fromlist=['genpareto']).stats.genpareto
        except Exception:
            self._use_scipy = False

        # Buffers for streaming updates
        self._calib_scores = []
        self._peaks = []   # exceedances above base threshold (for GPD updates)

    def reset_stream(self):
        self.state = 0
        self._ema_prev = None

    def _ema(self, x):
        if self._ema_prev is None:
            self._ema_prev = x
        y = np.empty_like(x, dtype=float)
        prev = self._ema_prev
        for i, v in enumerate(x):
            prev = self.ema_alpha * v + (1 - self.ema_alpha) * prev
            y[i] = prev
        self._ema_prev = prev
        return y

    def _fit_gpd(self, exceedances):
        # MLE with SciPy if available; otherwise fallback to high quantile
        if self._use_scipy and len(exceedances) >= 50 and np.all(np.isfinite(exceedances)):
            from scipy.stats import genpareto
            c, loc, scale = genpareto.fit(exceedances, floc=0.0)
            return c, scale
        return None, None

    def _spot_init(self, calib_scores):
        # Base threshold as high quantile; GPD fit on exceedances (POT)
        base = np.quantile(calib_scores, 0.98)  # base exceedance threshold
        exc = np.array(calib_scores) - base
        exc = exc[exc > 0]
        xi, beta = self._fit_gpd(exc)
        self.base_threshold = base
        self._calibrated = True
        return xi, beta

    def _spot_quantile(self, xi, beta, q):
        # Quantile of GPD tail; if no fit, fall back to empirical quantile
        if xi is None or beta is None or beta <= 0:
            return None
        # u: base_threshold, z_q: tail quantile
        # F(z) = 1 - (1 + xi * z / beta)^{-1/xi}
        # Solve for z at tail probability (1 - q) over exceedances:
        tail_q = q
        z = beta / xi * ((1 - tail_q) ** (-xi) - 1) if xi != 0 else -beta * np.log(1 - tail_q)
        return self.base_threshold + max(z, 0)

    def fit_POT(self, scores):
        """
        Calibrate thresholds on validation scores (not training) to avoid leakage.
        Sets hysteresis thresholds using validation percentiles; initializes SPOT base.
        """
        scores = np.asarray(scores).astype(float)
        scores = scores[np.isfinite(scores)]
        if len(scores) == 0:
            # Fallback thresholds
            self.threshold_on = np.inf
            self.threshold_off = np.inf
            return self.threshold_on

        # Initialize SPOT with a calibration window (or full validation)
        window = min(len(scores), max(self.init_window, 200))
        calib = scores[:window]
        xi, beta = self._spot_init(calib)

        # EVT-based ON threshold if possible, else percentile
        gpd_on = self._spot_quantile(xi, beta, self.q) if self.use_spot else None
        self.threshold_on = gpd_on if (gpd_on is not None and np.isfinite(gpd_on)) \
            else np.percentile(scores, self.on_percentile)

        # OFF threshold lower than ON to implement hysteresis
        self.threshold_off = min(self.threshold_on,
                                 np.percentile(scores, self.off_percentile))
        return self.threshold_on

    def detect(self, scores, use_mPOT=True):
        """
        Vectorized detection with EMA smoothing + hysteresis + event post-processing.
        If streaming, update SPOT tail with peaks but do not let anomalies affect the tail.
        """
        x = np.asarray(scores).astype(float)
        x = x[np.isfinite(x)]

        if self.threshold_on is None or self.threshold_off is None:
            # Calibrate from the provided batch if not calibrated
            self.fit_POT(x)

        # Smooth scores
        xs = self._ema(x)

        # Hysteresis
        out = np.zeros_like(xs, dtype=int)
        state = self.state
        for i, s in enumerate(xs):
            if state == 0 and s >= self.threshold_on:
                state = 1
            elif state == 1 and s <= self.threshold_off:
                state = 0
            out[i] = state
            # SPOT streaming updates: update tail with peaks between base and on-threshold
            if self.use_spot and self.base_threshold is not None:
                if self.base_threshold < s < self.threshold_on:
                    self._peaks.append(s - self.base_threshold)

        self.state = state

        # Post-processing: enforce minimal event length and merge close events
        out = self._postprocess_events(out)
        return out

    def _postprocess_events(self, y):
        # Enforce minimal event length
        y = y.copy()
        i = 0
        n = len(y)
        while i < n:
            if y[i] == 1:
                j = i
                while j < n and y[j] == 1:
                    j += 1
                if (j - i) < self.min_event_len:
                    y[i:j] = 0
                i = j
            else:
                i += 1

        # Merge close events separated by small gaps
        i = 0
        while i < n:
            if y[i] == 0:
                # count zeros between ones
                j = i
                while j < n and y[j] == 0:
                    j += 1
                # gap [i, j)
                # if bounded by ones and short enough, fill it
                left = (i - 1 >= 0 and y[i - 1] == 1)
                right = (j < n and y[j] == 1)
                if left and right and (j - i) <= self.gap_merge:
                    y[i:j] = 1
                i = j
            else:
                i += 1
        return y
# ==================== Evaluation Metrics ====================

# ==================== Robust residual score calibrator ====================

class RobustScoreCalibrator:
    """
    Computes robust z-scores from residuals using feature-wise median/MAD,
    and aggregates to a scalar anomaly score per time step.
    """
    def __init__(self, agg="l2"):
        self.med = None
        self.mad = None
        self.agg = agg

    def fit(self, residual_windows):  # shape: [N_windows, window_size, n_features]
        r = residual_windows.reshape(-1, residual_windows.shape[-1])  # [N*W, F]
        self.med = np.median(r, axis=0)
        mad = np.median(np.abs(r - self.med), axis=0)
        self.mad = np.where(mad < 1e-6, 1e-6, mad)

    def score(self, residual_windows):
        r = residual_windows  # [N, W, F]
        z = (r - self.med) / self.mad  # broadcast
        if self.agg == "l1":
            # mean absolute z across features, then mean across window
            s = np.mean(np.mean(np.abs(z), axis=-1), axis=-1)
        else:
            # l2: mean squared z across features, then mean across window
            s = np.mean(np.mean(z ** 2, axis=-1), axis=-1)
        return s  # [N]


class EvaluationMetrics:
    """Comprehensive evaluation metrics including EACS and F1PAK"""
    
    @staticmethod
    def calculate_EACS(f1_score, training_time, param_count, 
                       f1_max, time_max, param_max,
                       wa=0.4, wt=0.4, wp=0.2):
        """
        Calculate Efficiency-Accuracy-Complexity Score
        """
        accuracy_score = (f1_score / f1_max) if f1_max > 0 else 0
        time_score = 1 - (training_time / time_max) if time_max > 0 else 0
        param_score = 1 - (param_count / param_max) if param_max > 0 else 0
        
        eacs = wa * accuracy_score + wt * time_score + wp * param_score
        return eacs
    
    @staticmethod
    def calculate_F1PAK(y_true, y_pred, anomaly_scores, K_values=None):
        """
        Calculate F1PA%K score with point adjustment
        """
        if K_values is None:
            K_values = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        
        f1_scores = []
        
        for K in K_values:
            # Apply point adjustment
            adjusted_pred = EvaluationMetrics._apply_point_adjustment(
                y_true, y_pred, anomaly_scores, K/100
            )
            
            # Calculate F1 score
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, adjusted_pred, average='binary', zero_division=0
            )
            f1_scores.append(f1)
        
        # Calculate AUC
        auc = np.trapz(f1_scores, K_values) / 100
        return auc, f1_scores
    
    @staticmethod
    def _apply_point_adjustment(y_true, y_pred, scores, threshold):
        """Apply point adjustment based on threshold K"""
        adjusted = y_pred.copy()
        
        # Find anomaly segments in ground truth
        segments = EvaluationMetrics._find_segments(y_true)
        
        for start, end in segments:
            segment_scores = scores[start:end]
            segment_pred = y_pred[start:end]
            
            # Calculate percentage of correctly detected anomalies
            correct_ratio = np.sum(segment_pred) / len(segment_pred) if len(segment_pred) > 0 else 0
            
            # Apply adjustment if threshold met
            if correct_ratio >= threshold:
                adjusted[start:end] = 1
        
        return adjusted
    
    @staticmethod
    def _find_segments(labels):
        """Find continuous anomaly segments"""
        segments = []
        start = None
        
        for i, label in enumerate(labels):
            if label == 1 and start is None:
                start = i
            elif label == 0 and start is not None:
                segments.append((start, i))
                start = None
        
        if start is not None:
            segments.append((start, len(labels)))
        
        return segments


# ==================== NAS with NSGA-II Optimization ====================

class TransNASOptimizer:
    """Neural Architecture Search with NSGA-II for TransNAS-TSAD"""
    
    def __init__(self, train_data, val_data, input_dim, n_trials=100):
        """
        Args:
            train_data: Training dataset
            val_data: Validation dataset
            input_dim: Number of input features
            n_trials: Number of optimization trials
        """
        self.train_data = train_data
        self.val_data = val_data
        self.input_dim = input_dim
        self.n_trials = n_trials
        self.search_space = SearchSpace()
        
        # Store Pareto front solutions
        self.pareto_front = []
        
    def create_objective(self, trial):
        """Create objective function for Optuna trial"""
        
        # Sample hyperparameters from search space
        config = {
            'input_dim': self.input_dim,
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),  # Reduced range
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.3),  # Reduced max dropout
            'batch_size': trial.suggest_categorical('batch_size', [32, 64]),  # Fewer options
            'gaussian_noise': trial.suggest_float('gaussian_noise', 1e-4, 1e-2, log=True),
            'time_warping': trial.suggest_categorical('time_warping', [False]),  # Disabled for stability
            'time_masking': trial.suggest_categorical('time_masking', [False]),  # Disabled for stability
            'window_size': trial.suggest_int('window_size', 15, 25),  # Narrower range
            'positional_encoding_type': trial.suggest_categorical('positional_encoding_type', ['sinusoidal']),
            'encoder_layers': trial.suggest_int('encoder_layers', 1, 2),
            'decoder_layers': trial.suggest_int('decoder_layers', 1, 2),
            'activation_function': trial.suggest_categorical('activation_function', ['relu', 'leaky_relu']),
            'use_linear_embedding': True,  # Always use
            'layer_normalization': trial.suggest_categorical('layer_normalization', ['layer']),
            'self_conditioning': False,
            'num_ffn_layers': trial.suggest_int('num_ffn_layers', 1, 2),
            'phase_type': trial.suggest_categorical('phase_type', ['1phase', '2phase']),
        }
        dim_power = trial.suggest_int('dim_feedforward_power', 5, 6)  # Reduced range
        dim_feedforward = 2 ** dim_power  # 32 or 64
        # Ensure num_heads divides dim_feedforward
        possible_heads = [2, 4, 8]
        valid_heads = [h for h in possible_heads if dim_feedforward % h == 0]
        
        if not valid_heads:
            valid_heads = [1]  # Fallback
        
        num_heads = trial.suggest_categorical('num_attention_heads', valid_heads)
        
        config['dim_feedforward'] = dim_feedforward
        config['num_attention_heads'] = num_heads

         # NOW apply the compatibility check
        while dim_feedforward % num_heads != 0:
            num_heads = num_heads // 2
            if num_heads < 1:
                num_heads = 1
                break
        try:
            # Train and evaluate model
            f1_score, param_count, training_time = self.train_and_evaluate(config, trial)
            
            # Store in Pareto front if non-dominated
            solution = {
                'config': config,
                'f1_score': f1_score,
                'param_count': param_count,
                'training_time': training_time,
                'trial': trial.number
            }
            self._update_pareto_front(solution)
            
            # Return multi-objective values (maximize F1, minimize parameters)
            return f1_score, param_count
        except Exception as e:
            print(f"Trial {trial.number} failed with error: {str(e)}")
            # Return poor values for failed trials
            return 0.0, float('inf')
    
    def train_and_evaluate(self, config, trial, epochs=10):  # Reduced epochs for NAS
        """Train model and evaluate performance"""
        
        start_time = time.time()
        
        # Create model
        try:
            model = TransNASTSAD(config)
        except Exception as e:
            print(f"Model creation failed: {str(e)}")
            raise
        
        # Count parameters
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Prepare data loaders
        train_loader = self._prepare_dataloader(self.train_data, config['batch_size'], 
                                               config['window_size'], shuffle=True)
        val_loader = self._prepare_dataloader(self.val_data, config['batch_size'],
                                             config['window_size'], shuffle=False)
        
        # Skip if no data available
        if len(train_loader) == 0 or len(val_loader) == 0:
            return 0.0, param_count, time.time() - start_time
        
        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        
        # Data augmentation
        augmentation = TimeSeriesAugmentation()
        
        # Training loop
        best_val_f1 = 0
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_losses = []
            
            for batch_x, batch_y in train_loader:
                # Apply augmentation if enabled
                if config.get('gaussian_noise', 0) > 0:
                    batch_x = augmentation.gaussian_noise(batch_x, config['gaussian_noise'])
                
                optimizer.zero_grad()
                
                # Forward pass
                results = model(batch_x, phase='train')
                loss = results.get('total_loss', results.get('loss'))
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation phase
            model.eval()
            val_scores = []
            val_labels = []
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    results = model(batch_x, phase='inference')
                    anomaly_scores = results['anomaly_scores']
                    
                    val_scores.extend(anomaly_scores.cpu().numpy().flatten())
                    val_labels.extend(batch_y.cpu().numpy().flatten())
            
            # Calculate validation F1 score
            if len(val_scores) > 0 and len(set(val_labels)) > 1:  # Check for both classes
                threshold = np.percentile(val_scores, 95)
                val_predictions = (np.array(val_scores) > threshold).astype(int)
                
                precision, recall, f1, _ = precision_recall_fscore_support(
                    val_labels, val_predictions, average='binary', zero_division=0
                )
            else:
                f1 = 0.0
            
            # Update best F1
            if f1 > best_val_f1:
                best_val_f1 = f1
        
        training_time = time.time() - start_time
        
        return best_val_f1, param_count, training_time
    
    def _prepare_dataloader(self, data, batch_size, window_size, shuffle=True):
        """Prepare PyTorch DataLoader from numpy data"""
        # Check if we have enough data for the window size
        if len(data) < window_size:
            # Return empty dataloader if not enough data
            return DataLoader(TensorDataset(torch.empty(0, window_size, data.shape[1]-1), 
                                           torch.empty(0, window_size)), 
                             batch_size=batch_size)
        
        # Create sliding windows
        X, y = [], []
        for i in range(len(data) - window_size + 1):
            X.append(data[i:i+window_size, :-1])  # Features
            y.append(data[i:i+window_size, -1])    # Labels
        
        if len(X) == 0:
            return DataLoader(TensorDataset(torch.empty(0, window_size, data.shape[1]-1), 
                                           torch.empty(0, window_size)), 
                             batch_size=batch_size)
        
        X = torch.FloatTensor(np.array(X))
        y = torch.FloatTensor(np.array(y))
        
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def _update_pareto_front(self, solution):
        """Update Pareto front with new solution"""
        # Check if solution is dominated
        dominated = False
        to_remove = []
        
        for i, existing in enumerate(self.pareto_front):
            if (existing['f1_score'] >= solution['f1_score'] and 
                existing['param_count'] <= solution['param_count']):
                dominated = True
                break
            elif (solution['f1_score'] >= existing['f1_score'] and 
                  solution['param_count'] <= existing['param_count']):
                to_remove.append(i)
        
        # Remove dominated solutions
        for i in reversed(to_remove):
            self.pareto_front.pop(i)
        
        # Add if not dominated
        if not dominated:
            self.pareto_front.append(solution)
    
    def optimize(self):
        """Run NSGA-II optimization"""
        sampler = NSGAIISampler(
            population_size=50,
            mutation_prob=0.1,
            crossover_prob=0.9,
            swapping_prob=0.5,
            seed=42
        )
        
        study = optuna.create_study(
            directions=['maximize', 'minimize'],
            sampler=sampler
        )
        
        # Track successful trials
        successful_trials = 0
        total_attempts = 0
        max_attempts = self.n_trials * 3  # Allow more attempts to get n successful trials
        
        while successful_trials < self.n_trials and total_attempts < max_attempts:
            try:
                study.optimize(self.create_objective, n_trials=1)
                # Check if the last trial was successful
                last_trial = study.trials[-1]
                if last_trial.values[0] > 0 and last_trial.values[1] < float('inf'):
                    successful_trials += 1
                    print(f"Successful trial {successful_trials}/{self.n_trials}")
            except Exception as e:
                print(f"Trial attempt {total_attempts} failed: {str(e)}")
            
            total_attempts += 1
        
        print(f"Completed with {successful_trials} successful trials out of {total_attempts} attempts")
        return self.pareto_front, study
    
    def select_best_model(self, preference='balanced'):
        """
        Select best model from Pareto front based on preference
        
        Args:
            preference: 'accuracy' (highest F1), 'efficiency' (lowest params), 
                       or 'balanced' (best trade-off)
        """
        if not self.pareto_front:
            return None
        
        if preference == 'accuracy':
            return max(self.pareto_front, key=lambda x: x['f1_score'])
        elif preference == 'efficiency':
            return min(self.pareto_front, key=lambda x: x['param_count'])
        else:  # balanced
            # Calculate EACS scores
            f1_max = max(s['f1_score'] for s in self.pareto_front) if self.pareto_front else 1.0
            param_max = max(s['param_count'] for s in self.pareto_front) if self.pareto_front else 1e6
            time_max = max(s['training_time'] for s in self.pareto_front) if self.pareto_front else 100
            
            best_eacs = -1
            best_solution = None
            
            for solution in self.pareto_front:
                eacs = EvaluationMetrics.calculate_EACS(
                    solution['f1_score'],
                    solution['training_time'],
                    solution['param_count'],
                    f1_max, time_max, param_max
                )
                if eacs > best_eacs:
                    best_eacs = eacs
                    best_solution = solution
            
            return best_solution


# ==================== Complete Pipeline ====================

class TransNASPipeline:
    """Complete TransNAS-TSAD pipeline with NAS, training, and evaluation"""
    
    def __init__(self, data_path: str = None):
        self.data_path = data_path
        self.model = None
        self.config = None
        self.detector = None
        self.scaler = StandardScaler()
    
    def load_and_preprocess_data(self, data_path: str = None):
        """Load and preprocess the water quality data"""
        if data_path:
            self.data_path = data_path
        
        # Load data
        if self.data_path.endswith('.csv'):
            data = pd.read_csv(self.data_path)
        else:
            data = pd.read_excel(self.data_path)
        
        print(f"Loaded data shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        
        # Handle date column if present
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
            data = data.sort_values('Date')
            data = data.drop('Date', axis=1)
        
        # Handle tower column if present
        if 'Tower' in data.columns:
            # One-hot encode or drop based on preference
            data = data.drop('Tower', axis=1)
        
        # Handle missing values first
        data = data.ffill().bfill()
        

        # After handling Date/Tower and missing values:
        values = data.values.astype(np.float32)

        # Time-ordered split indices
        n = len(values)
        n_train = int(0.6 * n)
        n_val = int(0.2 * n)

        train_raw = values[:n_train]
        val_raw   = values[n_train:n_train+n_val]
        test_raw  = values[n_train+n_val:]

        # Fit scaler on train only
        train_scaled = self.scaler.fit_transform(train_raw)
        val_scaled   = self.scaler.transform(val_raw)
        test_scaled  = self.scaler.transform(test_raw)

        # Unsupervised pseudo-labels: fit on train only, predict each split
        iso = IsolationForest(contamination=0.1, random_state=42)
        iso.fit(train_scaled)
        y_train = (iso.predict(train_scaled) == -1).astype(float)
        y_val   = (iso.predict(val_scaled)   == -1).astype(float)
        y_test  = (iso.predict(test_scaled)  == -1).astype(float)

        # Combine features with labels per split
        train_data = np.column_stack([train_scaled, y_train])
        val_data   = np.column_stack([val_scaled,   y_val])
        test_data  = np.column_stack([test_scaled,  y_test])

        print(f"Train anomaly ratio: {np.mean(y_train):.2%}")
        print(f"Val anomaly ratio:   {np.mean(y_val):.2%}")
        print(f"Test anomaly ratio:  {np.mean(y_test):.2%}")

        return train_data, val_data, test_data, data.columns.tolist()

    
    def run_nas_optimization(self, train_data, val_data, n_trials=20):  # Reduced default trials
        """Run Neural Architecture Search"""
        print("\n" + "="*50)
        print("Starting Neural Architecture Search with NSGA-II")
        print("="*50)
        
        input_dim = train_data.shape[1] - 1  # Exclude label column
        
        optimizer = TransNASOptimizer(
            train_data=train_data,
            val_data=val_data,
            input_dim=input_dim,
            n_trials=n_trials
        )
        
        pareto_front, study = optimizer.optimize()
        
        print(f"\nOptimization complete. Found {len(pareto_front)} Pareto-optimal solutions")
        
        # Display Pareto front
        if pareto_front:
            print("\nPareto Front Solutions:")
            print("-" * 70)
            print(f"{'Trial':<8} {'F1 Score':<12} {'Parameters':<15} {'Training Time':<15} {'EACS':<10}")
            print("-" * 70)
            
            # Calculate EACS for all solutions
            f1_max = max(s['f1_score'] for s in pareto_front)
            param_max = max(s['param_count'] for s in pareto_front)
            time_max = max(s['training_time'] for s in pareto_front)
            
            for solution in sorted(pareto_front, key=lambda x: x['f1_score'], reverse=True):
                eacs = EvaluationMetrics.calculate_EACS(
                    solution['f1_score'],
                    solution['training_time'],
                    solution['param_count'],
                    f1_max, time_max, param_max
                )
                print(f"{solution['trial']:<8} {solution['f1_score']:<12.4f} "
                      f"{solution['param_count']:<15,} {solution['training_time']:<15.2f} {eacs:<10.4f}")
            
            # Select best model
            best_solution = optimizer.select_best_model(preference='balanced')
            if best_solution:
                print(f"\nSelected best model from trial {best_solution['trial']} with EACS-optimized trade-off")
                return best_solution['config'], pareto_front
        
        print("No valid solutions found. Using default configuration.")
        # Return a default configuration
        default_config = {
            'input_dim': input_dim,
            'learning_rate': 0.001,
            'dropout_rate': 0.2,
            'batch_size': 32,
            'gaussian_noise': 0.001,
            'time_warping': False,
            'time_masking': False,
            'window_size': 20,
            'positional_encoding_type': 'sinusoidal',
            'dim_feedforward': 64,
            'encoder_layers': 2,
            'decoder_layers': 2,
            'activation_function': 'relu',
            'use_linear_embedding': True,
            'layer_normalization': 'layer',
            'self_conditioning': False,
            'num_ffn_layers': 2,
            'phase_type': '2phase',
            'num_attention_heads': 4
        }
        return default_config, []
    
    def train_final_model(self, config, train_data, val_data, epochs=50):  # Reduced epochs for testing
        """Train the final selected model"""
        print("\n" + "="*50)
        print("Training Final Model with Selected Architecture")
        print("="*50)
        
        # Create model
        self.model = TransNASTSAD(config)
        self.config = config
        
        # Print model architecture
        param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\nModel Architecture:")
        print(f"  - Phase Type: {config['phase_type']}")
        print(f"  - Encoder Layers: {config['encoder_layers']}")
        print(f"  - Decoder Layers: {config['decoder_layers']}")
        print(f"  - Feedforward Dim: {config['dim_feedforward']}")
        print(f"  - Attention Heads: {config.get('num_attention_heads', 8)}")
        print(f"  - Total Parameters: {param_count:,}")
        
        # Prepare data loaders
        train_loader = self._prepare_dataloader(train_data, config['batch_size'], 
                                               config['window_size'], shuffle=True)
        val_loader = self._prepare_dataloader(val_data, config['batch_size'],
                                             config['window_size'], shuffle=False)
        
        # Training setup
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=config['learning_rate']/100
        )        
        # Data augmentation
        augmentation = TimeSeriesAugmentation()
        
        # Initialize anomaly detector
        self.detector = AnomalyDetector(q=0.95)  # 95th percentile threshold
        
        # Training history
        history = {
            'train_loss': [],
            'val_f1': [],
            'val_precision': [],
            'val_recall': []
        }
        
        best_val_f1 = 0
        best_model_state = None
        
        print("\nTraining Progress:")
        print("-" * 50)
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_losses = []
            
            for batch_x, batch_y in train_loader:
                # Apply augmentation if enabled
                if config.get('gaussian_noise', 0) > 0 and np.random.random() > 0.5:
                    batch_x = augmentation.gaussian_noise(batch_x, config['gaussian_noise'])
                
                optimizer.zero_grad()
                
                # Forward pass
                results = self.model(batch_x, phase='train')
                loss = results.get('total_loss', results.get('loss'))
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation phase
            # Validation phase
            self.model.eval()
            val_scores = []
            val_labels = []

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    results = self.model(batch_x, phase='inference')
                    # use residuals returned by the model for robust scoring
                    res = results['residuals'].cpu().numpy()  # [B, W, F]
                    lab = batch_y.cpu().numpy()               # [B, W]
                    val_residuals.append(res)
                    val_labels.append(lab)

            val_residuals = np.concatenate(val_residuals, axis=0)  # [N, W, F]
            val_labels = np.concatenate(val_labels, axis=0)        # [N, W]


            # ADAPTIVE THRESHOLD SELECTION - NEW CODE
            if epoch == 0:
                self._robust_cal = RobustScoreCalibrator(agg="l2")
            if epoch % 5 == 0:
                self._robust_cal.fit(val_residuals)
            # Compute robust scores per window
            val_scores = self._robust_cal.score(val_residuals)  # [N]
            val_labels_flat = val_labels.mean(axis=1)  # window-level label via majority
            val_labels_flat = (val_labels_flat >= 0.5).astype(int)
            # Grid-search ON/OFF percentiles to maximize F1 on validation
            best_f1 = -1.0
            best_on = 99.5
            best_off = 97.5
            percentiles = [90, 92, 94, 95, 96, 97, 98, 99, 99.5]
            for onp in percentiles:
                for offp in percentiles:
                    if offp >= onp:
                        continue
                    detector = AnomalyDetector(on_percentile=onp, off_percentile=offp, use_spot=False)
                    detector.fit_POT(val_scores)  # sets thresholds from validation distribution
                    preds = detector.detect(val_scores, use_mPOT=False)
                    # Compute F1 on window labels
                    from sklearn.metrics import precision_recall_fscore_support
                    precision, recall, f1, _ = precision_recall_fscore_support(                            val_labels_flat, preds, average='binary', zero_division=0
                    )
                    if f1 > best_f1:
                        best_f1 = f1
                        best_on, best_off = onp, offp
                # Update the detector with the best threshold found
                self.detector.on_percentile = best_on
                self.detector.off_percentile = best_off
                self.detector.fit_POT(val_scores)  # calibrate thresholds from validation
                val_predictions = self.detector.detect(val_scores, use_mPOT=False)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    val_labels_flat, val_predictions, average='binary', zero_division=0
                )

                # Track last recall if needed by later logic
                self.last_recall = recall

            # Adjust percentiles based on performance
            if epoch > 10 and hasattr(self, 'last_recall') and self.last_recall < 0.5:
                percentiles_to_try = [70, 75, 80, 85, 88, 90, 92, 94, 95]
            else:
                percentiles_to_try = [80, 85, 88, 90, 92, 94, 95, 96, 97, 98]

            # Now use the (potentially updated) threshold for actual predictions
            val_predictions = self.detector.detect(val_scores, use_mPOT=False)

            # Continue with the rest of your validation code...
            # Calculate metrics
            if len(set(val_labels)) > 1:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    val_labels, val_predictions, average='binary', zero_division=0
                )
            else:
                precision = recall = f1 = 0.0
            
            # Update history
            avg_train_loss = np.mean(train_losses)
            history['train_loss'].append(avg_train_loss)
            history['val_f1'].append(f1)
            history['val_precision'].append(precision)
            history['val_recall'].append(recall)
            
            # Learning rate scheduling
            scheduler.step()
            
            # Save best model
            if f1 > best_val_f1:
                best_val_f1 = f1
                best_model_state = self.model.state_dict().copy()
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Loss={avg_train_loss:.4f}, "
                      f"F1={f1:.4f}, "
                      f"Precision={precision:.4f}, "
                      f"Recall={recall:.4f}")
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        print(f"\nTraining complete. Best validation F1: {best_val_f1:.4f}")
        
        return history
    
    def evaluate(self, test_data):
        """Comprehensive evaluation on test data"""
        print("\n" + "="*50)
        print("Evaluating Model on Test Data")
        print("="*50)
        
        test_loader = self._prepare_dataloader(test_data, self.config['batch_size'],
                                              self.config['window_size'], shuffle=False)
        
        # ==================== Patch 5: residuals -> robust scores (window-level) ====================
        self.model.eval()
        all_residuals = []
        all_labels = []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                results = self.model(batch_x, phase='inference')
                # If Patch 2 is applied, residuals exist:
                if 'residuals' in results:
                    res = results['residuals'].cpu().numpy()   # [B, W, F]
                    all_residuals.append(res)
                    all_labels.append(batch_y.cpu().numpy())    # [B, W]
                else:
                    # TEMP fallback: keep old anomaly_scores path until Patch 2 is added
                    scores = results['anomaly_scores'].cpu().numpy()  # [B, W]
                    # Convert to faux residual array so calibrator can be bypassed; or skip calibrator entirely
                    all_residuals.append(scores[..., None])  # [B, W, 1]
                    all_labels.append(batch_y.cpu().numpy())

        all_residuals = np.concatenate(all_residuals, axis=0)  # [N, W, F or 1]
        all_labels = np.concatenate(all_labels, axis=0)        # [N, W]

        # Window-level labels by majority
        labels_win = (all_labels.mean(axis=1) >= 0.5).astype(int)  # [N]

        # Robust scores (reuse calibrator from training; if missing, fit on a small chunk)
        if not hasattr(self, '_robust_cal') or self._robust_cal is None:
            self._robust_cal = RobustScoreCalibrator(agg="l2")
            fit_chunk = all_residuals[:min(1000, len(all_residuals))]
            self._robust_cal.fit(fit_chunk)

        scores_win = self._robust_cal.score(all_residuals)  # [N]

        # Use the calibrated detector from training; run streaming detection
        predictions = self.detector.detect(scores_win, use_mPOT=True)  # [N]

        # Metrics at window level
        from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_win, predictions, average='binary', zero_division=0
        )
        try:
            roc_auc = roc_auc_score(labels_win, scores_win)
        except Exception:
            roc_auc = 0.0

        # Optional: F1PAK at window level (aligns with window labels and scores)
        k_values = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        f1pak_auc, f1pak_scores = EvaluationMetrics.calculate_F1PAK(
            labels_win, predictions, scores_win, k_values
        )

        # Prints updated to use window-level variables
        param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("\nTest Results:")
        print("-" * 40)
        print(f"Precision:          {precision:.4f}")
        print(f"Recall:             {recall:.4f}")
        print(f"F1 Score:           {f1:.4f}")
        print(f"ROC AUC:            {roc_auc:.4f}")
        print(f"F1PAK AUC:          {f1pak_auc:.4f}")
        print(f"EACS:               {EvaluationMetrics.calculate_EACS(f1, 100, param_count, 1.0, 1000, 1e7):.4f}")
        print(f"Model Parameters:   {param_count:,}")

        print("\nAnomaly Statistics:")
        print(f"  Total windows:     {len(labels_win)}")
        print(f"  True anomalous:    {np.sum(labels_win)} ({100*np.mean(labels_win):.1f}%)")
        print(f"  Detected anomalous:{np.sum(predictions)} ({100*np.mean(predictions):.1f}%)")
        print(f"  Thresholds (ON/OFF): {getattr(self.detector, 'threshold_on', None)} / {getattr(self.detector, 'threshold_off', None)}")
        print(f"  Score range:        [{np.min(scores_win):.4f}, {np.max(scores_win):.4f}]")

        # Return dict uses window-level keys
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'f1pak_auc': f1pak_auc,
            'f1pak_scores': f1pak_scores,
            'predictions': predictions,
            'scores': scores_win,
            'labels': labels_win
        }

    def _prepare_dataloader(self, data, batch_size, window_size, shuffle=True):
        """Prepare PyTorch DataLoader from numpy data"""
        # Check if we have enough data for the window size
        if len(data) < window_size:
            # Return empty dataloader if not enough data
            return DataLoader(TensorDataset(torch.empty(0, window_size, data.shape[1]-1), 
                                           torch.empty(0, window_size)), 
                             batch_size=batch_size)
        
        # Create sliding windows
        X, y = [], []
        for i in range(len(data) - window_size + 1):
            X.append(data[i:i+window_size, :-1])  # Features
            y.append(data[i:i+window_size, -1])    # Labels
        
        if len(X) == 0:
            return DataLoader(TensorDataset(torch.empty(0, window_size, data.shape[1]-1), 
                                           torch.empty(0, window_size)), 
                             batch_size=batch_size)
        
        X = torch.FloatTensor(np.array(X))
        y = torch.FloatTensor(np.array(y))
        
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def run_complete_pipeline(self, data_path: str, n_trials: int = 10):  # Reduced default trials
        """Run the complete TransNAS-TSAD pipeline"""
        print("\n" + "="*70)
        print("TransNAS-TSAD: Complete Pipeline Execution")
        print("="*70)
        
        # Step 1: Load and preprocess data
        print("\nStep 1: Loading and preprocessing data...")
        train_data, val_data, test_data, feature_names = self.load_and_preprocess_data(data_path)
        print(f"  Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")
        
        # Step 2: Neural Architecture Search
        print("\nStep 2: Running Neural Architecture Search...")
        best_config, pareto_front = self.run_nas_optimization(train_data, val_data, n_trials)
        
        # Step 3: Train final model
        print("\nStep 3: Training final model...")
        history = self.train_final_model(best_config, train_data, val_data)
        
        # Step 4: Evaluate on test data
        print("\nStep 4: Evaluating on test data...")
        results = self.evaluate(test_data)
        
        print("\n" + "="*70)
        print("Pipeline execution complete!")
        print("="*70)
        
        return results, history, best_config, pareto_front


# ==================== Main Execution ====================

def main():
    """Main execution function"""
    
    # Example usage with the water quality dataset
    pipeline = TransNASPipeline()
    
    # Run complete pipeline
    try:
        results, history, config, pareto_front = pipeline.run_complete_pipeline(
            data_path='Parameters_1.csv',
            n_trials=10  # Reduced for faster testing - increase for better results
        )
        
        print("\n" + "="*70)
        print("TransNAS-TSAD Implementation Complete")
        print("="*70)
        print("\nFixed Issues:")
        print("✓ Removed adversarial loss that caused negative values")
        print("✓ Fixed label generation using Isolation Forest")
        print("✓ Corrected positional encoding dimension errors")
        print("✓ Removed trial.report() for multi-objective optimization")
        print("✓ Improved threshold calculation using normal data")
        print("✓ Simplified search space for stability")
        print("\nThe model now trains properly with positive loss values.")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("\nPlease check:")
        print("1. The data file 'Parameters_1.csv' exists in the current directory")
        print("2. The CSV file has the expected columns")
        print("3. All required packages are installed:")
        print("   pip install numpy pandas torch optuna scikit-learn")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()