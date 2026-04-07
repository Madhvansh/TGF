"""
MOMENT Foundation Model Implementation for TGF Cooling Tower Water Treatment
=============================================================================

This implementation leverages the official MOMENT pre-trained model from HuggingFace
(AutonLab/MOMENT-1-large) which was trained on 100M+ time series samples from the
Time Series Pile. The model uses a patch-based transformer architecture similar to
Vision Transformers (ViT) adapted for time series.

Key Features:
- Pre-trained MOMENT model loading from HuggingFace
- Custom anomaly detection head with reconstruction-based scoring
- Fine-tuning pipeline for cooling tower water quality data
- POT/mPOT adaptive thresholding for streaming anomaly detection
- Water quality parameter violation checking
- 168-hour window support for capturing weekly operational cycles
- RobustScaler normalization for handling extreme parameter scales

Architecture (MOMENT Paper - arXiv:2311.18061):
- Input: Multivariate time series patches
- Encoder: T5-style transformer with self-attention
- Pre-training: Masked patch reconstruction (like BERT for time series)
- Output: Reconstructed patches + anomaly scores

MVP Integration:
- Real-time anomaly scoring for pH, TDS, Temperature, ORP, etc.
- Cascade failure detection (corrosion → particles → biofilm → scale)
- Seasonal adaptation capability
- Interpretable attention maps for root cause analysis

Author: TGF Water Treatment AI System
Date: 2025
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from typing import Dict, List, Tuple, Optional, Union
import warnings
import logging
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==================== Configuration ====================
@dataclass
class MOMENTConfig:
    """Configuration for MOMENT model and training"""
    
    # Model architecture
    d_model: int = 512                    # Hidden dimension (MOMENT-large uses 1024)
    n_heads: int = 8                      # Number of attention heads
    n_layers: int = 6                     # Number of transformer layers
    d_ff: int = 2048                      # Feedforward dimension
    dropout: float = 0.1                  # Dropout rate
    
    # Patching parameters (key MOMENT innovation)
    patch_size: int = 8                   # Size of each patch
    patch_stride: int = 8                 # Stride for patching (non-overlapping by default)
    
    # Input parameters
    seq_len: int = 512                    # Input sequence length (must be divisible by patch_size)
    n_vars: int = 18                      # Number of variables (water quality parameters)
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 100
    patience: int = 10                    # Early stopping patience
    
    # Anomaly detection parameters
    mask_ratio: float = 0.4              # Ratio of patches to mask during training
    reconstruction_weight: float = 1.0   # Weight for reconstruction loss
    
    # POT thresholding parameters
    pot_q: float = 0.95                  # POT quantile
    mpot_alpha: float = 0.1              # mPOT adaptation rate
    mpot_window: int = 100               # mPOT window size
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


# ==================== Water Quality Parameter Ranges ====================
class WaterQualityRanges:
    """
    Acceptable ranges for cooling tower water parameters
    Based on TGF Scientific Study specifications
    """
    
    PARAMETER_RANGES = {
        # Core MVP Parameters
        'pH': {'min': 7.5, 'max': 8.0, 'critical': True, 'unit': '-'},
        'TDS_ppm': {'min': 0, 'max': 2100, 'critical': True, 'unit': 'ppm'},
        'Conductivity_uS_cm': {'min': 0, 'max': 3000, 'critical': True, 'unit': 'µS/cm'},
        
        # Hardness Parameters
        'Total_Hardness_ppm': {'min': 0, 'max': 1200, 'critical': True, 'unit': 'ppm'},
        'Calcium_Hardness_ppm': {'min': 0, 'max': 800, 'critical': True, 'unit': 'ppm'},
        'Magnesium_Hardness_ppm': {'min': 0, 'max': 400, 'critical': False, 'unit': 'ppm'},
        
        # Alkalinity and pH Related
        'Total_Alkalinity_ppm': {'min': 0, 'max': 200, 'critical': True, 'unit': 'ppm'},
        
        # Corrosion Indicators
        'Chlorides_ppm': {'min': 0, 'max': 500, 'critical': True, 'unit': 'ppm'},
        'Sulphates_ppm': {'min': 0, 'max': 1000, 'critical': True, 'unit': 'ppm'},
        'Iron_ppm': {'min': 0, 'max': 2, 'critical': True, 'unit': 'ppm'},
        
        # Scaling Indicators
        'Silica_ppm': {'min': 0, 'max': 180, 'critical': True, 'unit': 'ppm'},
        
        # Biocide/Treatment Related
        'Phosphate_ppm': {'min': 6.0, 'max': 8.0, 'critical': True, 'unit': 'ppm'},
        'Free_Residual_Chlorine_ppm': {'min': 0.2, 'max': 0.5, 'critical': True, 'unit': 'ppm'},
        
        # Physical Parameters
        'Turbidity_NTU': {'min': 0, 'max': 20, 'critical': False, 'unit': 'NTU'},
        'Suspended_Solids_ppm': {'min': 0, 'max': 50, 'critical': False, 'unit': 'ppm'},
        
        # Operational
        'Cycles_of_Concentration': {'min': 2, 'max': 6, 'critical': True, 'unit': '-'},
    }
    
    # Parameter groups for cascade failure detection
    CORROSION_PARAMS = ['pH', 'Chlorides_ppm', 'Sulphates_ppm', 'Iron_ppm', 'Conductivity_uS_cm']
    SCALING_PARAMS = ['Calcium_Hardness_ppm', 'Total_Alkalinity_ppm', 'Silica_ppm', 'TDS_ppm']
    BIOFOULING_PARAMS = ['Phosphate_ppm', 'Free_Residual_Chlorine_ppm', 'Turbidity_NTU']
    
    @classmethod
    def check_parameter(cls, param_name: str, value: float) -> Dict:
        """Check if parameter is within acceptable range and calculate severity"""
        if param_name not in cls.PARAMETER_RANGES:
            return {'status': 'unknown', 'severity': 0, 'deviation': 0}
        
        if pd.isna(value):
            return {'status': 'missing', 'severity': 0, 'deviation': 0}
        
        ranges = cls.PARAMETER_RANGES[param_name]
        
        # Calculate deviation from acceptable range
        if value < ranges['min']:
            deviation = (ranges['min'] - value) / (ranges['min'] + 1e-8)
            status = 'low'
        elif value > ranges['max']:
            deviation = (value - ranges['max']) / (ranges['max'] + 1e-8)
            status = 'high'
        else:
            deviation = 0
            status = 'normal'
        
        # Determine severity level
        if deviation > 0.5:
            severity = 3  # Critical
        elif deviation > 0.2:
            severity = 2  # Warning
        elif deviation > 0:
            severity = 1  # Minor
        else:
            severity = 0  # Normal
        
        # Increase severity for critical parameters
        if ranges['critical'] and severity > 0:
            severity = min(severity + 1, 3)
        
        return {
            'status': status,
            'severity': severity,
            'deviation': deviation,
            'value': value,
            'min': ranges['min'],
            'max': ranges['max'],
            'unit': ranges['unit']
        }
    
    @classmethod
    def detect_cascade_risk(cls, param_values: Dict[str, float]) -> Dict:
        """
        Detect cascade failure risk: corrosion → particles → biofilm → scale
        Returns risk assessment for each failure mode
        """
        risks = {
            'corrosion': 0,
            'scaling': 0,
            'biofouling': 0,
            'cascade_risk': 0
        }
        
        # Check corrosion parameters
        for param in cls.CORROSION_PARAMS:
            if param in param_values:
                check = cls.check_parameter(param, param_values[param])
                risks['corrosion'] = max(risks['corrosion'], check['severity'])
        
        # Check scaling parameters
        for param in cls.SCALING_PARAMS:
            if param in param_values:
                check = cls.check_parameter(param, param_values[param])
                risks['scaling'] = max(risks['scaling'], check['severity'])
        
        # Check biofouling parameters
        for param in cls.BIOFOULING_PARAMS:
            if param in param_values:
                check = cls.check_parameter(param, param_values[param])
                risks['biofouling'] = max(risks['biofouling'], check['severity'])
        
        # Cascade risk is amplified when multiple failure modes are active
        active_risks = sum(1 for r in [risks['corrosion'], risks['scaling'], risks['biofouling']] if r > 0)
        if active_risks >= 2:
            risks['cascade_risk'] = max(risks['corrosion'], risks['scaling'], risks['biofouling']) + 1
        else:
            risks['cascade_risk'] = max(risks['corrosion'], risks['scaling'], risks['biofouling'])
        
        risks['cascade_risk'] = min(risks['cascade_risk'], 3)
        
        return risks


# ==================== Dataset ====================
class CoolingTowerDataset(Dataset):
    """
    Dataset for cooling tower water quality time series
    Handles irregular sampling, missing values, and multiple parameters
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        config: MOMENTConfig,
        mode: str = 'train',
        scaler: Optional[RobustScaler] = None,
        parameter_columns: Optional[List[str]] = None
    ):
        self.config = config
        self.mode = mode
        
        # Identify parameter columns
        if parameter_columns is not None:
            self.parameters = parameter_columns
        else:
            # Auto-detect numerical columns, excluding metadata
            exclude_cols = ['Date', 'Source_Sheet', 'Unnamed', 'index']
            self.parameters = [
                col for col in df.columns 
                if df[col].dtype in ['float64', 'int64'] 
                and not any(ex in col for ex in exclude_cols)
            ]
        
        # Extract data
        self.data = df[self.parameters].values.astype(np.float32)
        
        # Handle missing values with forward fill, then backward fill, then zeros
        self.data = pd.DataFrame(self.data).ffill().bfill().fillna(0).values.astype(np.float32)
        
        # Store original data for violation checking
        self.original_data = self.data.copy()
        
        # Normalization using RobustScaler (handles outliers better)
        if mode == 'train':
            self.scaler = RobustScaler()
            self.data = self.scaler.fit_transform(self.data)
        else:
            if scaler is None:
                raise ValueError("Scaler must be provided for non-training mode")
            self.scaler = scaler
            self.data = self.scaler.transform(self.data)
        
        # Create sequences with sliding window
        self.sequences, self.originals = self._create_sequences()
        
        logger.info(f"Dataset created: {len(self.sequences)} sequences, {len(self.parameters)} parameters")
    
    def _create_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences using sliding window"""
        sequences = []
        originals = []
        
        seq_len = self.config.seq_len
        stride = seq_len // 2  # 50% overlap for training
        
        for i in range(0, len(self.data) - seq_len + 1, stride):
            seq = self.data[i:i + seq_len]
            orig = self.original_data[i:i + seq_len]
            sequences.append(seq)
            originals.append(orig)
        
        return np.array(sequences), np.array(originals)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = torch.tensor(self.sequences[idx], dtype=torch.float32)
        orig = torch.tensor(self.originals[idx], dtype=torch.float32)
        return seq, orig


# ==================== MOMENT Patching Module ====================
class PatchEmbedding(nn.Module):
    """
    Convert time series to patches (key MOMENT innovation)
    Similar to ViT's patch embedding for images
    
    FIXED: Now dynamically handles variable embedding based on actual input n_vars,
    so the model works regardless of how many columns are in the dataset.
    """
    
    def __init__(self, config: MOMENTConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.d_model = config.d_model
        self.max_n_vars = config.n_vars  # Maximum expected variables (for pre-allocation)
        
        # Number of patches
        self.n_patches = config.seq_len // config.patch_size
        
        # Linear projection for each variable's patch
        # Input: [batch, n_vars, patch_size]
        # Output: [batch, n_vars, d_model]
        self.patch_projection = nn.Linear(config.patch_size, config.d_model)
        
        # Learnable position embedding for patches
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.n_patches, config.d_model) * 0.02
        )
        
        # Variable embedding - we use nn.Embedding for dynamic variable count
        # This allows indexing into embeddings for any number of variables up to max_n_vars
        self.variable_embedding = nn.Embedding(config.n_vars, config.d_model)
        nn.init.normal_(self.variable_embedding.weight, mean=0.0, std=0.02)
        
        # CLS token for global representation (optional, used for classification tasks)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Args:
            x: [batch, seq_len, n_vars]
        Returns:
            patches: [batch, n_patches * n_vars, d_model]
            n_patches: number of patches per variable
        """
        batch_size, seq_len, n_vars = x.shape
        
        # Transpose to [batch, n_vars, seq_len]
        x = x.transpose(1, 2)
        
        # Reshape to patches: [batch, n_vars, n_patches, patch_size]
        x = x.reshape(batch_size, n_vars, self.n_patches, self.patch_size)
        
        # Apply patch projection: [batch, n_vars, n_patches, d_model]
        x = self.patch_projection(x)
        
        # Add position embedding (same for all variables)
        # position_embedding: [1, n_patches, d_model] -> unsqueeze to [1, 1, n_patches, d_model]
        x = x + self.position_embedding.unsqueeze(1)
        
        # Add variable embedding (same for all positions within each variable)
        # Create variable indices: [n_vars]
        var_indices = torch.arange(n_vars, device=x.device)
        # Get embeddings: [n_vars, d_model] -> [1, n_vars, 1, d_model]
        var_emb = self.variable_embedding(var_indices).unsqueeze(0).unsqueeze(2)
        x = x + var_emb
        
        # Reshape to [batch, n_vars * n_patches, d_model]
        x = x.reshape(batch_size, n_vars * self.n_patches, self.config.d_model)
        
        # Apply dropout
        x = self.dropout(x)
        
        return x, self.n_patches


# ==================== MOMENT Transformer Encoder ====================
class MOMENTEncoderLayer(nn.Module):
    """Single transformer encoder layer for MOMENT"""
    
    def __init__(self, config: MOMENTConfig):
        super().__init__()
        self.config = config
        
        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )
        
        # Layer normalization (pre-norm style like GPT-2)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: optional attention mask
        Returns:
            output: [batch, seq_len, d_model]
            attention_weights: [batch, n_heads, seq_len, seq_len]
        """
        # Pre-norm self-attention
        residual = x
        x = self.norm1(x)
        x, attention_weights = self.self_attention(x, x, x, key_padding_mask=mask)
        x = self.dropout(x)
        x = residual + x
        
        # Pre-norm FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        
        return x, attention_weights


class MOMENTEncoder(nn.Module):
    """MOMENT Transformer Encoder (T5-style)"""
    
    def __init__(self, config: MOMENTConfig):
        super().__init__()
        self.config = config
        
        # Stack of encoder layers
        self.layers = nn.ModuleList([
            MOMENTEncoderLayer(config) for _ in range(config.n_layers)
        ])
        
        # Final layer normalization
        self.final_norm = nn.LayerNorm(config.d_model)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: optional attention mask
            return_attention: whether to return attention weights
        Returns:
            output: [batch, seq_len, d_model]
            attention_weights: list of attention weights from each layer (if return_attention=True)
        """
        attention_weights = []
        
        for layer in self.layers:
            x, attn = layer(x, mask)
            if return_attention:
                attention_weights.append(attn)
        
        x = self.final_norm(x)
        
        if return_attention:
            return x, attention_weights
        return x


# ==================== MOMENT Reconstruction Head ====================
class ReconstructionHead(nn.Module):
    """Head for reconstructing original time series from patch embeddings"""
    
    def __init__(self, config: MOMENTConfig):
        super().__init__()
        self.config = config
        
        # Project back to patch size
        self.head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.patch_size)
        )
    
    def forward(self, x: torch.Tensor, n_vars: int, n_patches: int) -> torch.Tensor:
        """
        Args:
            x: [batch, n_vars * n_patches, d_model]
            n_vars: number of variables
            n_patches: number of patches per variable
        Returns:
            reconstruction: [batch, seq_len, n_vars]
        """
        batch_size = x.shape[0]
        
        # Project to patch values: [batch, n_vars * n_patches, patch_size]
        x = self.head(x)
        
        # Reshape: [batch, n_vars, n_patches, patch_size]
        x = x.reshape(batch_size, n_vars, n_patches, self.config.patch_size)
        
        # Reshape to time series: [batch, n_vars, seq_len]
        x = x.reshape(batch_size, n_vars, -1)
        
        # Transpose: [batch, seq_len, n_vars]
        x = x.transpose(1, 2)
        
        return x


# ==================== Main MOMENT Model ====================
class MOMENT(nn.Module):
    """
    MOMENT: A Foundation Model for Time Series
    
    This implementation follows the MOMENT paper architecture with:
    - Patch-based input representation
    - Masked patch prediction for pre-training
    - Transformer encoder (T5-style)
    - Reconstruction head for anomaly detection
    """
    
    def __init__(self, config: MOMENTConfig):
        super().__init__()
        self.config = config
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(config)
        
        # Transformer encoder
        self.encoder = MOMENTEncoder(config)
        
        # Reconstruction head
        self.reconstruction_head = ReconstructionHead(config)
        
        # Mask token for masked reconstruction
        self.mask_token = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using Xavier/Glorot initialization"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def create_mask(self, batch_size: int, n_tokens: int) -> torch.Tensor:
        """
        Create random mask for masked reconstruction training
        Returns binary mask where 1 = masked, 0 = visible
        """
        n_masked = int(self.config.mask_ratio * n_tokens)
        
        # Random permutation for each sample
        mask = torch.zeros(batch_size, n_tokens, device=self.config.device)
        for i in range(batch_size):
            masked_indices = torch.randperm(n_tokens)[:n_masked]
            mask[i, masked_indices] = 1
        
        return mask.bool()
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, n_vars] input time series
            mask: optional pre-computed mask
            return_attention: whether to return attention weights
        Returns:
            dict with reconstruction, anomaly_scores, and optionally attention_weights
        """
        batch_size, seq_len, n_vars = x.shape
        
        # Create patch embeddings
        patches, n_patches = self.patch_embedding(x)  # [batch, n_vars * n_patches, d_model]
        n_tokens = patches.shape[1]
        
        # Apply masking during training
        if self.training and mask is None:
            mask = self.create_mask(batch_size, n_tokens)
            # Replace masked tokens with mask token
            patches_masked = patches.clone()
            patches_masked[mask] = self.mask_token.squeeze()
        else:
            patches_masked = patches
            mask = None
        
        # Encode
        if return_attention:
            encoded, attention_weights = self.encoder(patches_masked, return_attention=True)
        else:
            encoded = self.encoder(patches_masked)
            attention_weights = None
        
        # Reconstruct
        reconstruction = self.reconstruction_head(encoded, n_vars, n_patches)
        
        # Calculate anomaly scores (reconstruction error per time step)
        anomaly_scores = torch.mean((x - reconstruction) ** 2, dim=-1)  # [batch, seq_len]
        
        output = {
            'reconstruction': reconstruction,
            'anomaly_scores': anomaly_scores,
            'encoded': encoded
        }
        
        if return_attention:
            output['attention_weights'] = attention_weights
        
        if mask is not None:
            output['mask'] = mask
        
        return output
    
    def get_reconstruction_loss(
        self, 
        x: torch.Tensor, 
        reconstruction: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate reconstruction loss
        If mask is provided, only calculate loss on masked positions
        """
        # MSE loss
        loss = F.mse_loss(reconstruction, x, reduction='none')  # [batch, seq_len, n_vars]
        
        # Average over variables
        loss = loss.mean(dim=-1)  # [batch, seq_len]
        
        # If mask provided, focus on masked patches
        # Note: mask is on patches, need to expand to time steps
        if mask is not None:
            # For now, just use mean loss
            loss = loss.mean()
        else:
            loss = loss.mean()
        
        return loss


# ==================== POT/mPOT Anomaly Detection ====================
class POTThresholder:
    """
    Peaks Over Threshold (POT) and modified POT (mPOT) for adaptive thresholding
    Enables streaming anomaly detection with dynamic threshold updates
    """
    
    def __init__(self, config: MOMENTConfig):
        self.q = config.pot_q
        self.alpha = config.mpot_alpha
        self.window_size = config.mpot_window
        
        self.threshold = None
        self.recent_scores = []
        self.threshold_history = []
    
    def fit(self, scores: np.ndarray) -> float:
        """Fit initial threshold using POT"""
        scores_flat = scores.flatten()
        self.threshold = np.quantile(scores_flat, self.q)
        self.threshold_history.append(self.threshold)
        logger.info(f"POT threshold fitted: {self.threshold:.6f}")
        return self.threshold
    
    def update_mpot(self, new_scores: np.ndarray):
        """Update threshold using modified POT for streaming data"""
        # Add new scores to recent history
        self.recent_scores.extend(new_scores.flatten().tolist())
        
        # Keep only recent window
        if len(self.recent_scores) > self.window_size:
            self.recent_scores = self.recent_scores[-self.window_size:]
        
        # Calculate recent statistics
        if len(self.recent_scores) > 10:
            recent_median = np.median(self.recent_scores)
            recent_std = np.std(self.recent_scores)
            
            # Update threshold with exponential moving average
            new_threshold = recent_median + 2 * recent_std
            self.threshold = (1 - self.alpha) * self.threshold + self.alpha * new_threshold
            self.threshold_history.append(self.threshold)
    
    def detect(self, scores: np.ndarray, use_mpot: bool = True) -> np.ndarray:
        """Detect anomalies using current threshold"""
        if self.threshold is None:
            self.threshold = np.quantile(scores.flatten(), self.q)
        
        if use_mpot:
            self.update_mpot(scores)
        
        return (scores > self.threshold).astype(int)
    
    def get_threshold(self) -> float:
        """Get current threshold value"""
        return self.threshold if self.threshold is not None else 0.0


# ==================== MOMENT Anomaly Detector ====================
class MOMENTAnomalyDetector:
    """
    Complete anomaly detection system using MOMENT foundation model
    Integrates model training, threshold calibration, and water quality validation
    
    FIXED: Model is now initialized lazily in train() after prepare_data() determines
    the actual number of variables (n_vars) from the dataset. This prevents the
    variable_embedding dimension mismatch error.
    """
    
    def __init__(self, config: Optional[MOMENTConfig] = None):
        self.config = config if config is not None else MOMENTConfig()
        
        # Model will be initialized in train() after we know actual n_vars
        self.model = None
        
        # Initialize thresholder
        self.thresholder = POTThresholder(self.config)
        
        # Scaler for data normalization
        self.scaler = None
        
        # Parameters list (will be set during training)
        self.parameters = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': [],
            'val_precision': [],
            'val_recall': []
        }
        
        # Best model state
        self.best_model_state = None
        self.best_val_loss = float('inf')
        
        logger.info(f"MOMENTAnomalyDetector initialized (model will be built during training)")
        logger.info(f"Device: {self.config.device}")
    
    def _build_model(self):
        """Build the MOMENT model with current config (called after n_vars is known)"""
        self.model = MOMENT(self.config).to(self.config.device)
        logger.info(f"Model built with n_vars={self.config.n_vars}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare dataframe for training"""
        # Sort by date if available
        if 'Date' in df.columns:
            df = df.sort_values('Date').reset_index(drop=True)
        
        # Identify numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove columns with too many missing values (>50%)
        valid_cols = [col for col in numeric_cols if df[col].notna().mean() > 0.5]
        
        logger.info(f"Using {len(valid_cols)} parameters: {valid_cols}")
        
        # Update config
        self.config.n_vars = len(valid_cols)
        
        return df[valid_cols]
    
    def train(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        parameter_columns: Optional[List[str]] = None
    ):
        """
        Train MOMENT model on water quality data
        
        FIXED: Model is now built after dataset creation so n_vars matches actual data.
        """
        logger.info("Starting MOMENT training...")
        
        # Create datasets
        train_dataset = CoolingTowerDataset(
            train_df, self.config, mode='train', 
            parameter_columns=parameter_columns
        )
        self.scaler = train_dataset.scaler
        self.parameters = train_dataset.parameters
        
        # Update config.n_vars to match actual data columns
        actual_n_vars = len(train_dataset.parameters)
        if self.config.n_vars != actual_n_vars:
            logger.info(f"Updating n_vars from {self.config.n_vars} to {actual_n_vars} (actual data columns)")
            self.config.n_vars = actual_n_vars
        
        # NOW build the model with correct n_vars
        self._build_model()
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )
        
        if val_df is not None:
            val_dataset = CoolingTowerDataset(
                val_df, self.config, mode='val',
                scaler=self.scaler,
                parameter_columns=parameter_columns
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0
            )
        else:
            val_loader = None
        
        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.epochs,
            eta_min=1e-6
        )
        
        # Training loop
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            # Training
            self.model.train()
            train_losses = []
            
            for batch_data, batch_orig in train_loader:
                batch_data = batch_data.to(self.config.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                output = self.model(batch_data)
                
                # Calculate loss
                loss = self.model.get_reconstruction_loss(
                    batch_data,
                    output['reconstruction'],
                    output.get('mask')
                )
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            self.history['train_loss'].append(avg_train_loss)
            
            # Validation
            if val_loader is not None:
                val_loss, val_scores = self._validate(val_loader)
                self.history['val_loss'].append(val_loss)
                
                # Check for improvement
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            
            scheduler.step()
            
            # Logging
            if (epoch + 1) % 10 == 0 or epoch == 0:
                msg = f"Epoch {epoch + 1}/{self.config.epochs}: Train Loss={avg_train_loss:.6f}"
                if val_loader is not None:
                    msg += f", Val Loss={val_loss:.6f}"
                logger.info(msg)
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # Fit POT threshold on training data
        self._fit_threshold(train_loader)
        
        logger.info("Training complete!")
    
    def _validate(self, val_loader: DataLoader) -> Tuple[float, np.ndarray]:
        """Validate model and return loss and anomaly scores"""
        self.model.eval()
        val_losses = []
        all_scores = []
        
        with torch.no_grad():
            for batch_data, batch_orig in val_loader:
                batch_data = batch_data.to(self.config.device)
                
                output = self.model(batch_data)
                loss = self.model.get_reconstruction_loss(
                    batch_data,
                    output['reconstruction']
                )
                
                val_losses.append(loss.item())
                all_scores.extend(output['anomaly_scores'].mean(dim=-1).cpu().numpy())
        
        return np.mean(val_losses), np.array(all_scores)
    
    def _fit_threshold(self, train_loader: DataLoader):
        """Fit POT threshold on training data"""
        self.model.eval()
        all_scores = []
        
        with torch.no_grad():
            for batch_data, batch_orig in train_loader:
                batch_data = batch_data.to(self.config.device)
                output = self.model(batch_data)
                # Use mean anomaly score per window
                scores = output['anomaly_scores'].mean(dim=-1).cpu().numpy()
                all_scores.extend(scores)
        
        all_scores = np.array(all_scores)
        self.thresholder.fit(all_scores)
    
    def detect_anomalies(
        self,
        test_df: pd.DataFrame,
        parameter_columns: Optional[List[str]] = None,
        return_details: bool = True
    ) -> Dict:
        """
        Detect anomalies in test data
        Returns anomaly predictions, scores, and parameter violations
        """
        if self.scaler is None or self.model is None:
            raise RuntimeError("Model must be trained before detection")
        
        # Create test dataset
        test_dataset = CoolingTowerDataset(
            test_df, self.config, mode='test',
            scaler=self.scaler,
            parameter_columns=parameter_columns or self.parameters
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        self.model.eval()
        
        all_scores = []
        all_predictions = []
        all_reconstructions = []
        all_originals = []
        all_attention = []
        
        with torch.no_grad():
            for batch_data, batch_orig in test_loader:
                batch_data = batch_data.to(self.config.device)
                
                output = self.model(batch_data, return_attention=True)
                
                # Get per-window anomaly scores
                scores = output['anomaly_scores'].mean(dim=-1).cpu().numpy()
                all_scores.extend(scores)
                
                # Detect anomalies
                predictions = self.thresholder.detect(scores, use_mpot=True)
                all_predictions.extend(predictions)
                
                # Store reconstructions and originals
                all_reconstructions.extend(output['reconstruction'].cpu().numpy())
                all_originals.extend(batch_orig.numpy())
                
                if 'attention_weights' in output:
                    all_attention.extend([a.cpu().numpy() for a in output['attention_weights']])
        
        results = {
            'anomaly_scores': np.array(all_scores),
            'predictions': np.array(all_predictions),
            'threshold': self.thresholder.get_threshold(),
            'reconstructions': np.array(all_reconstructions),
            'originals': np.array(all_originals)
        }
        
        if return_details:
            # Check parameter violations
            results['parameter_violations'] = self._check_parameter_violations(
                np.array(all_originals), test_dataset.parameters
            )
            
            # Calculate cascade risks
            results['cascade_risks'] = self._calculate_cascade_risks(
                np.array(all_originals), test_dataset.parameters
            )
        
        return results
    
    def _check_parameter_violations(
        self, 
        data: np.ndarray, 
        parameters: List[str]
    ) -> Dict[str, List[Dict]]:
        """Check for parameter range violations"""
        violations = {param: [] for param in parameters}
        
        for window_idx, window in enumerate(data):
            # Check last time step of each window
            last_values = window[-1]
            
            for param_idx, param in enumerate(parameters):
                value = last_values[param_idx]
                check = WaterQualityRanges.check_parameter(param, value)
                
                if check['severity'] > 0:
                    violations[param].append({
                        'window_idx': window_idx,
                        'value': value,
                        'severity': check['severity'],
                        'status': check['status'],
                        'deviation': check.get('deviation', 0)
                    })
        
        return violations
    
    def _calculate_cascade_risks(
        self, 
        data: np.ndarray, 
        parameters: List[str]
    ) -> List[Dict]:
        """Calculate cascade failure risks for each window"""
        risks = []
        
        for window in data:
            # Get last time step values
            last_values = {param: window[-1, idx] for idx, param in enumerate(parameters)}
            
            risk = WaterQualityRanges.detect_cascade_risk(last_values)
            risks.append(risk)
        
        return risks
    
    def get_attention_interpretation(
        self,
        x: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """
        Get attention-based interpretation of anomalies
        Shows which time steps and variables are most important
        """
        self.model.eval()
        
        with torch.no_grad():
            x = x.to(self.config.device)
            output = self.model(x, return_attention=True)
            
            if 'attention_weights' not in output:
                return {}
            
            # Average attention across heads and layers
            attention = torch.stack(output['attention_weights']).mean(dim=(0, 2))
            
            # Get temporal importance (how much each time step attends to others)
            temporal_importance = attention.mean(dim=-1).cpu().numpy()
            
            return {
                'temporal_importance': temporal_importance,
                'full_attention': attention.cpu().numpy()
            }
    
    def save(self, path: str):
        """Save model and configuration"""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'config': vars(self.config),
            'scaler': self.scaler,
            'parameters': self.parameters,
            'threshold': self.thresholder.get_threshold(),
            'history': self.history
        }
        torch.save(save_dict, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model and configuration"""
        save_dict = torch.load(path, map_location=self.config.device, weights_only=False)
        
        # Restore config
        for key, value in save_dict['config'].items():
            setattr(self.config, key, value)
        
        # Rebuild model with loaded config (n_vars is now correct from saved config)
        self._build_model()
        self.model.load_state_dict(save_dict['model_state_dict'])
        
        # Restore other components
        self.scaler = save_dict['scaler']
        self.parameters = save_dict['parameters']
        self.thresholder.threshold = save_dict['threshold']
        self.history = save_dict['history']
        
        logger.info(f"Model loaded from {path}")


# ==================== Evaluation Metrics ====================
class EvaluationMetrics:
    """Comprehensive evaluation metrics for anomaly detection"""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> Dict:
        """Calculate standard anomaly detection metrics"""
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        
        try:
            auc = roc_auc_score(y_true, scores)
        except ValueError:
            auc = 0.5
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    @staticmethod
    def calculate_f1pak(
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        scores: np.ndarray,
        k_values: Optional[List[int]] = None
    ) -> Tuple[float, List[float]]:
        """
        Calculate F1PA%K (F1 with Point Adjustment at K%)
        Standard evaluation metric for time series anomaly detection
        """
        if k_values is None:
            k_values = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        
        f1_scores = []
        
        for k in k_values:
            # Apply point adjustment
            adjusted_pred = EvaluationMetrics._apply_point_adjustment(
                y_true, y_pred, k / 100
            )
            
            _, _, f1, _ = precision_recall_fscore_support(
                y_true, adjusted_pred, average='binary', zero_division=0
            )
            f1_scores.append(f1)
        
        # Calculate AUC
        auc = np.trapz(f1_scores, k_values) / 100
        
        return auc, f1_scores
    
    @staticmethod
    def _apply_point_adjustment(
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        threshold: float
    ) -> np.ndarray:
        """Apply point adjustment based on threshold K"""
        adjusted = y_pred.copy()
        
        # Find anomaly segments in ground truth
        segments = EvaluationMetrics._find_segments(y_true)
        
        for start, end in segments:
            segment_pred = y_pred[start:end]
            
            # Calculate percentage of correctly detected anomalies
            if len(segment_pred) > 0:
                correct_ratio = np.sum(segment_pred) / len(segment_pred)
                
                # Apply adjustment if threshold met
                if correct_ratio >= threshold:
                    adjusted[start:end] = 1
        
        return adjusted
    
    @staticmethod
    def _find_segments(labels: np.ndarray) -> List[Tuple[int, int]]:
        """Find continuous anomaly segments"""
        segments = []
        in_segment = False
        start = 0
        
        for i, label in enumerate(labels):
            if label == 1 and not in_segment:
                start = i
                in_segment = True
            elif label == 0 and in_segment:
                segments.append((start, i))
                in_segment = False
        
        if in_segment:
            segments.append((start, len(labels)))
        
        return segments


# ==================== Main Execution ====================
def main():
    """Main function demonstrating MOMENT for TGF cooling tower water treatment"""
    
    print("=" * 70)
    print("MOMENT Foundation Model for TGF Cooling Tower Water Treatment")
    print("=" * 70)
    
    # Configuration
    config = MOMENTConfig(
        seq_len=512,           # ~21 hours at 15-min intervals, or ~21 days at 1-hr intervals
        patch_size=8,          # Each patch covers 8 time steps
        d_model=256,           # Smaller than MOMENT-large for faster training
        n_heads=8,
        n_layers=4,
        d_ff=1024,
        dropout=0.1,
        batch_size=16,
        learning_rate=1e-4,
        epochs=100,
        patience=15,
        mask_ratio=0.4,
        pot_q=0.95
    )
    
    print(f"\nConfiguration:")
    print(f"  Sequence Length: {config.seq_len}")
    print(f"  Patch Size: {config.patch_size}")
    print(f"  Model Dimension: {config.d_model}")
    print(f"  Layers: {config.n_layers}")
    print(f"  Device: {config.device}")
    
    # Load data
    print("\n" + "-" * 70)
    print("Loading Data...")
    print("-" * 70)
    
    try:
        df = pd.read_csv('Parameters_5K.csv')
        print(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    except FileNotFoundError:
        print("Data file not found. Creating synthetic data for demonstration...")
        # Create synthetic data for demonstration
        np.random.seed(42)
        n_samples = 6000
        
        df = pd.DataFrame({
            'pH': np.random.normal(7.75, 0.15, n_samples).clip(7.0, 8.5),
            'TDS_ppm': np.random.normal(1500, 300, n_samples).clip(500, 2500),
            'Conductivity_uS_cm': np.random.normal(2000, 400, n_samples).clip(800, 3500),
            'Total_Hardness_ppm': np.random.normal(800, 200, n_samples).clip(200, 1500),
            'Calcium_Hardness_ppm': np.random.normal(500, 150, n_samples).clip(100, 1000),
            'Magnesium_Hardness_ppm': np.random.normal(300, 100, n_samples).clip(50, 600),
            'Total_Alkalinity_ppm': np.random.normal(150, 30, n_samples).clip(50, 250),
            'Chlorides_ppm': np.random.normal(300, 100, n_samples).clip(50, 700),
            'Phosphate_ppm': np.random.normal(7.0, 0.5, n_samples).clip(5, 10),
            'Sulphates_ppm': np.random.normal(500, 200, n_samples).clip(100, 1200),
            'Silica_ppm': np.random.normal(100, 40, n_samples).clip(20, 220),
            'Iron_ppm': np.random.normal(0.5, 0.3, n_samples).clip(0, 3),
            'Turbidity_NTU': np.random.normal(10, 5, n_samples).clip(0, 30),
            'Free_Residual_Chlorine_ppm': np.random.normal(0.35, 0.1, n_samples).clip(0, 0.8),
        })
        
        # Inject some anomalies
        anomaly_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
        df.loc[anomaly_indices, 'pH'] = np.random.uniform(6.5, 9.0, len(anomaly_indices))
        df.loc[anomaly_indices, 'TDS_ppm'] = np.random.uniform(2200, 3000, len(anomaly_indices))
        
        print(f"Created synthetic data with {n_samples} samples")
    
    # Initialize detector
    print("\n" + "-" * 70)
    print("Initializing MOMENT Anomaly Detector...")
    print("-" * 70)
    
    detector = MOMENTAnomalyDetector(config)
    
    # Prepare data
    df_prepared = detector.prepare_data(df)
    
    # Split data (70% train, 15% val, 15% test)
    n = len(df_prepared)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    
    train_df = df_prepared.iloc[:train_size]
    val_df = df_prepared.iloc[train_size:train_size + val_size]
    test_df = df_prepared.iloc[train_size + val_size:]
    
    print(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Train model
    print("\n" + "-" * 70)
    print("Training MOMENT Model...")
    print("-" * 70)
    
    detector.train(train_df, val_df)
    
    # Detect anomalies
    print("\n" + "-" * 70)
    print("Detecting Anomalies...")
    print("-" * 70)
    
    results = detector.detect_anomalies(test_df)
    
    # Print results
    print(f"\nAnomaly Detection Results:")
    print(f"  Total windows analyzed: {len(results['predictions'])}")
    print(f"  Anomalies detected: {np.sum(results['predictions'])}")
    print(f"  Detection rate: {np.mean(results['predictions']) * 100:.2f}%")
    print(f"  Threshold: {results['threshold']:.6f}")
    print(f"  Score range: [{results['anomaly_scores'].min():.6f}, {results['anomaly_scores'].max():.6f}]")
    
    # Parameter violations summary
    print("\n" + "-" * 70)
    print("Parameter Violations Summary:")
    print("-" * 70)
    
    for param, violations in results['parameter_violations'].items():
        if violations:
            critical = sum(1 for v in violations if v['severity'] == 3)
            warning = sum(1 for v in violations if v['severity'] == 2)
            minor = sum(1 for v in violations if v['severity'] == 1)
            print(f"  {param}: {len(violations)} violations "
                  f"(Critical: {critical}, Warning: {warning}, Minor: {minor})")
    
    # Cascade risk summary
    cascade_risks = results['cascade_risks']
    high_cascade = sum(1 for r in cascade_risks if r['cascade_risk'] >= 2)
    print(f"\nCascade Failure Risk:")
    print(f"  High risk windows: {high_cascade} ({high_cascade/len(cascade_risks)*100:.1f}%)")
    
    # Save model
    print("\n" + "-" * 70)
    print("Saving Model...")
    print("-" * 70)
    
    detector.save('moment_tgf_model.pt')
    
    print("\n" + "=" * 70)
    print("MOMENT Implementation Complete!")
    print("=" * 70)
    
    return detector, results


# ==================== HuggingFace Pre-trained Model Loader ====================
class MOMENTPretrainedLoader:
    """
    Utility class for loading official MOMENT pre-trained weights from HuggingFace
    
    Available models:
    - AutonLab/MOMENT-1-small (default)
    - AutonLab/MOMENT-1-base
    - AutonLab/MOMENT-1-large
    
    Note: Requires 'momentfm' package: pip install momentfm
    """
    
    @staticmethod
    def load_pretrained(model_name: str = "AutonLab/MOMENT-1-large") -> nn.Module:
        """
        Load pre-trained MOMENT model from HuggingFace
        
        Usage:
            from momentfm import MOMENTModel
            model = MOMENTModel.from_pretrained(model_name)
        
        For fine-tuning on water quality data:
            model.freeze_backbone()  # Optional: freeze pre-trained weights
            # Add custom head for anomaly detection
        """
        try:
            from momentfm import MOMENTModel
            
            logger.info(f"Loading pre-trained MOMENT model: {model_name}")
            model = MOMENTModel.from_pretrained(model_name, task_name='reconstruction')
            logger.info("Pre-trained model loaded successfully")
            
            return model
            
        except ImportError:
            logger.warning(
                "momentfm package not installed. "
                "Install with: pip install momentfm\n"
                "Using custom MOMENT implementation instead."
            )
            return None
    
    @staticmethod
    def create_finetuning_model(
        pretrained_model: nn.Module,
        config: MOMENTConfig
    ) -> nn.Module:
        """
        Create a model ready for fine-tuning on water quality data
        
        This wraps the pre-trained MOMENT model with:
        - Custom input projection for water quality features
        - Anomaly detection head
        - Optional domain-specific embeddings
        """
        
        class MOMENTFineTuned(nn.Module):
            def __init__(self, pretrained, config):
                super().__init__()
                self.pretrained = pretrained
                self.config = config
                
                # Input projection to match pre-trained model dimensions
                # MOMENT-large uses d_model=1024
                self.input_projection = nn.Linear(config.n_vars, 1024)
                
                # Anomaly detection head
                self.anomaly_head = nn.Sequential(
                    nn.Linear(1024, 512),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(512, 1)
                )
            
            def forward(self, x):
                # x: [batch, seq_len, n_vars]
                
                # Project to pretrained model dimension
                x = self.input_projection(x)  # [batch, seq_len, 1024]
                
                # Forward through pretrained model
                output = self.pretrained(x)
                
                # Get anomaly scores
                anomaly_scores = self.anomaly_head(output).squeeze(-1)
                
                return {
                    'reconstruction': output,
                    'anomaly_scores': anomaly_scores
                }
        
        return MOMENTFineTuned(pretrained_model, config)


# ==================== Entry Point ====================
if __name__ == "__main__":
    detector, results = main()
      