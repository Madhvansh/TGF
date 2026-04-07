#!/usr/bin/env python3
"""
MOMENT-Based Anomaly Detection System for TGF MVP
==================================================
A production-ready implementation of MOMENT (foundation model for time series)
adapted for cooling tower water quality anomaly detection.

Architecture follows the MOMENT paper principles:
- Patch-based time series representation
- Transformer backbone with reconstruction objective
- Unsupervised anomaly detection via reconstruction error

Author: TGF Project
Version: 1.0.0 MVP
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from typing import Dict, List, Tuple, Optional, Any
import logging
import os
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================

CONFIG = {
    # Data parameters
    'data_path': './Parameters_5K.csv',
    'output_dir': './moment_outputs',
    
    # Feature selection - parameters with good coverage for MVP
    'feature_columns': [
        'pH', 'Turbidity_NTU', 'TDS_ppm', 'Total_Hardness_ppm',
        'Calcium_Hardness_ppm', 'Magnesium_Hardness_ppm', 'Chlorides_ppm',
        'Phosphate_ppm', 'Silica_ppm', 'Conductivity_uS_cm'
    ],
    
    # MOMENT architecture parameters
    'window_size': 64,           # Sequence length for each window
    'stride': 16,                # Sliding window stride
    'patch_size': 8,             # Patch size for tokenization
    'd_model': 256,              # Transformer model dimension
    'n_heads': 8,                # Number of attention heads
    'n_encoder_layers': 4,       # Number of encoder layers
    'n_decoder_layers': 2,       # Number of decoder layers
    'd_ff': 512,                 # Feed-forward dimension
    'dropout': 0.1,              # Dropout rate
    
    # Training parameters
    'batch_size': 32,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'epochs': 100,
    'early_stopping_patience': 15,
    'gradient_clip': 1.0,
    
    # Anomaly detection parameters
    'anomaly_threshold_percentile': 95,  # Use 95th percentile as threshold
    'reconstruction_weight': 1.0,
    
    # Validation split
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    
    # Logging
    'log_level': 'INFO',
    'checkpoint_frequency': 10,
    
    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Random seed for reproducibility
    'seed': 42
}

# Water quality parameter ranges for domain-specific validation
PARAMETER_RANGES = {
    'pH': {'min': 5.0, 'max': 10.0, 'critical_min': 7.5, 'critical_max': 8.0},
    'Turbidity_NTU': {'min': 0, 'max': 100, 'critical_min': 0, 'critical_max': 20},
    'TDS_ppm': {'min': 0, 'max': 3000, 'critical_min': 0, 'critical_max': 2100},
    'Total_Hardness_ppm': {'min': 0, 'max': 2000, 'critical_min': 0, 'critical_max': 1200},
    'Calcium_Hardness_ppm': {'min': 0, 'max': 1500, 'critical_min': 0, 'critical_max': 800},
    'Magnesium_Hardness_ppm': {'min': 0, 'max': 1000, 'critical_min': 0, 'critical_max': 400},
    'Chlorides_ppm': {'min': 0, 'max': 1000, 'critical_min': 0, 'critical_max': 500},
    'Phosphate_ppm': {'min': 0, 'max': 20, 'critical_min': 2, 'critical_max': 10},
    'Silica_ppm': {'min': 0, 'max': 300, 'critical_min': 0, 'critical_max': 180},
    'Conductivity_uS_cm': {'min': 0, 'max': 5000, 'critical_min': 0, 'critical_max': 3000}
}


# ==================== LOGGING SETUP ====================

def setup_logging(config: Dict) -> logging.Logger:
    """Configure logging for the system"""
    os.makedirs(config['output_dir'], exist_ok=True)
    
    log_file = os.path.join(
        config['output_dir'], 
        f"moment_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    
    logging.basicConfig(
        level=getattr(logging, config['log_level']),
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


# ==================== DATA PREPROCESSING ====================

class WaterQualityDataProcessor:
    """
    Comprehensive data processor for cooling tower water quality data.
    Handles missing values, normalization, and validation.
    """
    
    def __init__(self, config: Dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.scaler = RobustScaler()
        self.feature_columns = config['feature_columns']
        self.fitted = False
        
    def load_data(self, path: str) -> pd.DataFrame:
        """Load and perform initial data validation"""
        self.logger.info(f"Loading data from {path}")
        df = pd.read_csv(path)
        self.logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def validate_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply domain-specific range validation"""
        self.logger.info("Applying domain-specific range validation")
        df_validated = df.copy()
        
        for col in self.feature_columns:
            if col in df_validated.columns and col in PARAMETER_RANGES:
                ranges = PARAMETER_RANGES[col]
                original_count = df_validated[col].notna().sum()
                
                # Mark values outside physical ranges as NaN
                mask = (df_validated[col] < ranges['min']) | (df_validated[col] > ranges['max'])
                invalid_count = mask.sum()
                
                if invalid_count > 0:
                    df_validated.loc[mask, col] = np.nan
                    self.logger.warning(
                        f"{col}: {invalid_count} values outside [{ranges['min']}, {ranges['max']}] set to NaN"
                    )
        
        return df_validated
    
    def handle_missing_values(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Handle missing values by creating a mask.
        Returns data with interpolated values AND a mask indicating original NaN positions.
        This allows the model to learn which values are real vs imputed.
        """
        self.logger.info("Handling missing values")
        df_clean = df[self.feature_columns].copy()
        
        # Create mask: 1 where data exists, 0 where missing
        mask = (~df_clean.isna()).astype(float).values
        
        # Log missing value statistics
        missing_stats = df_clean.isna().sum()
        self.logger.info("Missing value counts:")
        for col, count in missing_stats.items():
            pct = 100 * count / len(df_clean)
            self.logger.info(f"  {col}: {count} ({pct:.1f}%)")
        
        # Forward fill, then backward fill, then fill with median
        df_filled = df_clean.ffill().bfill()
        
        # For any remaining NaN (e.g., columns that are entirely NaN), use column median
        for col in df_filled.columns:
            if df_filled[col].isna().any():
                median_val = df_clean[col].median()
                if pd.isna(median_val):
                    median_val = 0  # Fallback
                df_filled[col].fillna(median_val, inplace=True)
        
        return df_filled, mask
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Fit scaler and transform data"""
        df_validated = self.validate_ranges(df)
        df_filled, mask = self.handle_missing_values(df_validated)
        
        self.logger.info("Fitting RobustScaler")
        data_scaled = self.scaler.fit_transform(df_filled.values)
        self.fitted = True
        
        self.logger.info(f"Scaled data shape: {data_scaled.shape}")
        return data_scaled, mask
    
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Transform data using fitted scaler"""
        if not self.fitted:
            raise RuntimeError("Scaler not fitted. Call fit_transform first.")
        
        df_validated = self.validate_ranges(df)
        df_filled, mask = self.handle_missing_values(df_validated)
        data_scaled = self.scaler.transform(df_filled.values)
        
        return data_scaled, mask
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data"""
        return self.scaler.inverse_transform(data)


# ==================== DATASET ====================

class MOMENTDataset(Dataset):
    """
    Dataset for MOMENT model with sliding window approach.
    Includes mask for handling missing values during training.
    """
    
    def __init__(
        self, 
        data: np.ndarray, 
        mask: np.ndarray,
        window_size: int,
        stride: int,
        mode: str = 'train'
    ):
        self.data = torch.FloatTensor(data)
        self.mask = torch.FloatTensor(mask)
        self.window_size = window_size
        self.stride = stride
        self.mode = mode
        
        # Calculate number of windows
        self.n_windows = max(1, (len(data) - window_size) // stride + 1)
        
    def __len__(self) -> int:
        return self.n_windows
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        start_idx = idx * self.stride
        end_idx = start_idx + self.window_size
        
        # Handle edge case at the end
        if end_idx > len(self.data):
            start_idx = len(self.data) - self.window_size
            end_idx = len(self.data)
        
        window_data = self.data[start_idx:end_idx]
        window_mask = self.mask[start_idx:end_idx]
        
        return window_data, window_mask, start_idx


# ==================== MOMENT MODEL ARCHITECTURE ====================

class PatchEmbedding(nn.Module):
    """
    Patch embedding layer for time series.
    Converts time series into patches and projects them to model dimension.
    """
    
    def __init__(self, n_features: int, patch_size: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        self.n_features = n_features
        
        # Linear projection for each patch
        self.projection = nn.Linear(patch_size * n_features, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, n_features]
        Returns:
            patches: [batch, n_patches, d_model]
        """
        batch_size, seq_len, n_features = x.shape
        
        # Ensure seq_len is divisible by patch_size
        n_patches = seq_len // self.patch_size
        x = x[:, :n_patches * self.patch_size, :]
        
        # Reshape into patches: [batch, n_patches, patch_size * n_features]
        x = x.reshape(batch_size, n_patches, self.patch_size * n_features)
        
        # Project to d_model
        x = self.projection(x)
        x = self.norm(x)
        
        return x


class PositionalEncoding(nn.Module):
    """Learnable positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Learnable positional embeddings
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class MOMENTEncoder(nn.Module):
    """
    MOMENT-style Transformer Encoder.
    Uses standard transformer encoder with pre-norm architecture.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm architecture
        )
        
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: Optional attention mask
        """
        x = self.encoder(x, src_key_padding_mask=mask)
        x = self.norm(x)
        return x


class MOMENTDecoder(nn.Module):
    """
    MOMENT-style Transformer Decoder for reconstruction.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(
        self, 
        tgt: torch.Tensor, 
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            tgt: [batch, seq_len, d_model] - target sequence
            memory: [batch, seq_len, d_model] - encoder output
        """
        x = self.decoder(tgt, memory, tgt_key_padding_mask=tgt_mask)
        x = self.norm(x)
        return x


class PatchReconstruction(nn.Module):
    """
    Reconstruct original time series from patch embeddings.
    """
    
    def __init__(self, d_model: int, patch_size: int, n_features: int):
        super().__init__()
        self.patch_size = patch_size
        self.n_features = n_features
        
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, patch_size * n_features)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, n_patches, d_model]
        Returns:
            reconstruction: [batch, seq_len, n_features]
        """
        batch_size, n_patches, d_model = x.shape
        
        # Project back to patch space
        x = self.projection(x)  # [batch, n_patches, patch_size * n_features]
        
        # Reshape to time series
        x = x.reshape(batch_size, n_patches * self.patch_size, self.n_features)
        
        return x


class MOMENT(nn.Module):
    """
    MOMENT: Foundation Model for Time Series Anomaly Detection
    
    Architecture:
    1. Patch Embedding: Convert time series into patch tokens
    2. Positional Encoding: Add positional information
    3. Transformer Encoder: Learn temporal representations
    4. Transformer Decoder: Reconstruct the input
    5. Patch Reconstruction: Convert patches back to time series
    
    Anomaly detection via reconstruction error.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        n_features = len(config['feature_columns'])
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            n_features=n_features,
            patch_size=config['patch_size'],
            d_model=config['d_model']
        )
        
        # Positional encoding
        max_patches = config['window_size'] // config['patch_size']
        self.pos_encoding = PositionalEncoding(
            d_model=config['d_model'],
            max_len=max_patches + 1,
            dropout=config['dropout']
        )
        
        # Encoder
        self.encoder = MOMENTEncoder(
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_encoder_layers'],
            d_ff=config['d_ff'],
            dropout=config['dropout']
        )
        
        # Decoder
        self.decoder = MOMENTDecoder(
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_decoder_layers'],
            d_ff=config['d_ff'],
            dropout=config['dropout']
        )
        
        # Reconstruction head
        self.reconstruction = PatchReconstruction(
            d_model=config['d_model'],
            patch_size=config['patch_size'],
            n_features=n_features
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, n_features] - input time series
            mask: [batch, seq_len, n_features] - mask for valid values (1=valid, 0=missing)
        
        Returns:
            reconstruction: [batch, seq_len, n_features]
            latent: [batch, n_patches, d_model] - encoded representation
        """
        # Store original length for output matching
        original_len = x.size(1)
        n_features = x.size(2)
        
        # Patch embedding
        patches = self.patch_embed(x)  # [batch, n_patches, d_model]
        
        # Add positional encoding
        patches = self.pos_encoding(patches)
        
        # Encode
        encoded = self.encoder(patches)  # [batch, n_patches, d_model]
        
        # Decode (using encoded as both memory and target for autoencoder)
        decoded = self.decoder(patches, encoded)  # [batch, n_patches, d_model]
        
        # Reconstruct
        reconstruction = self.reconstruction(decoded)  # [batch, recon_len, n_features]
        
        # Pad or truncate to match original length
        recon_len = reconstruction.size(1)
        if recon_len < original_len:
            # Pad with last values
            padding = reconstruction[:, -1:, :].expand(-1, original_len - recon_len, -1)
            reconstruction = torch.cat([reconstruction, padding], dim=1)
        elif recon_len > original_len:
            reconstruction = reconstruction[:, :original_len, :]
        
        return reconstruction, encoded
    
    def compute_anomaly_score(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute reconstruction-based anomaly score.
        
        Args:
            x: [batch, seq_len, n_features]
            mask: [batch, seq_len, n_features]
        
        Returns:
            scores: [batch, seq_len] - per-timestep anomaly scores
            feature_scores: [batch, seq_len, n_features] - per-feature scores
        """
        reconstruction, _ = self.forward(x, mask)
        
        # Compute per-feature reconstruction error
        feature_scores = (x - reconstruction) ** 2
        
        # Apply mask if provided (only score valid values)
        if mask is not None:
            feature_scores = feature_scores * mask
        
        # Aggregate across features for per-timestep score
        if mask is not None:
            # Weighted mean (only valid features)
            mask_sum = mask.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            scores = (feature_scores.sum(dim=-1) / mask_sum.squeeze(-1))
        else:
            scores = feature_scores.mean(dim=-1)
        
        return scores, feature_scores


# ==================== TRAINING ====================

class MOMENTTrainer:
    """
    Trainer for MOMENT model with comprehensive logging and checkpointing.
    """
    
    def __init__(self, model: MOMENT, config: Dict, logger: logging.Logger):
        self.model = model
        self.config = config
        self.logger = logger
        self.device = torch.device(config['device'])
        
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def masked_mse_loss(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MSE loss only on valid (non-missing) values.
        """
        # Compute squared error
        squared_error = (pred - target) ** 2
        
        # Apply mask
        masked_error = squared_error * mask
        
        # Mean over valid values
        loss = masked_error.sum() / mask.sum().clamp(min=1e-8)
        
        return loss
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch_data, batch_mask, _ in dataloader:
            batch_data = batch_data.to(self.device)
            batch_mask = batch_mask.to(self.device)
            
            # Forward pass
            reconstruction, _ = self.model(batch_data, batch_mask)
            
            # Compute masked loss
            loss = self.masked_mse_loss(reconstruction, batch_data, batch_mask)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['gradient_clip']
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        for batch_data, batch_mask, _ in dataloader:
            batch_data = batch_data.to(self.device)
            batch_mask = batch_mask.to(self.device)
            
            reconstruction, _ = self.model(batch_data, batch_mask)
            loss = self.masked_mse_loss(reconstruction, batch_data, batch_mask)
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    def save_checkpoint(self, path: str, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = path.replace('.pt', '_best.pt')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model to {best_path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']
        
        self.logger.info(f"Loaded checkpoint from {path}")
        return checkpoint['epoch']
    
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader
    ) -> Dict[str, List[float]]:
        """
        Full training loop with early stopping.
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting MOMENT Training")
        self.logger.info("=" * 60)
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Train batches: {len(train_loader)}")
        self.logger.info(f"Val batches: {len(val_loader)}")
        
        checkpoint_dir = os.path.join(self.config['output_dir'], 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for epoch in range(self.config['epochs']):
            epoch_start = datetime.now()
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            
            # Check for improvement
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Logging
            epoch_time = (datetime.now() - epoch_start).total_seconds()
            self.logger.info(
                f"Epoch {epoch+1:3d}/{self.config['epochs']} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time:.1f}s | "
                f"{'*BEST*' if is_best else ''}"
            )
            
            # Checkpointing
            if (epoch + 1) % self.config['checkpoint_frequency'] == 0:
                checkpoint_path = os.path.join(
                    checkpoint_dir, 
                    f"moment_epoch_{epoch+1}.pt"
                )
                self.save_checkpoint(checkpoint_path, epoch, is_best)
            
            # Early stopping
            if self.patience_counter >= self.config['early_stopping_patience']:
                self.logger.info(
                    f"Early stopping triggered after {epoch+1} epochs"
                )
                break
        
        # Save final checkpoint
        final_path = os.path.join(checkpoint_dir, "moment_final.pt")
        self.save_checkpoint(final_path, epoch, is_best=False)
        
        self.logger.info("=" * 60)
        self.logger.info("Training Complete")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
        self.logger.info("=" * 60)
        
        return self.history


# ==================== ANOMALY DETECTOR ====================

class MOMENTAnomalyDetector:
    """
    Complete anomaly detection system using MOMENT.
    """
    
    def __init__(self, model: MOMENT, config: Dict, logger: logging.Logger):
        self.model = model
        self.config = config
        self.logger = logger
        self.device = torch.device(config['device'])
        self.model.to(self.device)
        self.model.eval()
        
        self.threshold = None
        
    def fit_threshold(self, dataloader: DataLoader):
        """
        Fit anomaly threshold based on training data reconstruction errors.
        Uses the specified percentile as threshold.
        """
        self.logger.info("Fitting anomaly threshold...")
        
        all_scores = []
        
        with torch.no_grad():
            for batch_data, batch_mask, _ in dataloader:
                batch_data = batch_data.to(self.device)
                batch_mask = batch_mask.to(self.device)
                
                scores, _ = self.model.compute_anomaly_score(batch_data, batch_mask)
                all_scores.append(scores.cpu().numpy())
        
        all_scores = np.concatenate(all_scores).flatten()
        
        # Use specified percentile as threshold
        self.threshold = np.percentile(
            all_scores, 
            self.config['anomaly_threshold_percentile']
        )
        
        self.logger.info(f"Anomaly threshold set to {self.threshold:.6f}")
        self.logger.info(f"Score statistics: mean={all_scores.mean():.6f}, "
                        f"std={all_scores.std():.6f}, "
                        f"min={all_scores.min():.6f}, "
                        f"max={all_scores.max():.6f}")
        
        return self.threshold
    
    @torch.no_grad()
    def detect(
        self, 
        dataloader: DataLoader
    ) -> Dict[str, Any]:
        """
        Detect anomalies in the provided data.
        
        Returns:
            Dictionary containing:
            - anomaly_scores: Per-timestep anomaly scores
            - feature_scores: Per-feature anomaly scores  
            - predictions: Binary anomaly predictions
            - reconstructions: Reconstructed values
            - originals: Original values
            - indices: Original data indices
        """
        if self.threshold is None:
            raise RuntimeError("Threshold not fitted. Call fit_threshold first.")
        
        self.logger.info("Running anomaly detection...")
        
        results = {
            'anomaly_scores': [],
            'feature_scores': [],
            'predictions': [],
            'reconstructions': [],
            'originals': [],
            'indices': []
        }
        
        for batch_data, batch_mask, batch_indices in dataloader:
            batch_data = batch_data.to(self.device)
            batch_mask = batch_mask.to(self.device)
            
            # Get reconstruction and scores
            reconstruction, _ = self.model(batch_data, batch_mask)
            scores, feature_scores = self.model.compute_anomaly_score(batch_data, batch_mask)
            
            # Make predictions
            predictions = (scores > self.threshold).float()
            
            # Store results
            results['anomaly_scores'].append(scores.cpu().numpy())
            results['feature_scores'].append(feature_scores.cpu().numpy())
            results['predictions'].append(predictions.cpu().numpy())
            results['reconstructions'].append(reconstruction.cpu().numpy())
            results['originals'].append(batch_data.cpu().numpy())
            results['indices'].extend(batch_indices.tolist())
        
        # Concatenate results
        for key in ['anomaly_scores', 'feature_scores', 'predictions', 
                    'reconstructions', 'originals']:
            results[key] = np.concatenate(results[key], axis=0)
        
        # Summary statistics
        n_anomalies = results['predictions'].sum()
        anomaly_rate = n_anomalies / results['predictions'].size
        
        self.logger.info(f"Detection complete: {int(n_anomalies)} anomalies detected "
                        f"({100*anomaly_rate:.2f}% of timesteps)")
        
        return results
    
    def get_top_anomalies(
        self, 
        results: Dict[str, Any], 
        top_k: int = 10
    ) -> List[Dict]:
        """
        Get the top-k most anomalous windows with interpretation.
        """
        # Aggregate scores per window
        window_scores = results['anomaly_scores'].mean(axis=1)  # [n_windows]
        
        # Get top-k indices
        top_indices = np.argsort(window_scores)[-top_k:][::-1]
        
        feature_names = self.config['feature_columns']
        top_anomalies = []
        
        for idx in top_indices:
            # Get feature-level scores for this window
            window_feature_scores = results['feature_scores'][idx].mean(axis=0)
            
            # Rank features by contribution
            feature_ranking = np.argsort(window_feature_scores)[::-1]
            
            anomaly_info = {
                'window_index': int(idx),
                'data_index': results['indices'][idx],
                'anomaly_score': float(window_scores[idx]),
                'contributing_features': [
                    {
                        'feature': feature_names[f_idx],
                        'score': float(window_feature_scores[f_idx])
                    }
                    for f_idx in feature_ranking[:5]  # Top 5 features
                ]
            }
            top_anomalies.append(anomaly_info)
        
        return top_anomalies


# ==================== MAIN PIPELINE ====================

class MOMENTPipeline:
    """
    Complete pipeline for MOMENT-based anomaly detection.
    Handles data loading, preprocessing, training, and detection.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = setup_logging(config)
        
        # Set random seeds
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        
        # Initialize components
        self.data_processor = WaterQualityDataProcessor(config, self.logger)
        self.model = None
        self.trainer = None
        self.detector = None
        
    def prepare_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load and prepare data for training"""
        # Load data
        df = self.data_processor.load_data(self.config['data_path'])
        
        # Split data chronologically
        n_samples = len(df)
        train_end = int(n_samples * self.config['train_ratio'])
        val_end = train_end + int(n_samples * self.config['val_ratio'])
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        self.logger.info(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        # Fit scaler on training data and transform all sets
        train_data, train_mask = self.data_processor.fit_transform(train_df)
        val_data, val_mask = self.data_processor.transform(val_df)
        test_data, test_mask = self.data_processor.transform(test_df)
        
        # Create datasets
        train_dataset = MOMENTDataset(
            train_data, train_mask,
            self.config['window_size'],
            self.config['stride'],
            mode='train'
        )
        
        val_dataset = MOMENTDataset(
            val_data, val_mask,
            self.config['window_size'],
            self.config['stride'],
            mode='val'
        )
        
        test_dataset = MOMENTDataset(
            test_data, test_mask,
            self.config['window_size'],
            self.config['stride'],
            mode='test'
        )
        
        self.logger.info(f"Datasets: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        return train_loader, val_loader, test_loader
    
    def build_model(self) -> MOMENT:
        """Initialize the MOMENT model"""
        self.logger.info("Building MOMENT model...")
        
        self.model = MOMENT(self.config)
        
        # Log model statistics
        n_params = sum(p.numel() for p in self.model.parameters())
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model parameters: {n_params:,} total, {n_trainable:,} trainable")
        
        return self.model
    
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader
    ) -> Dict[str, List[float]]:
        """Train the model"""
        if self.model is None:
            self.build_model()
        
        self.trainer = MOMENTTrainer(self.model, self.config, self.logger)
        history = self.trainer.train(train_loader, val_loader)
        
        return history
    
    def setup_detector(self, train_loader: DataLoader):
        """Initialize and fit the anomaly detector"""
        self.detector = MOMENTAnomalyDetector(self.model, self.config, self.logger)
        self.detector.fit_threshold(train_loader)
    
    def detect_anomalies(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Run anomaly detection"""
        if self.detector is None:
            raise RuntimeError("Detector not initialized. Call setup_detector first.")
        
        return self.detector.detect(test_loader)
    
    def run(self) -> Dict[str, Any]:
        """
        Execute the complete pipeline:
        1. Load and prepare data
        2. Build model
        3. Train
        4. Detect anomalies
        5. Generate report
        """
        self.logger.info("=" * 60)
        self.logger.info("MOMENT Anomaly Detection Pipeline")
        self.logger.info("=" * 60)
        
        # Prepare data
        train_loader, val_loader, test_loader = self.prepare_data()
        
        # Build and train model
        self.build_model()
        history = self.train(train_loader, val_loader)
        
        # Setup detector
        self.setup_detector(train_loader)
        
        # Detect anomalies
        results = self.detect_anomalies(test_loader)
        
        # Get top anomalies
        top_anomalies = self.detector.get_top_anomalies(results, top_k=10)
        
        # Generate report
        report = self.generate_report(results, top_anomalies, history)
        
        # Save report
        report_path = os.path.join(self.config['output_dir'], 'anomaly_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        self.logger.info(f"Report saved to {report_path}")
        
        return {
            'results': results,
            'top_anomalies': top_anomalies,
            'history': history,
            'report': report
        }
    
    def generate_report(
        self, 
        results: Dict[str, Any], 
        top_anomalies: List[Dict],
        history: Dict[str, List[float]]
    ) -> Dict:
        """Generate a comprehensive report"""
        # Calculate statistics
        scores = results['anomaly_scores'].flatten()
        predictions = results['predictions'].flatten()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': {k: v for k, v in self.config.items() if not callable(v)},
            'data_statistics': {
                'total_windows': len(results['anomaly_scores']),
                'total_timesteps': predictions.size,
                'anomaly_count': int(predictions.sum()),
                'anomaly_rate': float(predictions.mean())
            },
            'score_statistics': {
                'mean': float(scores.mean()),
                'std': float(scores.std()),
                'min': float(scores.min()),
                'max': float(scores.max()),
                'threshold': float(self.detector.threshold)
            },
            'training_statistics': {
                'final_train_loss': history['train_loss'][-1] if history['train_loss'] else None,
                'final_val_loss': history['val_loss'][-1] if history['val_loss'] else None,
                'best_val_loss': min(history['val_loss']) if history['val_loss'] else None,
                'epochs_trained': len(history['train_loss'])
            },
            'top_anomalies': top_anomalies
        }
        
        return report


# ==================== MAIN ====================

def main():
    """Main entry point"""
    print("=" * 60)
    print("MOMENT Anomaly Detection System for TGF MVP")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Initialize and run pipeline
    pipeline = MOMENTPipeline(CONFIG)
    results = pipeline.run()
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Total anomalies detected: {results['report']['data_statistics']['anomaly_count']}")
    print(f"Anomaly rate: {100*results['report']['data_statistics']['anomaly_rate']:.2f}%")
    print(f"Best validation loss: {results['report']['training_statistics']['best_val_loss']:.6f}")
    print(f"\nTop 5 most anomalous windows:")
    for i, anomaly in enumerate(results['top_anomalies'][:5]):
        print(f"  {i+1}. Window {anomaly['window_index']} (score: {anomaly['anomaly_score']:.4f})")
        print(f"     Top feature: {anomaly['contributing_features'][0]['feature']}")
    
    print(f"\nResults saved to: {CONFIG['output_dir']}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    results = main()
