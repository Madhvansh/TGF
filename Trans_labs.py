import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import optuna
from optuna.samplers import NSGAIISampler
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
from scipy import stats
import pickle
import os
warnings.filterwarnings('ignore')

# ==================== Water Quality Parameter Ranges ====================

class WaterQualityRanges:
    """Acceptable ranges for cooling tower water parameters"""
    PARAMETER_RANGES = {
        'pH': {'min': 7.5, 'max': 8.0, 'critical': True},
        'Turbidity': {'min': 0, 'max': 20, 'critical': False},
        'TSS': {'min': 0, 'max': 20, 'critical': False},
        'FRC': {'min': 0.2, 'max': 0.5, 'critical': True},
        'Conductivity': {'min': 0, 'max': 3000, 'critical': True},
        'TDS': {'min': 0, 'max': 2100, 'critical': True},
        'Total Hardness': {'min': 0, 'max': 1200, 'critical': True},
        'Calcium Hardness': {'min': 0, 'max': 800, 'critical': True},
        'Magnesium Hardness': {'min': 0, 'max': 400, 'critical': False},
        'Chlorides': {'min': 0, 'max': 500, 'critical': True},
        'Ortho PO4': {'min': 6.0, 'max': 8.0, 'critical': True},
        'Total Alkalinity': {'min': 0, 'max': 200, 'critical': True},
        'P Alkalinity': {'min': 0, 'max': 0, 'critical': False},
        'Total Iron': {'min': 0, 'max': 2, 'critical': True},
        'SS': {'min': 0, 'max': 50, 'critical': False},
        'Sulphate': {'min': 0, 'max': 1000, 'critical': True},
        'Silica': {'min': 0, 'max': 180, 'critical': True}
    }

    @classmethod
    def check_parameter(cls, param_name: str, value: float) -> Dict:
        """Check if parameter is within acceptable range"""
        if param_name not in cls.PARAMETER_RANGES:
            return {'status': 'unknown', 'severity': 0}
        
        ranges = cls.PARAMETER_RANGES[param_name]
        if pd.isna(value):
            return {'status': 'missing', 'severity': 0}
        
        # Calculate deviation
        if value < ranges['min']:
            deviation = (ranges['min'] - value) / (ranges['min'] + 1e-8)
            status = 'low'
        elif value > ranges['max']:
            deviation = (value - ranges['max']) / (ranges['max'] + 1e-8)
            status = 'high'
        else:
            deviation = 0
            status = 'normal'
        
        # Determine severity
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
            'max': ranges['max']
        }

# ==================== Enhanced Dataset for Water Quality ====================

class CoolingTowerDataset(Dataset):
    """Dataset specifically for cooling tower water quality monitoring"""
    
    def __init__(self, data: pd.DataFrame, window_size: int = 20,
                 stride: int = 1, mode: str = 'train', use_ranges: bool = True,
                 scaler: Optional[StandardScaler] = None):
        """
        Initialize dataset with range-based anomaly labels
        """
        self.data = data.copy()
        self.window_size = window_size
        self.stride = stride
        self.mode = mode
        self.use_ranges = use_ranges
        self.scaler = scaler
        
        # Map column names to standardized names
        self.column_mapping = {
            'pH': 'pH', 'Turbidity': 'Turbidity', 'TSS': 'TSS', 'FRC': 'FRC',
            'Conductivity': 'Conductivity', 'COND': 'Conductivity', 'TDS': 'TDS',
            'Total Hardness': 'Total Hardness', 'TH': 'Total Hardness',
            'Calcium Hardness': 'Calcium Hardness', 'CaH': 'Calcium Hardness',
            'Magnesium Hardness': 'Magnesium Hardness', 'MgH': 'Magnesium Hardness',
            'Chlorides': 'Chlorides', 'Cl': 'Chlorides',
            'Ortho PO4': 'Ortho PO4', 'ORTHO PO4': 'Ortho PO4',
            'Total Alkalinity': 'Total Alkalinity', 'T. Alk.': 'Total Alkalinity',
            'P Alkalinity': 'P Alkalinity', 'P. Alk.': 'P Alkalinity',
            'Total Iron': 'Total Iron', 'SS': 'SS', 'Sulphate': 'Sulphate',
            'Silica': 'Silica', 'SiO2': 'Silica'
        }
        
        # Standardize column names
        self._standardize_columns()
        
        # Get available parameters
        self.parameters = [col for col in WaterQualityRanges.PARAMETER_RANGES.keys()
                          if col in self.data.columns]
        
        # Clean and prepare data
        self._prepare_data()
        
        # Generate anomaly labels based on ranges
        if self.use_ranges:
            self.anomaly_labels = self._generate_anomaly_labels()
        
        # Create sliding windows
        self.windows, self.window_labels = self._create_windows()
    
    def _standardize_columns(self):
        """Standardize column names"""
        for old_name, new_name in self.column_mapping.items():
            if old_name in self.data.columns:
                self.data.rename(columns={old_name: new_name}, inplace=True)
    
    def _prepare_data(self):
        """Clean and prepare data"""
        # Handle numeric conversion
        for param in self.parameters:
            if param in self.data.columns:
                self.data[param] = pd.to_numeric(self.data[param], errors='coerce')
        
        # Fill missing values with interpolation for time series continuity
        self.data[self.parameters] = self.data[self.parameters].interpolate(method='linear', limit=3)
        self.data[self.parameters] = self.data[self.parameters].bfill()
        self.data[self.parameters] = self.data[self.parameters].ffill()
        
        # Final fill with median for any remaining NaN
        for param in self.parameters:
            if self.data[param].isna().any():
                self.data[param].fillna(self.data[param].median(), inplace=True)
    
    def _generate_anomaly_labels(self) -> np.ndarray:
        """Generate anomaly labels based on parameter ranges"""
        anomaly_scores = np.zeros(len(self.data))
        parameter_status = {}
        
        for i, (idx, row) in enumerate(self.data.iterrows()):
            max_severity = 0
            anomalous_params = []
            
            for param in self.parameters:
                if param in row:
                    check_result = WaterQualityRanges.check_parameter(param, row[param])
                    if check_result['severity'] > 0:
                        anomalous_params.append((param, check_result))
                        max_severity = max(max_severity, check_result['severity'])
            
            anomaly_scores[i] = max_severity
            parameter_status[i] = anomalous_params
        
        self.parameter_status = parameter_status
        return anomaly_scores
    
    def _create_windows(self):
        """Create sliding windows with anomaly labels"""
        # Normalize data
        if self.mode == 'train':
            if self.scaler is None:
                self.scaler = StandardScaler()
            normalized_data = self.scaler.fit_transform(self.data[self.parameters].values)
        else:
            if self.scaler is None:
                raise ValueError("Scaler must be provided for test/validation mode")
            normalized_data = self.scaler.transform(self.data[self.parameters].values)
        
        windows = []
        labels = []
        
        for i in range(0, len(normalized_data) - self.window_size + 1, self.stride):
            window = normalized_data[i:i + self.window_size]
            windows.append(window)
            
            if self.use_ranges:
                # Label is 1 if any point in window has severity > 1
                window_labels = self.anomaly_labels[i:i + self.window_size]
                labels.append(1 if np.any(window_labels > 1) else 0)
            else:
                labels.append(0)
        
        return np.array(windows), np.array(labels)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.windows[idx]), torch.FloatTensor([self.window_labels[idx]])

# ==================== Enhanced POT and Thresholding Methods ====================

class EnhancedPOTThresholding:
    """Enhanced POT method with mPOT, MAT, and rolling statistics"""
    
    def __init__(self, quantile: float = 0.95, alpha: float = 0.1):
        self.quantile = quantile
        self.alpha = alpha
        self.threshold = None
        self.rolling_stats = {'mean': None, 'std': None}
        
    def fit_pot_threshold(self, scores: np.ndarray):
        """Fit POT threshold on training scores"""
        self.threshold = np.percentile(scores, self.quantile * 100)
        return self.threshold
    
    def modified_pot(self, scores: np.ndarray, recent_window: int = 50) -> float:
        """Modified POT with recent deviation (mPOT)"""
        if len(scores) < recent_window:
            return self.threshold
        
        recent_scores = scores[-recent_window:]
        recent_median = np.median(recent_scores)
        recent_deviation = np.abs(recent_scores - recent_median).mean()
        
        # mPOT formula from paper
        modified_threshold = self.threshold + self.alpha * recent_deviation
        return modified_threshold
    
    def moving_average_threshold(self, scores: np.ndarray, window_size: int = 30) -> np.ndarray:
        """Moving Average Thresholding (MAT)"""
        if len(scores) < window_size:
            return np.full(len(scores), self.threshold)
        
        mat_thresholds = np.zeros(len(scores))
        for i in range(len(scores)):
            start_idx = max(0, i - window_size + 1)
            mat_thresholds[i] = np.mean(scores[start_idx:i+1])
        
        return mat_thresholds
    
    def compute_rolling_statistics(self, data: np.ndarray, window_size: int = 50):
        """Compute rolling statistics for nuanced detection"""
        rolling_mean = pd.Series(data).rolling(window=window_size, min_periods=1).mean().values
        rolling_std = pd.Series(data).rolling(window=window_size, min_periods=1).std().values
        
        self.rolling_stats = {
            'mean': rolling_mean,
            'std': rolling_std
        }
        
        return rolling_mean, rolling_std
    
    def detect_anomalies(self, scores: np.ndarray, use_enhanced: bool = True) -> Dict:
        """Detect anomalies using enhanced POT methods"""
        results = {}
        
        # Basic POT
        basic_anomalies = scores > self.threshold
        results['basic_pot'] = basic_anomalies
        
        if use_enhanced:
            # Modified POT (mPOT)
            mpot_threshold = self.modified_pot(scores)
            mpot_anomalies = scores > mpot_threshold
            results['mpot'] = mpot_anomalies
            results['mpot_threshold'] = mpot_threshold
            
            # Moving Average Thresholding (MAT)
            mat_thresholds = self.moving_average_threshold(scores)
            mat_anomalies = scores > mat_thresholds
            results['mat'] = mat_anomalies
            results['mat_thresholds'] = mat_thresholds
            
            # Combined enhanced detection
            enhanced_anomalies = mpot_anomalies | mat_anomalies
            results['enhanced'] = enhanced_anomalies
        
        return results

# ==================== TransNAS-TSAD Transformer Architecture ====================

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 1000, encoding_type: str = 'sinusoidal'):
        super().__init__()
        self.encoding_type = encoding_type
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        if encoding_type == 'sinusoidal':
            div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                               -(np.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
        elif encoding_type == 'fourier':
            # Fourier-based positional encoding
            frequencies = torch.linspace(0.1, 100, d_model//2)
            for i in range(d_model//2):
                pe[:, 2*i] = torch.sin(2 * np.pi * frequencies[i] * position.squeeze() / max_len)
                pe[:, 2*i+1] = torch.cos(2 * np.pi * frequencies[i] * position.squeeze() / max_len)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerBlock(nn.Module):
    """Transformer block with multi-head attention and feedforward"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1, activation: str = 'relu',
                 layer_norm_type: str = 'layer'):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # Layer normalization
        if layer_norm_type == 'layer':
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        elif layer_norm_type == 'batch':
            self.norm1 = nn.BatchNorm1d(d_model)
            self.norm2 = nn.BatchNorm1d(d_model)
        elif layer_norm_type == 'instance':
            self.norm1 = nn.InstanceNorm1d(d_model)
            self.norm2 = nn.InstanceNorm1d(d_model)
        
        self.layer_norm_type = layer_norm_type
        
        # Activation functions
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh()
        }
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            activations.get(activation, nn.ReLU()),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Self-attention
        attn_output, _ = self.attention(x, x, x)
        
        # Apply normalization
        if self.layer_norm_type in ['batch', 'instance']:
            # For batch/instance norm, need to transpose
            x_norm = x.transpose(1, 2)
            attn_norm = self.norm1(x_norm + self.dropout(attn_output).transpose(1, 2))
            x = attn_norm.transpose(1, 2)
        else:
            x = self.norm1(x + self.dropout(attn_output))
        
        # Feedforward
        ff_output = self.ff(x)
        
        if self.layer_norm_type in ['batch', 'instance']:
            x_norm = x.transpose(1, 2)
            ff_norm = self.norm2(x_norm + self.dropout(ff_output).transpose(1, 2))
            x = ff_norm.transpose(1, 2)
        else:
            x = self.norm2(x + self.dropout(ff_output))
        
        return x

class TransNAS_TSAD_Model(nn.Module):
    """Complete TransNAS-TSAD model with three-phase adversarial approach"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Input dimensions
        self.input_dim = config['input_dim']
        self.window_size = config['window_size']
        self.d_model = config['dim_feedforward']
        self.phase_type = config.get('phase_type', '2phase')
        self.use_linear_embedding = config.get('use_linear_embedding', False)
        
        # Embedding layers
        if self.use_linear_embedding:
            self.embedding = nn.Linear(self.input_dim, self.d_model)
        else:
            self.embedding = nn.Identity()
            self.d_model = self.input_dim
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            self.d_model, 
            self.window_size, 
            config.get('pos_encoding_type', 'sinusoidal')
        )
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(
                d_model=self.d_model,
                n_heads=config.get('num_attention_heads', self.input_dim),
                d_ff=config['dim_feedforward'],
                dropout=config['dropout_rate'],
                activation=config.get('activation_function', 'relu'),
                layer_norm_type=config.get('layer_normalization', 'layer')
            ) for _ in range(config['encoder_layers'])
        ])
        
        # Decoder layers (for all phases)
        self.decoder1_layers = nn.ModuleList([
            TransformerBlock(
                d_model=self.d_model,
                n_heads=config.get('num_attention_heads', self.input_dim),
                d_ff=config['dim_feedforward'],
                dropout=config['dropout_rate'],
                activation=config.get('activation_function', 'relu'),
                layer_norm_type=config.get('layer_normalization', 'layer')
            ) for _ in range(config['decoder_layers'])
        ])
        
        # Second decoder for adversarial training
        self.decoder2_layers = nn.ModuleList([
            TransformerBlock(
                d_model=self.d_model,
                n_heads=config.get('num_attention_heads', self.input_dim),
                d_ff=config['dim_feedforward'],
                dropout=config['dropout_rate'],
                activation=config.get('activation_function', 'relu'),
                layer_norm_type=config.get('layer_normalization', 'layer')
            ) for _ in range(config['decoder_layers'])
        ])
        
        # Output reconstruction layers
        self.reconstruction_output1 = nn.Linear(self.d_model, self.input_dim)
        self.reconstruction_output2 = nn.Linear(self.d_model, self.input_dim)
        
        # Classification head for anomaly detection
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model * self.window_size, 256),
            nn.ReLU(),
            nn.Dropout(config['dropout_rate']),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(config['dropout_rate']),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Data augmentation parameters
        self.gaussian_noise = config.get('gaussian_noise', 0.0)
        self.time_warping = config.get('time_warping', False)
        self.time_masking = config.get('time_masking', False)
    
    def apply_data_augmentation(self, x, training=True):
        """Apply data augmentation during training"""
        if not training:
            return x
        
        # Gaussian noise augmentation
        if self.gaussian_noise > 0:
            noise = torch.randn_like(x) * self.gaussian_noise
            x = x + noise
        
        # Time warping (simple implementation)
        if self.time_warping and torch.rand(1) > 0.5:
            # Randomly warp the time dimension slightly
            stretch_factor = torch.uniform(0.8, 1.2, (1,))
            # Simple time warping by interpolation
            x = torch.nn.functional.interpolate(
                x.transpose(1, 2), 
                scale_factor=stretch_factor.item(), 
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
            
            # Crop or pad to original size
            if x.size(1) > self.window_size:
                x = x[:, :self.window_size, :]
            elif x.size(1) < self.window_size:
                padding = self.window_size - x.size(1)
                x = torch.nn.functional.pad(x, (0, 0, 0, padding))
        
        # Time masking
        if self.time_masking and torch.rand(1) > 0.5:
            mask_length = int(self.window_size * 0.1)  # Mask 10% of time steps
            mask_start = torch.randint(0, self.window_size - mask_length, (1,))
            x[:, mask_start:mask_start + mask_length, :] = 0
        
        return x
    
    def forward_phase1(self, x):
        """Phase 1: Preliminary Input Reconstruction"""
        # Apply data augmentation
        x = self.apply_data_augmentation(x, self.training)
        
        # Embedding
        if self.use_linear_embedding:
            x_embedded = self.embedding(x)
        else:
            x_embedded = x
        
        # Positional encoding
        x_embedded = self.pos_encoding(x_embedded)
        
        # Encoder
        encoder_output = x_embedded
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output)
        
        # Decoder 1
        decoder1_output = encoder_output
        for layer in self.decoder1_layers:
            decoder1_output = layer(decoder1_output)
        
        # Reconstruction
        reconstruction1 = self.reconstruction_output1(decoder1_output)
        
        # Focus score (reconstruction error)
        focus_score = torch.mean((reconstruction1 - x) ** 2, dim=(1, 2))
        
        return reconstruction1, focus_score, encoder_output
    
    def forward_phase2(self, x, encoder_output, focus_score):
        """Phase 2: Adversarial, Focus-Driven Reconstruction"""
        # Decoder 2 (adversarial)
        decoder2_output = encoder_output
        
        # Apply focus score as attention modulation
        focus_weights = focus_score.unsqueeze(1).unsqueeze(2).expand_as(decoder2_output)
        decoder2_output = decoder2_output * (1 + focus_weights)
        
        for layer in self.decoder2_layers:
            decoder2_output = layer(decoder2_output)
        
        # Reconstruction 2
        reconstruction2 = self.reconstruction_output2(decoder2_output)
        
        return reconstruction2
    
    def forward_iterative(self, x, encoder_output, max_iterations=5, convergence_threshold=1e-5):
        """Phase 3: Iterative Self-Adversarial Approach"""
        current_reconstruction = encoder_output
        best_reconstruction = None
        best_loss = float('inf')
        prev_loss = float('inf')
        
        for iteration in range(max_iterations):
            # Apply decoder layers
            decoder_output = current_reconstruction
            for layer in self.decoder1_layers:
                decoder_output = layer(decoder_output)
            
            reconstruction = self.reconstruction_output1(decoder_output)
            
            # Calculate iteration loss
            iteration_loss = torch.mean((reconstruction - x) ** 2)
            
            # Self-adversarial mechanism
            if iteration > 0:
                self_adv_loss = (prev_loss - iteration_loss) ** 2
                total_loss = iteration_loss + 0.1 * self_adv_loss
            else:
                total_loss = iteration_loss
            
            # Check for best reconstruction
            if iteration_loss < best_loss:
                best_loss = iteration_loss
                best_reconstruction = reconstruction
            
            # Check convergence
            if iteration > 0:
                delta_loss = abs(iteration_loss - prev_loss)
                if delta_loss < convergence_threshold:
                    break
            
            prev_loss = iteration_loss
            
            # Update current reconstruction for next iteration
            current_reconstruction = decoder_output.detach()
        
        return best_reconstruction
    
    def forward(self, x, return_all_outputs=False):
        """Forward pass with three-phase approach"""
        # Phase 1: Preliminary reconstruction
        reconstruction1, focus_score, encoder_output = self.forward_phase1(x)
        
        # Phase 2 or Iterative phase based on configuration
        if self.phase_type == 'iterative':
            # Phase 3: Iterative self-adversarial
            final_reconstruction = self.forward_iterative(x, encoder_output)
            reconstruction2 = final_reconstruction
        else:
            # Phase 2: Adversarial reconstruction
            reconstruction2 = self.forward_phase2(x, encoder_output, focus_score)
        
        # Classification pathway
        classifier_input = encoder_output.reshape(encoder_output.size(0), -1)
        anomaly_score = self.classifier(classifier_input)
        
        if return_all_outputs:
            return reconstruction1, reconstruction2, anomaly_score, focus_score
        
        return reconstruction1, reconstruction2, anomaly_score

# ==================== NSGA-II Optimization Framework ====================

class TransNAS_TSAD_Optimizer:
    """NSGA-II based neural architecture search for TransNAS-TSAD"""
    
    def __init__(self, train_dataset, val_dataset, n_trials=100):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.n_trials = n_trials
        self.best_models = []
        self.pareto_front = []
        
    def get_search_space(self, trial):
        """Define search space according to Table 1 from paper"""
        config = {
            # Training hyperparameters
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
            'dropout_rate': trial.suggest_uniform('dropout_rate', 0.1, 0.5),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 48, 64, 80, 96, 112, 128]),
            'gaussian_noise': trial.suggest_loguniform('gaussian_noise', 1e-4, 1e-1),
            'time_warping': trial.suggest_categorical('time_warping', [True, False]),
            'time_masking': trial.suggest_categorical('time_masking', [True, False]),
            'window_size': trial.suggest_int('window_size', 10, 30),
            
            # Architectural parameters
            'pos_encoding_type': trial.suggest_categorical('pos_encoding_type', ['sinusoidal', 'fourier']),
            'dim_feedforward': trial.suggest_int('dim_feedforward', 8, 128),
            'encoder_layers': trial.suggest_int('encoder_layers', 1, 3),
            'decoder_layers': trial.suggest_int('decoder_layers', 1, 3),
            'activation_function': trial.suggest_categorical('activation_function', 
                                                           ['relu', 'leaky_relu', 'sigmoid', 'tanh']),
            'num_attention_heads': self.train_dataset.data[self.train_dataset.parameters].shape[1],  # Equal to feature dimension
            'use_linear_embedding': trial.suggest_categorical('use_linear_embedding', [True, False]),
            'layer_normalization': trial.suggest_categorical('layer_normalization', 
                                                           ['layer', 'batch', 'instance']),
            'self_conditioning': trial.suggest_categorical('self_conditioning', [True, False]),
            'num_ffn_layers': trial.suggest_int('num_ffn_layers', 1, 3),
            'phase_type': trial.suggest_categorical('phase_type', ['1phase', '2phase', 'iterative']),
            
            # Model-specific parameters
            'input_dim': len(self.train_dataset.parameters)
        }
        
        return config
    
    def objective(self, trial):
        """Multi-objective optimization function"""
        config = self.get_search_space(trial)
        
        # Create data loaders
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=config['batch_size'], 
            shuffle=True
        )
        val_loader = DataLoader(
            self.val_dataset, 
            batch_size=config['batch_size'], 
            shuffle=False
        )
        
        # Initialize model
        model = TransNAS_TSAD_Model(config)
        
        # Count parameters
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Training setup
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        reconstruction_criterion = nn.MSELoss()
        classification_criterion = nn.BCELoss()
        
        # Training loop (abbreviated for speed)
        model.train()
        training_start_time = time.time()
        
        epochs = 20  # Reduced for faster optimization
        for epoch in range(epochs):
            for batch_input, batch_labels in train_loader:
                optimizer.zero_grad()
                
                reconstruction1, reconstruction2, anomaly_scores = model(batch_input)
                
                # Combined loss
                recon_loss1 = reconstruction_criterion(reconstruction1, batch_input)
                recon_loss2 = reconstruction_criterion(reconstruction2, batch_input)
                class_loss = classification_criterion(anomaly_scores, batch_labels)
                
                total_loss = 0.4 * recon_loss1 + 0.4 * recon_loss2 + 0.2 * class_loss
                total_loss.backward()
                optimizer.step()
        
        training_time = time.time() - training_start_time
        
        # Validation
        model.eval()
        val_predictions = []
        val_labels = []
        
        with torch.no_grad():
            for batch_input, batch_labels in val_loader:
                _, _, anomaly_scores = model(batch_input)
                val_predictions.extend((anomaly_scores > 0.5).float().cpu().numpy())
                val_labels.extend(batch_labels.cpu().numpy())
        
        # Calculate F1 score
        precision, recall, f1, _ = precision_recall_fscore_support(
            val_labels, val_predictions, average='binary', zero_division=0
        )
        
        # Return objectives: maximize F1, minimize parameters
        return f1[0] if isinstance(f1, np.ndarray) else f1, param_count, training_time
    
    def run_optimization(self):
        """Run NSGA-II optimization"""
        # Create study with NSGA-II sampler
        sampler = NSGAIISampler(population_size=20)
        study = optuna.create_study(
            directions=['maximize', 'minimize', 'minimize'],  # F1, parameters, training time
            sampler=sampler
        )
        
        study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True)
        
        # Extract Pareto front
        self.pareto_front = study.best_trials
        
        return study, self.pareto_front

# ==================== EACS Metric ====================

class EACSCalculator:
    """Efficiency-Accuracy-Complexity Score calculator"""
    
    def __init__(self, wa=0.4, wt=0.4, wp=0.2):
        self.wa = wa  # Weight for accuracy (F1 score)
        self.wt = wt  # Weight for training time efficiency
        self.wp = wp  # Weight for parameter complexity
    
    def calculate_eacs(self, f1_scores, training_times, param_counts):
        """Calculate EACS for multiple models"""
        # Normalize values
        f1_max = np.max(f1_scores)
        time_max = np.max(training_times)
        param_max = np.max(param_counts)
        
        # Avoid division by zero
        f1_max = f1_max if f1_max > 0 else 1
        time_max = time_max if time_max > 0 else 1
        param_max = param_max if param_max > 0 else 1
        
        eacs_scores = []
        
        for f1, time, params in zip(f1_scores, training_times, param_counts):
            accuracy_component = self.wa * (f1 / f1_max)
            efficiency_component = self.wt * (1 - time / time_max)
            complexity_component = self.wp * (1 - params / param_max)
            
            eacs = accuracy_component + efficiency_component + complexity_component
            eacs_scores.append(eacs)
        
        return np.array(eacs_scores)

# ==================== Complete TransNAS-TSAD System ====================

class TransNAS_TSAD_System:
    """Complete TransNAS-TSAD system with NAS optimization"""
    
    def __init__(self):
        self.optimizer = None
        self.best_model = None
        self.best_config = None
        self.scaler = None
        self.pot_thresholder = None
        self.eacs_calculator = EACSCalculator()
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess cooling tower data"""
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        else:
            data = pd.read_excel(file_path)
        
        print(f"Loaded data with shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        
        # Handle date column if present
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
            data = data.sort_values('Date')
        
        return data
    
    def run_nas_optimization(self, train_data: pd.DataFrame, val_data: pd.DataFrame, n_trials: int = 100):
        """Run complete NAS optimization"""
        print("Starting Neural Architecture Search with NSGA-II optimization...")
        
        # Create datasets
        train_dataset = CoolingTowerDataset(
            train_data, window_size=20, mode='train', use_ranges=True
        )
        
        # Store scaler
        self.scaler = train_dataset.scaler
        
        val_dataset = CoolingTowerDataset(
            val_data, window_size=20, mode='test', use_ranges=True, scaler=self.scaler
        )
        
        # Initialize optimizer
        self.optimizer = TransNAS_TSAD_Optimizer(train_dataset, val_dataset, n_trials)
        
        # Run optimization
        study, pareto_front = self.optimizer.run_optimization()
        
        print(f"Optimization completed. Found {len(pareto_front)} solutions on Pareto front.")
        
        # Select best model (highest F1 score from Pareto front)
        best_trial = max(pareto_front, key=lambda t: t.values[0])  # Max F1
        self.best_config = best_trial.params
        self.best_config['input_dim'] = len(train_dataset.parameters)
        
        print(f"Best configuration found with F1 score: {best_trial.values[0]:.4f}")
        
        return study, pareto_front
    
    def train_best_model(self, train_data: pd.DataFrame, val_data: pd.DataFrame):
        """Train the best model found by NAS"""
        if self.best_config is None:
            raise ValueError("Must run NAS optimization first")
        
        # Create datasets
        train_dataset = CoolingTowerDataset(
            train_data, 
            window_size=self.best_config['window_size'], 
            mode='train', 
            use_ranges=True
        )
        
        self.scaler = train_dataset.scaler
        
        val_dataset = CoolingTowerDataset(
            val_data, 
            window_size=self.best_config['window_size'], 
            mode='test', 
            use_ranges=True, 
            scaler=self.scaler
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.best_config['batch_size'], 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.best_config['batch_size'], 
            shuffle=False
        )
        
        # Initialize best model
        self.best_model = TransNAS_TSAD_Model(self.best_config)
        
        # Training setup
        optimizer = optim.Adam(self.best_model.parameters(), lr=self.best_config['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        reconstruction_criterion = nn.MSELoss()
        classification_criterion = nn.BCELoss()
        
        # Training loop
        epochs = 100
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        training_history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
        
        for epoch in range(epochs):
            # Training
            self.best_model.train()
            train_losses = []
            
            for batch_input, batch_labels in train_loader:
                optimizer.zero_grad()
                
                reconstruction1, reconstruction2, anomaly_scores = self.best_model(batch_input)
                
                # Three-phase loss
                recon_loss1 = reconstruction_criterion(reconstruction1, batch_input)
                recon_loss2 = reconstruction_criterion(reconstruction2, batch_input)
                class_loss = classification_criterion(anomaly_scores, batch_labels)
                
                # Combined loss with phase-specific weighting
                if self.best_config['phase_type'] == 'iterative':
                    total_loss = 0.5 * recon_loss2 + 0.5 * class_loss
                else:
                    total_loss = 0.3 * recon_loss1 + 0.4 * recon_loss2 + 0.3 * class_loss
                
                total_loss.backward()
                optimizer.step()
                train_losses.append(total_loss.item())
            
            # Validation
            self.best_model.eval()
            val_losses = []
            val_predictions = []
            val_labels = []
            
            with torch.no_grad():
                for batch_input, batch_labels in val_loader:
                    reconstruction1, reconstruction2, anomaly_scores = self.best_model(batch_input)
                    
                    recon_loss1 = reconstruction_criterion(reconstruction1, batch_input)
                    recon_loss2 = reconstruction_criterion(reconstruction2, batch_input)
                    class_loss = classification_criterion(anomaly_scores, batch_labels)
                    
                    if self.best_config['phase_type'] == 'iterative':
                        total_loss = 0.5 * recon_loss2 + 0.5 * class_loss
                    else:
                        total_loss = 0.3 * recon_loss1 + 0.4 * recon_loss2 + 0.3 * class_loss
                    
                    val_losses.append(total_loss.item())
                    val_predictions.extend((anomaly_scores > 0.5).float().cpu().numpy())
                    val_labels.extend(batch_labels.cpu().numpy())
            
            # Calculate metrics
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                val_labels, val_predictions, average='binary', zero_division=0
            )
            val_f1 = f1 if not isinstance(f1, np.ndarray) else f1[0]
            
            # Update history
            training_history['train_loss'].append(avg_train_loss)
            training_history['val_loss'].append(avg_val_loss)
            training_history['val_f1'].append(val_f1)
            
            # Update scheduler
            scheduler.step(avg_val_loss)
            
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, "
                  f"Val F1: {val_f1:.4f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(self.best_model.state_dict(), 'best_transnas_tsad_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.best_model.load_state_dict(torch.load('best_transnas_tsad_model.pth'))
        print("Training completed!")
        
        return training_history
    
    def detect_anomalies(self, test_data: pd.DataFrame) -> Dict:
        """Detect anomalies using trained model and enhanced POT"""
        if self.best_model is None:
            raise ValueError("Must train model first")
        
        # Create test dataset
        test_dataset = CoolingTowerDataset(
            test_data,
            window_size=self.best_config['window_size'],
            mode='test',
            use_ranges=True,
            scaler=self.scaler
        )
        
        test_loader = DataLoader(test_dataset, batch_size=self.best_config['batch_size'], shuffle=False)
        
        # Get model predictions
        self.best_model.eval()
        all_reconstruction_errors = []
        all_anomaly_scores = []
        all_labels = []
        
        with torch.no_grad():
            for batch_input, batch_labels in test_loader:
                reconstruction1, reconstruction2, anomaly_scores = self.best_model(batch_input)
                
                # Calculate reconstruction errors (dual pathway)
                recon_error1 = torch.mean((batch_input - reconstruction1) ** 2, dim=(1, 2))
                recon_error2 = torch.mean((batch_input - reconstruction2) ** 2, dim=(1, 2))
                
                # Combined anomaly score as in paper
                combined_scores = 0.5 * recon_error1 + 0.5 * anomaly_scores.squeeze()
                
                all_reconstruction_errors.extend(combined_scores.cpu().numpy())
                all_anomaly_scores.extend(anomaly_scores.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        # Initialize enhanced POT thresholder
        self.pot_thresholder = EnhancedPOTThresholding(quantile=0.95, alpha=0.1)
        
        # Fit POT threshold
        reconstruction_errors = np.array(all_reconstruction_errors)
        self.pot_thresholder.fit_pot_threshold(reconstruction_errors)
        
        # Detect anomalies using enhanced POT methods
        pot_results = self.pot_thresholder.detect_anomalies(reconstruction_errors, use_enhanced=True)
        
        # Calculate metrics for different detection methods
        results = {
            'reconstruction_errors': reconstruction_errors,
            'anomaly_scores': np.array(all_anomaly_scores).flatten(),
            'ground_truth_labels': np.array(all_labels).flatten(),
            'pot_results': pot_results
        }
        
        # Calculate performance metrics for each method
        methods = ['basic_pot', 'mpot', 'mat', 'enhanced']
        for method in methods:
            if method in pot_results:
                anomalies = pot_results[method]
                precision, recall, f1, _ = precision_recall_fscore_support(
                    results['ground_truth_labels'][:len(anomalies)], 
                    anomalies.astype(int), 
                    average='binary', 
                    zero_division=0
                )
                
                results[f'{method}_metrics'] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'accuracy': accuracy_score(results['ground_truth_labels'][:len(anomalies)], anomalies.astype(int))
                }
        
        return results
    
    def generate_report(self, results: Dict, pareto_front: List = None) -> str:
        """Generate comprehensive TransNAS-TSAD report"""
        report = f"""
================================================================================
TRANSNAS-TSAD: NEURAL ARCHITECTURE SEARCH FOR TIME SERIES ANOMALY DETECTION
================================================================================

ANALYSIS DATE: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

SYSTEM CONFIGURATION:
--------------------
Best Architecture Configuration:
{self._format_config(self.best_config)}

NSGA-II OPTIMIZATION RESULTS:
----------------------------
"""
        
        if pareto_front:
            report += f"Pareto Front Solutions: {len(pareto_front)}\n"
            report += "Top 5 Solutions (F1, Parameters, Training Time):\n"
            sorted_solutions = sorted(pareto_front, key=lambda t: t.values[0], reverse=True)[:5]
            for i, trial in enumerate(sorted_solutions, 1):
                report += f"  {i}. F1: {trial.values[0]:.4f}, Params: {trial.values[1]}, Time: {trial.values[2]:.2f}s\n"
        
        report += f"""
ANOMALY DETECTION PERFORMANCE:
-----------------------------
Total Samples Analyzed: {len(results['reconstruction_errors'])}

Enhanced POT Detection Results:
"""
        
        # Add metrics for each detection method
        methods = ['basic_pot', 'mpot', 'mat', 'enhanced']
        for method in methods:
            if f'{method}_metrics' in results:
                metrics = results[f'{method}_metrics']
                method_name = method.upper().replace('_', ' ')
                report += f"""
{method_name} Method:
  Precision: {metrics['precision']:.4f}
  Recall: {metrics['recall']:.4f}
  F1 Score: {metrics['f1']:.4f}
  Accuracy: {metrics['accuracy']:.4f}
"""
        
        # Calculate EACS if multiple models available
        if pareto_front:
            f1_scores = [t.values[0] for t in pareto_front]
            param_counts = [t.values[1] for t in pareto_front]
            training_times = [t.values[2] for t in pareto_front]
            
            eacs_scores = self.eacs_calculator.calculate_eacs(f1_scores, training_times, param_counts)
            best_eacs_idx = np.argmax(eacs_scores)
            
            report += f"""
EFFICIENCY-ACCURACY-COMPLEXITY SCORE (EACS):
-------------------------------------------
Best EACS Score: {eacs_scores[best_eacs_idx]:.4f}
Corresponding Solution:
  F1 Score: {f1_scores[best_eacs_idx]:.4f}
  Parameters: {param_counts[best_eacs_idx]}
  Training Time: {training_times[best_eacs_idx]:.2f}s
"""
        
        report += "\n" + "=" * 80 + "\n"
        
        return report
    
    def _format_config(self, config: Dict) -> str:
        """Format configuration for report"""
        formatted = ""
        for key, value in config.items():
            formatted += f"  {key}: {value}\n"
        return formatted
    
    def visualize_results(self, results: Dict, pareto_front: List = None):
        """Visualize TransNAS-TSAD results"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Anomaly scores over time
        recon_errors = results['reconstruction_errors']
        axes[0, 0].plot(recon_errors, label='Reconstruction Error', alpha=0.7)
        
        if 'mpot_threshold' in results['pot_results']:
            axes[0, 0].axhline(y=results['pot_results']['mpot_threshold'], 
                              color='r', linestyle='--', label='mPOT Threshold')
        
        enhanced_anomalies = results['pot_results'].get('enhanced', np.array([]))
        if len(enhanced_anomalies) > 0:
            anomaly_indices = np.where(enhanced_anomalies)[0]
            axes[0, 0].scatter(anomaly_indices, recon_errors[anomaly_indices], 
                              color='red', s=20, label='Detected Anomalies')
        
        axes[0, 0].set_title('TransNAS-TSAD Anomaly Detection Results')
        axes[0, 0].set_xlabel('Sample Index')
        axes[0, 0].set_ylabel('Anomaly Score')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Pareto Front visualization
        if pareto_front and len(pareto_front) > 1:
            f1_scores = [t.values[0] for t in pareto_front]
            param_counts = [t.values[1] for t in pareto_front]
            
            axes[0, 1].scatter(param_counts, f1_scores, alpha=0.6, s=50)
            axes[0, 1].set_xlabel('Number of Parameters')
            axes[0, 1].set_ylabel('F1 Score')
            axes[0, 1].set_title('Pareto Front: F1 Score vs Model Complexity')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Detection method comparison
        methods = ['basic_pot', 'mpot', 'mat', 'enhanced']
        f1_scores = []
        method_names = []
        
        for method in methods:
            if f'{method}_metrics' in results:
                f1_scores.append(results[f'{method}_metrics']['f1'])
                method_names.append(method.upper().replace('_', ' '))
        
        if f1_scores:
            axes[1, 0].bar(method_names, f1_scores, color=['skyblue', 'lightgreen', 'orange', 'coral'])
            axes[1, 0].set_ylabel('F1 Score')
            axes[1, 0].set_title('Detection Method Performance Comparison')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. EACS visualization
        if pareto_front:
            f1_vals = [t.values[0] for t in pareto_front]
            param_vals = [t.values[1] for t in pareto_front]
            time_vals = [t.values[2] for t in pareto_front]
            
            eacs_scores = self.eacs_calculator.calculate_eacs(f1_vals, time_vals, param_vals)
            
            scatter = axes[1, 1].scatter(param_vals, f1_vals, c=eacs_scores, 
                                        cmap='viridis', s=50, alpha=0.7)
            plt.colorbar(scatter, ax=axes[1, 1], label='EACS Score')
            axes[1, 1].set_xlabel('Number of Parameters')
            axes[1, 1].set_ylabel('F1 Score')
            axes[1, 1].set_title('EACS Distribution on Pareto Front')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# ==================== Main Execution Function ====================

def main():
    """Main execution function for TransNAS-TSAD"""
    # Initialize system
    system = TransNAS_TSAD_System()
    
    # Load your data
    data = system.load_data('Parameters_1.csv')  # Update with your file path
    
    # Split data for training/validation/testing
    train_size = int(0.6 * len(data))
    val_size = int(0.2 * len(data))
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    print(f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Run NAS optimization
    print("\n" + "="*50)
    print("STEP 1: NEURAL ARCHITECTURE SEARCH")
    print("="*50)
    
    study, pareto_front = system.run_nas_optimization(train_data, val_data, n_trials=50)  # Reduced for demo
    
    # Train best model
    print("\n" + "="*50)
    print("STEP 2: TRAINING BEST MODEL")
    print("="*50)
    
    training_history = system.train_best_model(train_data, val_data)
    
    # Detect anomalies on test data
    print("\n" + "="*50)
    print("STEP 3: ANOMALY DETECTION")
    print("="*50)
    
    results = system.detect_anomalies(test_data)
    
    # Generate comprehensive report
    print("\n" + "="*50)
    print("STEP 4: RESULTS AND ANALYSIS")
    print("="*50)
    
    report = system.generate_report(results, pareto_front)
    print(report)
    
    # Save report
    with open('TransNAS_TSAD_Report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Visualize results
    system.visualize_results(results, pareto_front)
    
    print("\nTransNAS-TSAD analysis complete!")
    print("Report saved as 'TransNAS_TSAD_Report.txt'")
    
    return system, results, pareto_front

if __name__ == "__main__":
    # Run the complete TransNAS-TSAD system
    system, results, pareto_front = main()