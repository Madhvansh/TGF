"""
MOMENT Anomaly Detection - Runnable Script
==========================================

Run directly: python moment_detector.py

This script trains a MOMENT transformer on your cooling tower data
and detects anomalies. Easy to modify and tweak parameters.

Author: TGF AI System
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import warnings
import pickle
import os
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - MODIFY THESE PARAMETERS AS NEEDED
# ============================================================================

CONFIG = {
    # Data parameters
    'data_path': './Parameters_5K.csv',
    'window_size': 168,          # Hours in window (168 = 1 week)
    'stride': 24,                # Hours between windows (24 = 1 day)
    
    # Model parameters
    'd_model': 256,              # Transformer dimension
    'nhead': 8,                  # Number of attention heads
    'num_layers': 4,             # Number of transformer layers
    'dropout': 0.1,              # Dropout rate
    
    # Training parameters
    'batch_size': 32,
    'epochs': 50,
    'lr': 1e-4,                  # Learning rate
    'patience': 10,              # Early stopping patience
    
    # Data split
    'train_split': 0.7,          # 70% for training
    'val_split': 0.15,           # 15% for validation
    # test_split = 0.15 (implicit)
    
    # Anomaly detection
    'threshold_method': 'zscore',  # 'zscore' or 'percentile'
    'z_multiplier': 3.0,           # For zscore method
    'contamination': 0.01,         # For percentile method
    
    # Output
    'save_model': True,
    'model_name': 'moment_model.pkl',
    'save_plots': True,
}

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'moment_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATASET CLASS
# ============================================================================

class CoolingTowerDataset(Dataset):
    """Dataset for cooling tower time series data"""
    
    def __init__(self, data, window_size, stride, scalers=None, fit_scalers=True):
        self.window_size = window_size
        self.stride = stride
        
        # Select numeric columns
        self.feature_cols = [col for col in data.columns 
                           if col not in ['Date', 'Source_Sheet'] 
                           and data[col].dtype in ['float64', 'int64']]
        
        logger.info(f"Using {len(self.feature_cols)} features: {self.feature_cols}")
        
        # Sort by date
        self.data = data.sort_values('Date').reset_index(drop=True)
        self.dates = self.data['Date'].values
        
        # Handle missing values
        logger.info("Handling missing values...")
        self.data[self.feature_cols] = self.data[self.feature_cols].fillna(method='ffill').fillna(method='bfill')
        
        # Scaling
        if scalers is None and fit_scalers:
            self.scalers = {}
            logger.info("Fitting RobustScalers...")
            for col in self.feature_cols:
                scaler = RobustScaler()
                self.data[col] = scaler.fit_transform(self.data[[col]])
                self.scalers[col] = scaler
        elif scalers is not None:
            self.scalers = scalers
            for col in self.feature_cols:
                if col in self.scalers:
                    self.data[col] = self.scalers[col].transform(self.data[[col]])
        else:
            self.scalers = None
        
        # Create sliding windows
        self.windows = []
        self.window_dates = []
        self.window_indices = []
        
        for i in range(0, len(self.data) - window_size + 1, stride):
            window = self.data[self.feature_cols].iloc[i:i+window_size].values
            if not np.isnan(window).any():
                self.windows.append(window)
                self.window_dates.append(self.dates[i:i+window_size])
                self.window_indices.append(i)
        
        self.windows = np.array(self.windows, dtype=np.float32)
        logger.info(f"Created {len(self.windows)} valid windows")
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.windows[idx])
    
    def get_window_info(self, idx):
        return {
            'start_date': self.window_dates[idx][0],
            'end_date': self.window_dates[idx][-1],
            'start_idx': self.window_indices[idx]
        }


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MOMENT_Model(nn.Module):
    """MOMENT transformer for reconstruction-based anomaly detection"""
    
    def __init__(self, n_features, d_model=256, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        
        self.n_features = n_features
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(n_features, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_features)
        )
        
        logger.info(f"Initialized MOMENT model with {sum(p.numel() for p in self.parameters()):,} parameters")
    
    def forward(self, x):
        # x: [batch_size, seq_len, n_features]
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        x = self.transformer_encoder(x)
        x = self.norm(x)
        reconstruction = self.output_projection(x)
        return reconstruction


# ============================================================================
# TRAINING AND DETECTION
# ============================================================================

def train_model(model, train_loader, val_loader, config, device):
    """Train the MOMENT model"""
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['lr'], 
        weight_decay=1e-5
    )
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    logger.info(f"Starting training for {config['epochs']} epochs...")
    
    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_losses = []
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstruction = model(batch)
            loss = criterion(reconstruction, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                reconstruction = model(batch)
                loss = criterion(reconstruction, batch)
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        history['val_loss'].append(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            logger.info(f"Epoch {epoch+1}/{config['epochs']} - Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f} [BEST]")
        else:
            patience_counter += 1
            logger.info(f"Epoch {epoch+1}/{config['epochs']} - Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f}")
            if patience_counter >= config['patience']:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        scheduler.step()
    
    # Load best model
    model.load_state_dict(best_model_state)
    logger.info(f"Training complete. Best val loss: {best_val_loss:.6f}")
    
    return model, history


def compute_reconstruction_errors(model, dataloader, device):
    """Compute reconstruction errors"""
    
    model.eval()
    errors = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            reconstruction = model(batch)
            mse = torch.mean((batch - reconstruction) ** 2, dim=(1, 2))
            errors.extend(mse.cpu().numpy())
    
    errors = np.array(errors)
    logger.info(f"Computed {len(errors)} reconstruction errors")
    logger.info(f"  Mean: {np.mean(errors):.6f}, Std: {np.std(errors):.6f}")
    logger.info(f"  Min: {np.min(errors):.6f}, Max: {np.max(errors):.6f}")
    
    return errors


def fit_threshold(val_errors, method='zscore', z_multiplier=3.0, contamination=0.01):
    """Fit anomaly detection threshold"""
    
    if method == 'zscore':
        mean_error = np.mean(val_errors)
        std_error = np.std(val_errors)
        threshold = mean_error + z_multiplier * std_error
        logger.info(f"Fitted z-score threshold ({z_multiplier}σ): {threshold:.6f}")
    elif method == 'percentile':
        percentile = 100 * (1 - contamination)
        threshold = np.percentile(val_errors, percentile)
        logger.info(f"Fitted percentile threshold: {threshold:.6f} ({percentile:.1f}th percentile)")
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return threshold


def detect_anomalies(errors, threshold):
    """Detect anomalies based on threshold"""
    
    anomalies = errors > threshold
    n_anomalies = np.sum(anomalies)
    logger.info(f"Detected {n_anomalies}/{len(anomalies)} anomalies ({n_anomalies/len(anomalies)*100:.2f}%)")
    
    return anomalies


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_history(history, save_path='training_history.png'):
    """Plot training curves"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.set_title('MOMENT Training History', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved training history to {save_path}")
    plt.close()


def plot_results(errors, anomalies, threshold, save_path='anomaly_results.png'):
    """Plot anomaly detection results"""
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # Plot 1: Reconstruction errors
    ax1 = axes[0]
    window_indices = np.arange(len(errors))
    ax1.plot(window_indices, errors, color='steelblue', alpha=0.7, linewidth=1)
    ax1.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.2f})', linewidth=2)
    ax1.scatter(window_indices[anomalies], errors[anomalies], 
               color='red', s=100, label=f'Anomalies (n={np.sum(anomalies)})', 
               zorder=5, alpha=0.7, edgecolors='darkred', linewidths=1.5)
    ax1.set_xlabel('Window Index', fontsize=11)
    ax1.set_ylabel('Reconstruction Error (MSE)', fontsize=11)
    ax1.set_title('Reconstruction Error Timeline', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Distribution
    ax2 = axes[1]
    ax2.hist(errors, bins=50, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
    ax2.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.2f})', linewidth=2.5)
    ax2.axvline(x=np.mean(errors), color='green', linestyle=':', alpha=0.7, 
               label=f'Mean ({np.mean(errors):.2f})', linewidth=2)
    ax2.set_xlabel('Reconstruction Error (MSE)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Distribution of Reconstruction Errors', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Anomaly timeline
    ax3 = axes[2]
    anomaly_signal = anomalies.astype(float)
    ax3.fill_between(range(len(anomaly_signal)), 0, anomaly_signal, 
                    alpha=0.6, color='crimson', label='Anomaly Windows', step='mid')
    ax3.set_xlabel('Window Index', fontsize=11)
    ax3.set_ylabel('Anomaly Flag', fontsize=11)
    ax3.set_title('Anomaly Detection Timeline', fontsize=13, fontweight='bold')
    ax3.set_ylim(-0.1, 1.1)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved results to {save_path}")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    logger.info("=" * 80)
    logger.info("MOMENT ANOMALY DETECTION - STARTING")
    logger.info("=" * 80)
    
    # Print configuration
    logger.info("\nConfiguration:")
    for key, value in CONFIG.items():
        logger.info(f"  {key}: {value}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"\nUsing device: {device}")
    
    # ========================================================================
    # STEP 1: Load and preprocess data
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Loading and Preprocessing Data")
    logger.info("=" * 80)
    
    df = pd.read_csv(CONFIG['data_path'])
    df['Date'] = pd.to_datetime(df['Date'])
    logger.info(f"Loaded {len(df)} samples")
    
    # Create dataset
    full_dataset = CoolingTowerDataset(
        df, 
        window_size=CONFIG['window_size'],
        stride=CONFIG['stride'],
        fit_scalers=True
    )
    
    # ========================================================================
    # STEP 2: Split data
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Splitting Data")
    logger.info("=" * 80)
    
    n_total = len(full_dataset)
    n_train = int(n_total * CONFIG['train_split'])
    n_val = int(n_total * CONFIG['val_split'])
    n_test = n_total - n_train - n_val
    
    logger.info(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")
    
    train_dataset = torch.utils.data.Subset(full_dataset, range(0, n_train))
    val_dataset = torch.utils.data.Subset(full_dataset, range(n_train, n_train + n_val))
    test_dataset = torch.utils.data.Subset(full_dataset, range(n_train + n_val, n_total))
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # ========================================================================
    # STEP 3: Initialize model
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Initializing Model")
    logger.info("=" * 80)
    
    n_features = len(full_dataset.feature_cols)
    model = MOMENT_Model(
        n_features=n_features,
        d_model=CONFIG['d_model'],
        nhead=CONFIG['nhead'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout']
    ).to(device)
    
    # ========================================================================
    # STEP 4: Train model
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Training Model")
    logger.info("=" * 80)
    
    model, history = train_model(model, train_loader, val_loader, CONFIG, device)
    
    if CONFIG['save_plots']:
        plot_training_history(history)
    
    # ========================================================================
    # STEP 5: Compute reconstruction errors
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Computing Reconstruction Errors")
    logger.info("=" * 80)
    
    val_errors = compute_reconstruction_errors(model, val_loader, device)
    test_errors = compute_reconstruction_errors(model, test_loader, device)
    
    # ========================================================================
    # STEP 6: Fit threshold and detect anomalies
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: Detecting Anomalies")
    logger.info("=" * 80)
    
    threshold = fit_threshold(
        val_errors, 
        method=CONFIG['threshold_method'],
        z_multiplier=CONFIG['z_multiplier'],
        contamination=CONFIG['contamination']
    )
    
    test_anomalies = detect_anomalies(test_errors, threshold)
    
    # ========================================================================
    # STEP 7: Save results
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 7: Saving Results")
    logger.info("=" * 80)
    
    # Save results CSV
    results_df = pd.DataFrame({
        'window_idx': range(len(test_errors)),
        'reconstruction_error': test_errors,
        'is_anomaly': test_anomalies,
        'threshold': threshold,
        'error_zscore': (test_errors - np.mean(val_errors)) / np.std(val_errors)
    })
    results_df.to_csv('anomaly_results.csv', index=False)
    logger.info("Saved results to: anomaly_results.csv")
    
    # Save anomalies only
    anomalous_df = results_df[results_df['is_anomaly']]
    anomalous_df.to_csv('anomalous_windows.csv', index=False)
    logger.info(f"Saved {len(anomalous_df)} anomalies to: anomalous_windows.csv")
    
    # Save plots
    if CONFIG['save_plots']:
        plot_results(test_errors, test_anomalies, threshold)
    
    # Save model and metadata
    if CONFIG['save_model']:
        model_package = {
            'model_state_dict': model.state_dict(),
            'config': CONFIG,
            'scalers': full_dataset.scalers,
            'feature_cols': full_dataset.feature_cols,
            'threshold': threshold,
            'val_errors_stats': {
                'mean': float(np.mean(val_errors)),
                'std': float(np.std(val_errors))
            },
            'n_features': n_features
        }
        
        with open(CONFIG['model_name'], 'wb') as f:
            pickle.dump(model_package, f)
        
        logger.info(f"Saved model to: {CONFIG['model_name']}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Total windows: {n_total}")
    logger.info(f"Features: {n_features}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Final train loss: {history['train_loss'][-1]:.6f}")
    logger.info(f"Final val loss: {history['val_loss'][-1]:.6f}")
    logger.info(f"Threshold: {threshold:.6f}")
    logger.info(f"Test anomalies: {np.sum(test_anomalies)}/{len(test_anomalies)} ({np.sum(test_anomalies)/len(test_anomalies)*100:.2f}%)")
    logger.info("\n" + "=" * 80)
    logger.info("MOMENT ANOMALY DETECTION - COMPLETE ✓")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()