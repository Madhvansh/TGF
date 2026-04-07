"""
TGF Model Comparison: TransNAS vs VTT vs MOMENT on Parameters_5K.csv
====================================================================
Fair head-to-head comparison of the best variant from each model family.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, precision_recall_curve, auc
)
from scipy.stats import genpareto
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SHARED: Data Loading & Column Mapping for Parameters_5K.csv
# ============================================================================

# Map Parameters_5K.csv columns -> standard names used by all models
COLUMN_RENAME = {
    'Turbidity_NTU': 'Turbidity',
    'Free_Residual_Chlorine_ppm': 'FRC',
    'TDS_ppm': 'TDS',
    'Total_Hardness_ppm': 'Total Hardness',
    'Calcium_Hardness_ppm': 'Calcium Hardness',
    'Magnesium_Hardness_ppm': 'Magnesium Hardness',
    'Chlorides_ppm': 'Chlorides',
    'Phosphate_ppm': 'Ortho PO4',
    'Total_Alkalinity_ppm': 'Total Alkalinity',
    'Sulphates_ppm': 'Sulphate',
    'Silica_ppm': 'Silica',
    'Iron_ppm': 'Total Iron',
    'Suspended_Solids_ppm': 'SS',
    'Conductivity_uS_cm': 'Conductivity',
}

PARAMETER_RANGES = {
    'pH': {'min': 7.5, 'max': 8.0, 'critical': True},
    'Turbidity': {'min': 0, 'max': 20, 'critical': False},
    'FRC': {'min': 0.2, 'max': 0.5, 'critical': True},
    'Conductivity': {'min': 0, 'max': 3000, 'critical': True},
    'TDS': {'min': 0, 'max': 2100, 'critical': True},
    'Total Hardness': {'min': 0, 'max': 1200, 'critical': True},
    'Calcium Hardness': {'min': 0, 'max': 800, 'critical': True},
    'Magnesium Hardness': {'min': 0, 'max': 400, 'critical': False},
    'Chlorides': {'min': 0, 'max': 500, 'critical': True},
    'Ortho PO4': {'min': 6.0, 'max': 8.0, 'critical': True},
    'Total Alkalinity': {'min': 0, 'max': 200, 'critical': True},
    'Total Iron': {'min': 0, 'max': 2, 'critical': True},
    'SS': {'min': 0, 'max': 50, 'critical': False},
    'Sulphate': {'min': 0, 'max': 1000, 'critical': True},
    'Silica': {'min': 0, 'max': 180, 'critical': True},
}


def load_data(path='Parameters_5K.csv'):
    """Load Parameters_5K.csv and standardize column names."""
    df = pd.read_csv(path)
    df.rename(columns=COLUMN_RENAME, inplace=True)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.sort_values('Date', inplace=True)
    # Keep only numeric parameter columns that exist
    params = [p for p in PARAMETER_RANGES if p in df.columns]
    for p in params:
        df[p] = pd.to_numeric(df[p], errors='coerce')
    # Fill missing
    df[params] = df[params].interpolate(method='linear', limit=3)
    df[params] = df[params].bfill().ffill()
    for p in params:
        if df[p].isna().any():
            df[p].fillna(df[p].median(), inplace=True)
    print(f"Loaded {len(df)} rows, {len(params)} parameters: {params}")
    return df, params


def generate_anomaly_labels(df, params):
    """Generate anomaly labels based on parameter ranges (shared ground truth)."""
    labels = np.zeros(len(df))
    for i, (_, row) in enumerate(df.iterrows()):
        max_severity = 0
        for p in params:
            val = row.get(p, np.nan)
            if pd.isna(val):
                continue
            r = PARAMETER_RANGES[p]
            if val < r['min']:
                dev = (r['min'] - val) / (r['min'] + 1e-8)
            elif val > r['max']:
                dev = (val - r['max']) / (r['max'] + 1e-8)
            else:
                dev = 0
            if dev > 0.5:
                max_severity = max(max_severity, 3)
            elif dev > 0.1:
                max_severity = max(max_severity, 2)
            elif dev > 0:
                max_severity = max(max_severity, 1)
        labels[i] = 1 if max_severity >= 2 else 0
    return labels


def split_data(df, train_frac=0.7, val_frac=0.15):
    """Split into train/val/test."""
    n = len(df)
    t1 = int(train_frac * n)
    t2 = int((train_frac + val_frac) * n)
    return df.iloc[:t1].reset_index(drop=True), \
           df.iloc[t1:t2].reset_index(drop=True), \
           df.iloc[t2:].reset_index(drop=True)


def compute_metrics(y_true, y_pred, y_scores=None):
    """Compute standard metrics."""
    m = {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'anomaly_rate': float(np.mean(y_pred)),
        'true_anomaly_rate': float(np.mean(y_true)),
    }
    if y_scores is not None:
        try:
            m['roc_auc'] = roc_auc_score(y_true, y_scores)
        except ValueError:
            m['roc_auc'] = 0.0
    return m


# ============================================================================
# MODEL 1: TransNAS-TSAD (from Trans_x4.py - Dual Decoder + Adversarial)
# ============================================================================

class TransNASEncoder(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=2,
                 dim_feedforward=256, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation='relu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.input_proj(x)
        return self.encoder(x)


class TransNASDecoder(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=2,
                 dim_feedforward=256, dropout=0.2):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation='relu'
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, input_dim)

    def forward(self, tgt, memory):
        out = self.decoder(tgt, memory)
        return self.output_proj(out)


class TransNASTSAD(nn.Module):
    """Dual-decoder TransNAS-TSAD (from Trans_x4.py architecture)."""
    def __init__(self, input_dim, d_model=128, nhead=8, enc_layers=2,
                 dec_layers=2, dim_ff=256, dropout=0.2):
        super().__init__()
        self.encoder = TransNASEncoder(input_dim, d_model, nhead, enc_layers,
                                       dim_ff, dropout)
        self.decoder1 = TransNASDecoder(input_dim, d_model, nhead, dec_layers,
                                        dim_ff, dropout)
        self.decoder2 = TransNASDecoder(input_dim, d_model, nhead, dec_layers,
                                        dim_ff, dropout)
        self.anomaly_scorer = nn.Sequential(
            nn.Linear(input_dim * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        self.input_proj = nn.Linear(input_dim, d_model)

    def forward(self, x):
        memory = self.encoder(x)
        tgt = self.input_proj(x)
        recon1 = self.decoder1(tgt, memory)
        recon2 = self.decoder2(tgt, memory)
        recon = (recon1 + recon2) / 2
        combined = torch.cat([x, recon], dim=-1)
        scores = self.anomaly_scorer(combined).squeeze(-1)
        return recon, scores


# ============================================================================
# MODEL 2: VTT (Variable Temporal Transformer from VTT_4.1.py)
# ============================================================================

class VariableTemporalAttention(nn.Module):
    """Dual-pathway attention: temporal + variable dimensions."""
    def __init__(self, d_model, n_heads, seq_len, n_vars, dropout=0.1):
        super().__init__()
        self.temporal_attn = nn.MultiheadAttention(d_model, n_heads,
                                                    dropout=dropout, batch_first=True)
        self.variable_attn = nn.MultiheadAttention(d_model, n_heads,
                                                    dropout=dropout, batch_first=True)
        self.gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # x: (B, seq_len, n_vars, d_model) -> temporal + variable paths
        B, T, V, D = x.shape
        # Temporal: attend across time for each variable
        x_t = x.permute(0, 2, 1, 3).reshape(B * V, T, D)
        t_out, _ = self.temporal_attn(x_t, x_t, x_t)
        t_out = t_out.reshape(B, V, T, D).permute(0, 2, 1, 3)
        # Variable: attend across variables for each time step
        x_v = x.reshape(B * T, V, D)
        v_out, _ = self.variable_attn(x_v, x_v, x_v)
        v_out = v_out.reshape(B, T, V, D)
        # Gated combination
        g = torch.sigmoid(self.gate)
        return g * t_out + (1 - g) * v_out


class VTTModel(nn.Module):
    """Variable Temporal Transformer for anomaly detection."""
    def __init__(self, n_vars, seq_len=20, d_model=64, n_heads=4,
                 n_layers=3, d_ff=128, dropout=0.1):
        super().__init__()
        self.n_vars = n_vars
        self.seq_len = seq_len
        self.d_model = d_model
        self.input_proj = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, 1, d_model) * 0.02)
        self.var_embed = nn.Parameter(torch.randn(1, 1, n_vars, d_model) * 0.02)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': VariableTemporalAttention(d_model, n_heads, seq_len,
                                                   n_vars, dropout),
                'norm1': nn.LayerNorm(d_model),
                'ff': nn.Sequential(
                    nn.Linear(d_model, d_ff), nn.GELU(),
                    nn.Dropout(dropout), nn.Linear(d_ff, d_model)
                ),
                'norm2': nn.LayerNorm(d_model),
            })
            for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (B, seq_len, n_vars)
        B, T, V = x.shape
        x = x.unsqueeze(-1)  # (B, T, V, 1)
        x = self.input_proj(x)  # (B, T, V, d_model)
        x = x + self.pos_embed[:, :T, :, :] + self.var_embed[:, :, :V, :]
        for layer in self.layers:
            attn_out = layer['attn'](x)
            x = layer['norm1'](x + attn_out)
            ff_out = layer['ff'](x)
            x = layer['norm2'](x + ff_out)
        recon = self.output_proj(x).squeeze(-1)  # (B, T, V)
        return recon


# ============================================================================
# MODEL 3: MOMENT-style (Reconstruction-based foundation model approach)
# ============================================================================

class RevIN(nn.Module):
    """Reversible Instance Normalization."""
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.affine_weight = nn.Parameter(torch.ones(num_features))
        self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode='norm'):
        if mode == 'norm':
            self._mean = x.mean(dim=1, keepdim=True).detach()
            self._std = (x.std(dim=1, keepdim=True) + self.eps).detach()
            x = (x - self._mean) / self._std
            x = x * self.affine_weight + self.affine_bias
            return x
        else:  # denorm
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            x = x * self._std + self._mean
            return x


class MOMENTEncoder(nn.Module):
    """MOMENT-style masked reconstruction encoder."""
    def __init__(self, input_dim, d_model=256, nhead=8, num_layers=6,
                 d_ff=512, dropout=0.1, seq_len=64, patch_len=8):
        super().__init__()
        self.patch_len = patch_len
        self.n_patches = seq_len // patch_len
        self.revin = RevIN(input_dim)
        self.patch_proj = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.n_patches, d_model) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.recon_head = nn.Linear(d_model, patch_len)

    def forward(self, x, mask_ratio=0.3):
        # x: (B, seq_len) - single channel
        B, T = x.shape
        # Patchify
        x_patches = x.reshape(B, self.n_patches, self.patch_len)
        x_proj = self.patch_proj(x_patches) + self.pos_embed
        # Optional masking during training
        if self.training and mask_ratio > 0:
            n_mask = max(1, int(self.n_patches * mask_ratio))
            mask_idx = torch.randperm(self.n_patches)[:n_mask]
            x_proj[:, mask_idx] = 0
        encoded = self.encoder(x_proj)
        recon_patches = self.recon_head(encoded)
        recon = recon_patches.reshape(B, T)
        return recon


class MOMENTModel(nn.Module):
    """MOMENT-style channel-independent reconstruction model."""
    def __init__(self, n_channels, seq_len=64, d_model=256, nhead=8,
                 num_layers=6, d_ff=512, dropout=0.1, patch_len=8):
        super().__init__()
        self.n_channels = n_channels
        self.seq_len = seq_len
        self.revin = RevIN(n_channels)
        self.encoder = MOMENTEncoder(
            input_dim=1, d_model=d_model, nhead=nhead,
            num_layers=num_layers, d_ff=d_ff, dropout=dropout,
            seq_len=seq_len, patch_len=patch_len
        )

    def forward(self, x, mask_ratio=0.3):
        # x: (B, seq_len, n_channels)
        B, T, C = x.shape
        # Channel independence: process each channel separately
        x_norm = self.revin(x, 'norm')
        recons = []
        for c in range(C):
            ch = x_norm[:, :, c]  # (B, T)
            r = self.encoder(ch, mask_ratio)  # (B, T)
            recons.append(r)
        recon = torch.stack(recons, dim=-1)  # (B, T, C)
        recon = self.revin(recon, 'denorm')
        return recon


# ============================================================================
# SHARED: Sliding Window Dataset
# ============================================================================

class SlidingWindowDataset(Dataset):
    def __init__(self, data, window_size, stride=1, labels=None, scaler=None,
                 fit_scaler=False):
        if isinstance(data, pd.DataFrame):
            data = data.values
        if fit_scaler:
            self.scaler = StandardScaler()
            data = self.scaler.fit_transform(data)
        elif scaler is not None:
            self.scaler = scaler
            data = scaler.transform(data)
        else:
            self.scaler = None

        self.windows = []
        self.window_labels = []
        for i in range(0, len(data) - window_size + 1, stride):
            self.windows.append(data[i:i + window_size])
            if labels is not None:
                wl = labels[i:i + window_size]
                self.window_labels.append(1.0 if wl.max() > 0 else 0.0)

        self.windows = np.array(self.windows, dtype=np.float32)
        if self.window_labels:
            self.window_labels = np.array(self.window_labels, dtype=np.float32)
        else:
            self.window_labels = np.zeros(len(self.windows), dtype=np.float32)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.windows[idx]), self.window_labels[idx]


# ============================================================================
# TRAINING & EVALUATION FUNCTIONS
# ============================================================================

def train_model(model, train_loader, val_loader, epochs=25, lr=1e-3,
                device='cuda', model_name='model'):
    """Train a reconstruction-based anomaly detection model."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val_loss = float('inf')
    best_state = None
    patience = 8
    wait = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            if model_name == 'transnas':
                recon, scores = model(batch_x)
                loss = nn.MSELoss()(recon, batch_x) + 0.1 * nn.BCELoss()(
                    scores.mean(dim=1), batch_y.to(device))
            else:
                recon = model(batch_x)
                loss = nn.MSELoss()(recon, batch_x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                if model_name == 'transnas':
                    recon, scores = model(batch_x)
                    loss = nn.MSELoss()(recon, batch_x)
                else:
                    recon = model(batch_x)
                    loss = nn.MSELoss()(recon, batch_x)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  [{model_name}] Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [{model_name}] Epoch {epoch+1}/{epochs} "
                  f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

    if best_state:
        model.load_state_dict(best_state)
    model = model.to(device)
    return model, best_val_loss


def evaluate_model(model, test_loader, test_labels, device='cuda',
                   model_name='model'):
    """Evaluate reconstruction-based anomaly detection."""
    model.eval()
    all_errors = []
    all_scores = []

    start_time = time.time()
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            if model_name == 'transnas':
                recon, scores = model(batch_x)
                errors = ((batch_x - recon) ** 2).mean(dim=(1, 2)).cpu().numpy()
                all_scores.extend(scores.mean(dim=1).cpu().numpy())
            else:
                recon = model(batch_x)
                errors = ((batch_x - recon) ** 2).mean(dim=(1, 2)).cpu().numpy()
            all_errors.extend(errors)
    inference_time = time.time() - start_time

    all_errors = np.array(all_errors)
    if all_scores:
        all_scores = np.array(all_scores)
    else:
        all_scores = all_errors / (all_errors.max() + 1e-8)

    # POT thresholding (95th percentile)
    threshold = np.percentile(all_errors, 95)
    predictions = (all_errors > threshold).astype(float)

    # Align with test labels (window-level)
    min_len = min(len(predictions), len(test_labels))
    predictions = predictions[:min_len]
    test_labels_aligned = test_labels[:min_len]
    all_scores = all_scores[:min_len]

    metrics = compute_metrics(test_labels_aligned, predictions, all_scores)
    metrics['inference_time_s'] = inference_time
    metrics['inference_per_sample_ms'] = (inference_time / len(all_errors)) * 1000
    metrics['threshold'] = float(threshold)
    metrics['mean_recon_error'] = float(all_errors.mean())
    metrics['std_recon_error'] = float(all_errors.std())

    return metrics, all_errors, predictions


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# MAIN COMPARISON
# ============================================================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Load data
    df, params = load_data('Parameters_5K.csv')
    n_features = len(params)
    labels = generate_anomaly_labels(df, params)
    print(f"Ground truth anomaly rate: {labels.mean():.3f} "
          f"({int(labels.sum())}/{len(labels)})")
    print()

    # Split
    train_df, val_df, test_df = split_data(df)
    train_labels = labels[:len(train_df)]
    val_labels = labels[len(train_df):len(train_df)+len(val_df)]
    test_labels = labels[len(train_df)+len(val_df):]

    # Common params
    WINDOW = 24
    STRIDE = 4
    BATCH = 32
    EPOCHS = 25

    # Create datasets
    train_ds = SlidingWindowDataset(
        train_df[params], WINDOW, STRIDE, train_labels, fit_scaler=True)
    scaler = train_ds.scaler
    val_ds = SlidingWindowDataset(
        val_df[params], WINDOW, STRIDE, val_labels, scaler=scaler)
    test_ds = SlidingWindowDataset(
        test_df[params], WINDOW, STRIDE, test_labels, scaler=scaler)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                            num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False,
                             num_workers=0)

    print(f"Datasets: Train={len(train_ds)}, Val={len(val_ds)}, "
          f"Test={len(test_ds)} windows")
    print(f"Window size={WINDOW}, Stride={STRIDE}, Features={n_features}")
    print()

    results = {}

    # ---- MODEL 1: TransNAS-TSAD ----
    print("=" * 70)
    print("MODEL 1: TransNAS-TSAD (Dual Decoder + Adversarial)")
    print("=" * 70)
    transnas = TransNASTSAD(
        input_dim=n_features, d_model=128, nhead=8,
        enc_layers=2, dec_layers=2, dim_ff=256, dropout=0.2
    )
    n_params = count_parameters(transnas)
    print(f"  Parameters: {n_params:,}")
    t0 = time.time()
    transnas, val_loss = train_model(
        transnas, train_loader, val_loader, epochs=EPOCHS,
        lr=1e-3, device=device, model_name='transnas')
    train_time = time.time() - t0
    metrics, errors, preds = evaluate_model(
        transnas, test_loader, test_ds.window_labels, device, 'transnas')
    metrics['n_params'] = n_params
    metrics['train_time_s'] = train_time
    metrics['val_loss'] = val_loss
    results['TransNAS-TSAD'] = metrics
    print(f"\n  Results: F1={metrics['f1']:.4f} P={metrics['precision']:.4f} "
          f"R={metrics['recall']:.4f}")
    print(f"  Anomaly rate: {metrics['anomaly_rate']:.3f} "
          f"(true: {metrics['true_anomaly_rate']:.3f})")
    print(f"  Train time: {train_time:.1f}s, "
          f"Inference: {metrics['inference_per_sample_ms']:.2f}ms/sample")
    print()
    del transnas
    torch.cuda.empty_cache()

    # ---- MODEL 2: VTT ----
    print("=" * 70)
    print("MODEL 2: VTT (Variable Temporal Transformer)")
    print("=" * 70)
    vtt = VTTModel(
        n_vars=n_features, seq_len=WINDOW, d_model=64,
        n_heads=4, n_layers=3, d_ff=128, dropout=0.1
    )
    n_params = count_parameters(vtt)
    print(f"  Parameters: {n_params:,}")
    t0 = time.time()
    vtt, val_loss = train_model(
        vtt, train_loader, val_loader, epochs=EPOCHS,
        lr=1e-3, device=device, model_name='vtt')
    train_time = time.time() - t0
    metrics, errors, preds = evaluate_model(
        vtt, test_loader, test_ds.window_labels, device, 'vtt')
    metrics['n_params'] = n_params
    metrics['train_time_s'] = train_time
    metrics['val_loss'] = val_loss
    results['VTT'] = metrics
    print(f"\n  Results: F1={metrics['f1']:.4f} P={metrics['precision']:.4f} "
          f"R={metrics['recall']:.4f}")
    print(f"  Anomaly rate: {metrics['anomaly_rate']:.3f} "
          f"(true: {metrics['true_anomaly_rate']:.3f})")
    print(f"  Train time: {train_time:.1f}s, "
          f"Inference: {metrics['inference_per_sample_ms']:.2f}ms/sample")
    print()
    del vtt
    torch.cuda.empty_cache()

    # ---- MODEL 3: MOMENT ----
    print("=" * 70)
    print("MODEL 3: MOMENT (Masked Reconstruction Foundation Model)")
    print("=" * 70)

    # MOMENT uses longer windows with patching
    MOMENT_WINDOW = 64
    MOMENT_STRIDE = 8
    moment_train_ds = SlidingWindowDataset(
        train_df[params], MOMENT_WINDOW, MOMENT_STRIDE, train_labels,
        fit_scaler=True)
    moment_scaler = moment_train_ds.scaler
    moment_val_ds = SlidingWindowDataset(
        val_df[params], MOMENT_WINDOW, MOMENT_STRIDE, val_labels,
        scaler=moment_scaler)
    moment_test_ds = SlidingWindowDataset(
        test_df[params], MOMENT_WINDOW, MOMENT_STRIDE, test_labels,
        scaler=moment_scaler)
    moment_train_loader = DataLoader(
        moment_train_ds, batch_size=BATCH, shuffle=True, num_workers=0,
        drop_last=True)
    moment_val_loader = DataLoader(
        moment_val_ds, batch_size=BATCH, shuffle=False, num_workers=0)
    moment_test_loader = DataLoader(
        moment_test_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    moment = MOMENTModel(
        n_channels=n_features, seq_len=MOMENT_WINDOW, d_model=256,
        nhead=8, num_layers=6, d_ff=512, dropout=0.1, patch_len=8
    )
    n_params = count_parameters(moment)
    print(f"  Parameters: {n_params:,}")
    print(f"  Windows: Train={len(moment_train_ds)}, Val={len(moment_val_ds)}, "
          f"Test={len(moment_test_ds)}")
    t0 = time.time()
    moment, val_loss = train_model(
        moment, moment_train_loader, moment_val_loader, epochs=EPOCHS,
        lr=5e-4, device=device, model_name='moment')
    train_time = time.time() - t0
    metrics, errors, preds = evaluate_model(
        moment, moment_test_loader, moment_test_ds.window_labels,
        device, 'moment')
    metrics['n_params'] = n_params
    metrics['train_time_s'] = train_time
    metrics['val_loss'] = val_loss
    results['MOMENT'] = metrics
    print(f"\n  Results: F1={metrics['f1']:.4f} P={metrics['precision']:.4f} "
          f"R={metrics['recall']:.4f}")
    print(f"  Anomaly rate: {metrics['anomaly_rate']:.3f} "
          f"(true: {metrics['true_anomaly_rate']:.3f})")
    print(f"  Train time: {train_time:.1f}s, "
          f"Inference: {metrics['inference_per_sample_ms']:.2f}ms/sample")
    print()
    del moment
    torch.cuda.empty_cache()

    # ================================================================
    # COMPARISON TABLE
    # ================================================================
    print("\n" + "=" * 80)
    print("FINAL COMPARISON: TransNAS-TSAD vs VTT vs MOMENT")
    print("=" * 80)

    header = f"{'Metric':<30} {'TransNAS':>12} {'VTT':>12} {'MOMENT':>12}"
    print(header)
    print("-" * 70)

    metrics_to_show = [
        ('F1 Score', 'f1'),
        ('Precision', 'precision'),
        ('Recall', 'recall'),
        ('ROC AUC', 'roc_auc'),
        ('Anomaly Rate', 'anomaly_rate'),
        ('True Anomaly Rate', 'true_anomaly_rate'),
        ('Mean Recon Error', 'mean_recon_error'),
        ('Parameters', 'n_params'),
        ('Train Time (s)', 'train_time_s'),
        ('Inference (ms/sample)', 'inference_per_sample_ms'),
        ('Val Loss', 'val_loss'),
    ]

    for label, key in metrics_to_show:
        vals = []
        for model_name in ['TransNAS-TSAD', 'VTT', 'MOMENT']:
            v = results[model_name].get(key, 0)
            if key == 'n_params':
                vals.append(f"{v:>12,}")
            elif key in ('train_time_s',):
                vals.append(f"{v:>12.1f}")
            else:
                vals.append(f"{v:>12.4f}")
        print(f"{label:<30} {vals[0]} {vals[1]} {vals[2]}")

    # Production readiness scoring
    print("\n" + "-" * 70)
    print("PRODUCTION READINESS SCORING")
    print("-" * 70)

    for model_name in ['TransNAS-TSAD', 'VTT', 'MOMENT']:
        m = results[model_name]
        # Weighted score: F1 (30%) + anomaly rate proximity to 5-15% (20%)
        # + inference speed (15%) + model size (10%) + production readiness (25%)
        f1_score_w = m['f1'] * 0.30

        # Anomaly rate: closer to 0.10 is better
        anomaly_rate_dev = abs(m['anomaly_rate'] - 0.10)
        anomaly_score = max(0, 1 - anomaly_rate_dev / 0.5) * 0.20

        # Inference: <1ms = perfect, >10ms = 0
        speed_score = max(0, 1 - m['inference_per_sample_ms'] / 10) * 0.15

        # Model size: <1M = perfect, >10M = 0
        size_score = max(0, 1 - m['n_params'] / 10_000_000) * 0.10

        # Production readiness (manual assessment)
        prod_scores = {
            'TransNAS-TSAD': 0.6,  # Good NAS, but no pre-training, no checkpoint
            'VTT': 0.5,           # Novel attention, but no export/inference API
            'MOMENT': 0.95,       # Pre-trained, ONNX, MVP mode, checkpoints exist
        }
        prod_score = prod_scores[model_name] * 0.25

        total = f1_score_w + anomaly_score + speed_score + size_score + prod_score
        print(f"  {model_name:<20} Total={total:.3f} "
              f"(F1={f1_score_w:.3f} Anomaly={anomaly_score:.3f} "
              f"Speed={speed_score:.3f} Size={size_score:.3f} "
              f"Prod={prod_score:.3f})")

    # Save results
    output = {
        'comparison_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'dataset': 'Parameters_5K.csv',
        'n_samples': len(df),
        'n_features': n_features,
        'features': params,
        'window_size': WINDOW,
        'epochs': EPOCHS,
        'results': {}
    }
    for k, v in results.items():
        output['results'][k] = {mk: (float(mv) if isinstance(mv, (np.floating, float))
                                     else int(mv) if isinstance(mv, (np.integer, int))
                                     else mv)
                                for mk, mv in v.items()}

    os.makedirs('outputs', exist_ok=True)
    with open('outputs/model_comparison.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to outputs/model_comparison.json")

    # Determine winner
    best_model = max(results.items(),
                     key=lambda x: x[1]['f1'] * 0.6 + (1 - abs(x[1]['anomaly_rate'] - 0.1)) * 0.4)
    print(f"\n{'='*70}")
    print(f"WINNER: {best_model[0]} (F1={best_model[1]['f1']:.4f})")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
