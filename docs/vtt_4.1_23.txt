import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==================== Water Quality Parameter Ranges ====================
class WaterQualityRanges:
    """Acceptable ranges for cooling tower water parameters based on provided specifications"""
    
    PARAMETER_RANGES = {
        'pH': {'min': 7.5, 'max': 8.0, 'critical': True},
        'Turbidity': {'min': 0, 'max': 20, 'critical': False},  # NTU
        'TSS': {'min': 0, 'max': 20, 'critical': False},  # PPM
        'FRC': {'min': 0.2, 'max': 0.5, 'critical': True},  # ppm
        'Conductivity': {'min': 0, 'max': 3000, 'critical': True},  # umho/cm
        'TDS': {'min': 0, 'max': 2100, 'critical': True},  # ppm
        'Total Hardness': {'min': 0, 'max': 1200, 'critical': True},  # ppm
        'Calcium Hardness': {'min': 0, 'max': 800, 'critical': True},  # ppm
        'Magnesium Hardness': {'min': 0, 'max': 400, 'critical': False},  # ppm
        'Chlorides': {'min': 0, 'max': 500, 'critical': True},  # ppm
        'Ortho PO4': {'min': 6.0, 'max': 8.0, 'critical': True},  # ppm
        'Total Alkalinity': {'min': 0, 'max': 200, 'critical': True},  # ppm
        'P Alkalinity': {'min': 0, 'max': 0, 'critical': False},  # ppm
        'Total Iron': {'min': 0, 'max': 2, 'critical': True},  # ppm
        'SS': {'min': 0, 'max': 50, 'critical': False},  # ppm
        'Sulphate': {'min': 0, 'max': 1000, 'critical': True},  # ppm
        'Silica': {'min': 0, 'max': 180, 'critical': True}  # ppm
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

# ==================== Dilated Causal Convolution for Embedding ====================
class DilatedCausalConvolution(nn.Module):
    """Dilated causal convolution for multi-resolution embedding"""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )
    
    def forward(self, x):
        # x: [batch, channels, time]
        x = self.conv(x)
        # Remove future values (causal)
        x = x[:, :, :x.size(2) - self.padding]
        return x

# ==================== Variable Temporal Attention ====================
class VariableTemporalAttention(nn.Module):
    """Combined Variable and Temporal Self-Attention mechanism"""
    
    def __init__(self, d_model, n_heads, seq_len, n_vars, dropout=0.1, mode='serial'):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.n_vars = n_vars
        self.mode = mode  # 'serial' or 'parallel'
        
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        
        # Temporal attention projections
        self.W_Q_temp = nn.Linear(d_model * n_vars, d_model * n_vars)
        self.W_K_temp = nn.Linear(d_model * n_vars, d_model * n_vars)
        self.W_V_temp = nn.Linear(d_model * n_vars, d_model * n_vars)
        
        # Variable attention projections  
        self.W_Q_var = nn.Linear(d_model * seq_len, d_model * seq_len)
        self.W_K_var = nn.Linear(d_model * seq_len, d_model * seq_len)
        self.W_V_var = nn.Linear(d_model * seq_len, d_model * seq_len)
        
        self.dropout = nn.Dropout(dropout)
        
        if mode == 'parallel':
            self.output_linear = nn.Linear(d_model * 2, d_model)
    
    def temporal_attention(self, x):
        """Temporal self-attention across time dimension"""
        batch_size, seq_len, n_vars, d_model = x.size()
        
        # Reshape for temporal attention: [batch, seq_len, n_vars * d_model]
        x_temp = x.view(batch_size, seq_len, -1)
        
        # Generate Q, K, V
        Q = self.W_Q_temp(x_temp)
        K = self.W_K_temp(x_temp)
        V = self.W_V_temp(x_temp)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, -1).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, -1).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, -1).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(n_vars * d_model)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # Reshape back
        output = context.view(batch_size, seq_len, n_vars, d_model)
        
        return output, attn_weights
    
    def variable_attention(self, x):
        """Variable self-attention across variable dimension"""
        batch_size, seq_len, n_vars, d_model = x.size()
        
        # Reshape for variable attention: [batch, n_vars, seq_len * d_model]
        x_var = x.transpose(1, 2).contiguous().view(batch_size, n_vars, -1)
        
        # Generate Q, K, V
        Q = self.W_Q_var(x_var)
        K = self.W_K_var(x_var)
        V = self.W_V_var(x_var)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, n_vars, self.n_heads, -1).transpose(1, 2)
        K = K.view(batch_size, n_vars, self.n_heads, -1).transpose(1, 2)
        V = V.view(batch_size, n_vars, self.n_heads, -1).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(seq_len * d_model)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, n_vars, -1)
        
        # Reshape back
        output = context.view(batch_size, n_vars, seq_len, d_model)
        output = output.transpose(1, 2).contiguous()
        
        return output, attn_weights
    
    def forward(self, x):
        if self.mode == 'serial':
            # Variable attention first, then temporal
            var_out, var_weights = self.variable_attention(x)
            final_out, temp_weights = self.temporal_attention(var_out)
            return final_out, (temp_weights, var_weights)
        else:  # parallel
            temp_out, temp_weights = self.temporal_attention(x)
            var_out, var_weights = self.variable_attention(x)
            # Concatenate and project
            combined = torch.cat([temp_out, var_out], dim=-1)
            final_out = self.output_linear(combined)
            return final_out, (temp_weights, var_weights)

# ==================== VTT Encoder Layer ====================
class VTTEncoderLayer(nn.Module):
    """Single encoder layer of Variable Temporal Transformer"""
    
    def __init__(self, d_model, n_heads, seq_len, n_vars, d_ff=None, dropout=0.1, mode='serial'):
        super().__init__()
        
        if d_ff is None:
            d_ff = 4 * d_model
        
        self.attention = VariableTemporalAttention(
            d_model, n_heads, seq_len, n_vars, dropout, mode
        )
        self.norm1 = nn.LayerNorm([seq_len, n_vars, d_model])
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm([seq_len, n_vars, d_model])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: [batch, seq_len, n_vars, d_model]
        
        # Variable & Temporal Attention with residual
        attn_out, attn_weights = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x, attn_weights

# ==================== Complete VTT Model ====================
class VariableTemporalTransformer(nn.Module):
    """Complete Variable Temporal Transformer for water quality anomaly detection"""
    
    def __init__(self, n_vars, seq_len, d_model=128, n_heads=8, n_layers=3, 
                 d_ff=None, dropout=0.1, mode='serial'):
        super().__init__()
        
        self.n_vars = n_vars
        self.seq_len = seq_len
        self.d_model = d_model
        
        # Multi-resolution embedding using dilated causal convolutions
        self.embedding_layers = nn.ModuleList([
            DilatedCausalConvolution(1, d_model // 3, kernel_size=4, dilation=1),
            DilatedCausalConvolution(1, d_model // 3, kernel_size=8, dilation=2),
            DilatedCausalConvolution(1, d_model - 2 * (d_model // 3), kernel_size=16, dilation=3)
        ])
        
        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(seq_len, d_model)
        
        # VTT encoder layers
        self.encoder_layers = nn.ModuleList([
            VTTEncoderLayer(d_model, n_heads, seq_len, n_vars, d_ff, dropout, mode)
            for _ in range(n_layers)
        ])
        
        # Output projection for reconstruction
        self.output_projection = nn.Linear(d_model, 1)
        
        self.dropout = nn.Dropout(dropout)
    
    def _create_positional_encoding(self, seq_len, d_model):
        """Create positional encoding"""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def embed(self, x):
        """Multi-resolution embedding"""
        batch_size, seq_len, n_vars = x.size()
        
        embedded = []
        for var_idx in range(n_vars):
            var_data = x[:, :, var_idx:var_idx+1].transpose(1, 2)  # [batch, 1, seq_len]
            
            var_embeddings = []
            for conv_layer in self.embedding_layers:
                var_embeddings.append(conv_layer(var_data))
            
            # Concatenate multi-resolution embeddings
            var_embedded = torch.cat(var_embeddings, dim=1)  # [batch, d_model, seq_len]
            var_embedded = var_embedded.transpose(1, 2)  # [batch, seq_len, d_model]
            embedded.append(var_embedded)
        
        # Stack all variables
        embedded = torch.stack(embedded, dim=2)  # [batch, seq_len, n_vars, d_model]
        
        # Add positional encoding
        embedded = embedded + self.positional_encoding[:, :seq_len, :].unsqueeze(2)
        
        return self.dropout(embedded)
    
    def forward(self, x, return_attention=False):
        """
        Forward pass
        x: [batch, seq_len, n_vars]
        """
        # Embedding
        x = self.embed(x)
        
        # Pass through encoder layers
        attention_weights = []
        for layer in self.encoder_layers:
            x, attn = layer(x)
            attention_weights.append(attn)
        
        # Project to output
        output = self.output_projection(x)  # [batch, seq_len, n_vars, 1]
        output = output.squeeze(-1)  # [batch, seq_len, n_vars]
        
        if return_attention:
            return output, attention_weights
        return output

# ==================== Dataset Class ====================
class CoolingTowerDataset(Dataset):
    """Dataset for cooling tower water quality monitoring"""
    
    def __init__(self, data, window_size=100, stride=50, mode='train', scaler=None):
        """
        data: DataFrame with water quality parameters
        window_size: length of time window
        stride: step size for sliding window
        mode: 'train' or 'test'
        scaler: pre-fitted scaler for normalization (required for test/val mode)
        """
        self.data = data
        self.window_size = window_size
        self.stride = stride
        self.mode = mode
        
        # Select water quality parameters
        self.parameters = [col for col in data.columns 
                          if col in WaterQualityRanges.PARAMETER_RANGES.keys()]
        
        # Prepare data
        self.values = data[self.parameters].values
        
        # Normalize
        self.scaler = scaler or StandardScaler()
        if mode == 'train':
            self.normalized_values = self.scaler.fit_transform(self.values)
        else:
            if scaler is None:
                raise ValueError("Scaler must be provided for validation/test mode")
            self.normalized_values = self.scaler.transform(self.values)
        
        # Create windows
        self.windows = []
        self.labels = []
        self.original_windows = []
        
        for i in range(0, len(self.normalized_values) - window_size + 1, stride):
            window = self.normalized_values[i:i + window_size]
            original_window = self.values[i:i + window_size]
            
            self.windows.append(window)
            self.original_windows.append(original_window)
            
            # Generate labels based on parameter ranges
            window_labels = self._generate_labels(original_window)
            self.labels.append(window_labels)
    
    def _generate_labels(self, window):
        """Generate anomaly labels based on parameter ranges"""
        labels = np.zeros(len(window))
        
        for i, row in enumerate(window):
            max_severity = 0
            for j, param in enumerate(self.parameters):
                check_result = WaterQualityRanges.check_parameter(param, row[j])
                max_severity = max(max_severity, check_result['severity'])
            
            # Label as anomaly if severity > 1 (Warning or Critical)
            labels[i] = 1 if max_severity > 1 else 0
        
        return labels
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        return (torch.FloatTensor(self.windows[idx]), 
                torch.FloatTensor(self.labels[idx]),
                self.original_windows[idx])

# ==================== Training and Evaluation ====================
class VTTAnomalyDetector:
    """Complete system for training and using VTT for anomaly detection"""
    
    def __init__(self, config=None):
        self.config = config or self._get_default_config()
        self.model = None
        self.scaler = None  # Store the scaler for later use
        self.training_history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    
    def _get_default_config(self):
        return {
            'seq_len': 100,
            'd_model': 128,
            'n_heads': 8,
            'n_layers': 3,
            'd_ff': 512,
            'dropout': 0.2,
            'mode': 'serial',  # or 'parallel'
            'batch_size': 32,
            'learning_rate': 1e-4,
            'epochs': 100,
            'patience': 10,
            'window_size': 100,
            'stride': 50
        }
    
    def prepare_data(self, df):
        """Prepare data for training"""
        # Handle date column if present
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.sort_values('Date')
        
        # Handle non-numeric values
        parameter_cols = [col for col in df.columns 
                         if col in WaterQualityRanges.PARAMETER_RANGES.keys()]
        
        for col in parameter_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill missing values
        df[parameter_cols] = df[parameter_cols].interpolate(method='linear', limit=3)
        df[parameter_cols] = df[parameter_cols].fillna(method='bfill')
        df[parameter_cols] = df[parameter_cols].fillna(method='ffill')
        
        # Final fill with median
        for col in parameter_cols:
            if df[col].isna().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        return df
    
    def train(self, train_df, val_df=None):
        """Train the VTT model"""
        # Prepare datasets
        train_dataset = CoolingTowerDataset(
            train_df, 
            window_size=self.config['window_size'],
            stride=self.config['stride'],
            mode='train'
        )
        
        # Store the scaler for later use
        self.scaler = train_dataset.scaler
        
        if val_df is not None:
            val_dataset = CoolingTowerDataset(
                val_df,
                window_size=self.config['window_size'],
                stride=self.config['stride'],
                mode='test',
                scaler=self.scaler  # Use the scaler from training data
            )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True
        )
        
        if val_df is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False
            )
        
        # Initialize model
        n_vars = len(train_dataset.parameters)
        self.model = VariableTemporalTransformer(
            n_vars=n_vars,
            seq_len=self.config['seq_len'],
            d_model=self.config['d_model'],
            n_heads=self.config['n_heads'],
            n_layers=self.config['n_layers'],
            d_ff=self.config['d_ff'],
            dropout=self.config['dropout'],
            mode=self.config['mode']
        )
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            # Training
            self.model.train()
            train_losses = []
            
            for batch_data, batch_labels, _ in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                reconstruction = self.model(batch_data)
                loss = criterion(reconstruction, batch_data)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            self.training_history['train_loss'].append(avg_train_loss)
            
            # Validation
            if val_df is not None:
                self.model.eval()
                val_losses = []
                
                with torch.no_grad():
                    for batch_data, batch_labels, _ in val_loader:
                        reconstruction = self.model(batch_data)
                        loss = criterion(reconstruction, batch_data)
                        val_losses.append(loss.item())
                
                avg_val_loss = np.mean(val_losses)
                self.training_history['val_loss'].append(avg_val_loss)
                
                scheduler.step(avg_val_loss)
                
                print(f"Epoch {epoch+1}/{self.config['epochs']}: "
                      f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    torch.save(self.model.state_dict(), 'best_vtt_model.pth')
                else:
                    patience_counter += 1
                    if patience_counter >= self.config['patience']:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                print(f"Epoch {epoch+1}/{self.config['epochs']}: Train Loss: {avg_train_loss:.4f}")
        
        # Load best model
        if val_df is not None:
            self.model.load_state_dict(torch.load('best_vtt_model.pth'))
    
    def detect_anomalies(self, test_df, threshold=None):
        """Detect anomalies in test data"""
        if self.scaler is None:
            raise ValueError("Scaler not available. Please train the model first.")
            
        test_dataset = CoolingTowerDataset(
            test_df,
            window_size=self.config['window_size'],
            stride=self.config['stride'],
            mode='test',
            scaler=self.scaler  # Use the scaler from training
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False
        )
        
        self.model.eval()
        all_scores = []
        all_labels = []
        all_reconstructions = []
        all_originals = []
        
        with torch.no_grad():
            for batch_data, batch_labels, batch_originals in test_loader:
                reconstruction = self.model(batch_data)
                
                # Calculate reconstruction error
                errors = torch.mean((batch_data - reconstruction) ** 2, dim=2)
                
                all_scores.extend(errors.numpy())
                all_labels.extend(batch_labels.numpy())
                all_reconstructions.extend(reconstruction.numpy())
                all_originals.extend(batch_originals)
        
        # Convert to arrays
        all_scores = np.concatenate(all_scores)
        all_labels = np.concatenate(all_labels)
        
        # Determine threshold if not provided
        if threshold is None:
            threshold = np.percentile(all_scores, 95)
        
        # Detect anomalies
        predictions = (all_scores > threshold).astype(int)
        
        # Analyze parameter-specific violations
        parameter_violations = self._analyze_violations(all_originals, test_dataset.parameters)
        
        return {
            'scores': all_scores,
            'labels': all_labels,
            'predictions': predictions,
            'threshold': threshold,
            'parameter_violations': parameter_violations,
            'reconstructions': all_reconstructions,
            'originals': all_originals
        }
    
    def _analyze_violations(self, windows, parameters):
        """Analyze which parameters are violating their ranges"""
        violations = {param: [] for param in parameters}
        
        for window in windows:
            for t in range(len(window)):
                for i, param in enumerate(parameters):
                    check = WaterQualityRanges.check_parameter(param, window[t, i])
                    if check['severity'] > 1:  # Warning or Critical
                        violations[param].append({
                            'time': t,
                            'value': window[t, i],
                            'severity': check['severity'],
                            'status': check['status'],
                            'deviation': check['deviation']
                        })
        
        return violations
    
    def visualize_results(self, results, parameter_names):
        """Visualize anomaly detection results"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # 1. Anomaly scores over time
        axes[0, 0].plot(results['scores'], label='Anomaly Score', alpha=0.7)
        axes[0, 0].axhline(y=results['threshold'], color='r', linestyle='--', label='Threshold')
        axes[0, 0].scatter(np.where(results['predictions'])[0], 
                          results['scores'][results['predictions'] == 1], 
                          color='red', s=20, label='Detected Anomalies')
        axes[0, 0].set_title('Anomaly Detection Results')
        axes[0, 0].set_xlabel('Time Window')
        axes[0, 0].set_ylabel('Anomaly Score')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Ground truth vs predictions
        axes[0, 1].plot(results['labels'], label='Ground Truth', alpha=0.7)
        axes[0, 1].plot(results['predictions'], label='Predictions', alpha=0.7)
        axes[0, 1].set_title('Ground Truth vs Predictions')
        axes[0, 1].set_xlabel('Time Window')
        axes[0, 1].set_ylabel('Anomaly Label')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Parameter violations
        violations_count = {param: len(viols) 
                          for param, viols in results['parameter_violations'].items()}
        if violations_count:
            sorted_params = sorted(violations_count.items(), key=lambda x: x[1], reverse=True)[:10]
            params, counts = zip(*sorted_params)
            
            axes[1, 0].bar(range(len(params)), counts, color='coral')
            axes[1, 0].set_xticks(range(len(params)))
            axes[1, 0].set_xticklabels(params, rotation=45, ha='right')
            axes[1, 0].set_title('Top 10 Parameters with Violations')
            axes[1, 0].set_ylabel('Number of Violations')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Severity distribution
        all_severities = []
        for viols in results['parameter_violations'].values():
            all_severities.extend([v['severity'] for v in viols])
        
        if all_severities:
            severity_counts = np.bincount(all_severities, minlength=4)
            axes[1, 1].bar(['Normal', 'Minor', 'Warning', 'Critical'], 
                          severity_counts, color=['green', 'yellow', 'orange', 'red'])
            axes[1, 1].set_title('Violation Severity Distribution')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 5. Training history (if available)
        if hasattr(self, 'training_history') and self.training_history['train_loss']:
            axes[2, 0].plot(self.training_history['train_loss'], label='Train Loss')
            if 'val_loss' in self.training_history:
                axes[2, 0].plot(self.training_history['val_loss'], label='Val Loss')
            axes[2, 0].set_title('Training History')
            axes[2, 0].set_xlabel('Epoch')
            axes[2, 0].set_ylabel('Loss')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
        
        # 6. Reconstruction quality
        if len(results['originals']) > 0 and len(parameter_names) > 0:
            # Sample one window for visualization
            idx = 0
            original = results['originals'][idx]
            reconstruction = results['reconstructions'][idx]
            
            param_idx = min(3, len(parameter_names)-1)  # Show first 3 parameters
            for i in range(min(3, len(parameter_names))):
                axes[2, 1].plot(original[:, i], label=f'{parameter_names[i]} (Original)', 
                               alpha=0.7, linestyle='-')
                axes[2, 1].plot(reconstruction[:, i], label=f'{parameter_names[i]} (Reconstructed)', 
                               alpha=0.7, linestyle='--')
            
            axes[2, 1].set_title('Reconstruction Quality (Sample Window)')
            axes[2, 1].set_xlabel('Time Steps')
            axes[2, 1].set_ylabel('Normalized Value')
            axes[2, 1].legend()
            axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def interpret_anomalies(self, test_df):
        """Interpret anomalies using attention mechanisms"""
        if self.scaler is None:
            raise ValueError("Scaler not available. Please train the model first.")
            
        test_dataset = CoolingTowerDataset(
            test_df,
            window_size=self.config['window_size'],
            stride=self.config['stride'],
            mode='test',
            scaler=self.scaler
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,  # Process one at a time for interpretation
            shuffle=False
        )
        
        self.model.eval()
        interpretations = []
        
        with torch.no_grad():
            for batch_data, batch_labels, batch_originals in test_loader:
                # Get reconstruction and attention weights
                reconstruction, attention_weights = self.model(batch_data, return_attention=True)
                
                # Calculate reconstruction error
                error = torch.mean((batch_data - reconstruction) ** 2).item()
                
                # Extract attention patterns
                # attention_weights is a list of (temporal_attn, variable_attn) for each layer
                last_layer_attn = attention_weights[-1]
                temp_attn, var_attn = last_layer_attn
                
                # Identify most attended time steps and variables
                temp_importance = temp_attn.mean(dim=(0, 1)).squeeze()  # Average over batch and heads
                var_importance = var_attn.mean(dim=(0, 1)).squeeze()
                
                interpretation = {
                    'error': error,
                    'temporal_importance': temp_importance.numpy(),
                    'variable_importance': var_importance.numpy(),
                    'original_data': batch_originals[0],
                    'parameters': test_dataset.parameters
                }
                
                interpretations.append(interpretation)
        
        return interpretations

# ==================== Main Execution Function ====================
def main():
    """Main function to demonstrate VTT usage"""
    
    # Load your data
    print("Loading data...")
    df = pd.read_csv('Parameters_1.csv')  # Update with your file path
    
    # Initialize detector
    print("Initializing VTT Anomaly Detector...")
    detector = VTTAnomalyDetector()
    
    # Prepare data
    print("Preparing data...")
    df = detector.prepare_data(df)
    
    # Split data
    train_size = int(0.7 * len(df))
    val_size = int(0.15 * len(df))
    
    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    test_df = df[train_size + val_size:]
    
    print(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Train model
    print("\nTraining VTT model...")
    detector.train(train_df, val_df)
    
    # Detect anomalies
    print("\nDetecting anomalies...")
    results = detector.detect_anomalies(test_df)
    
    # Calculate metrics
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    precision = precision_score(results['labels'], results['predictions'])
    recall = recall_score(results['labels'], results['predictions'])
    f1 = f1_score(results['labels'], results['predictions'])
    
    print(f"\nPerformance Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Analyze violations
    print("\nParameter Violations Summary:")
    for param, violations in results['parameter_violations'].items():
        if violations:
            severities = [v['severity'] for v in violations]
            critical_count = sum(1 for s in severities if s == 3)
            warning_count = sum(1 for s in severities if s == 2)
            print(f"{param}: {len(violations)} violations "
                  f"(Critical: {critical_count}, Warning: {warning_count})")
    
    # Get parameter names for visualization
    test_dataset = CoolingTowerDataset(test_df, mode='test', scaler=detector.scaler)
    parameter_names = test_dataset.parameters
    
    # Visualize results
    print("\nVisualizing results...")
    detector.visualize_results(results, parameter_names)
    
    # Interpret anomalies
    print("\nInterpreting anomalies...")
    interpretations = detector.interpret_anomalies(test_df[:100])  # Interpret first 100 samples
    
    # Show interpretation for most anomalous window
    most_anomalous_idx = np.argmax([interp['error'] for interp in interpretations])
    most_anomalous = interpretations[most_anomalous_idx]
    
    print(f"\nMost anomalous window (Error: {most_anomalous['error']:.4f}):")
    print("Top 5 most important variables:")
    var_importance = most_anomalous['variable_importance']
    top_vars_idx = np.argsort(var_importance)[-5:]
    for idx in top_vars_idx:
        if idx < len(parameter_names):
            print(f"  - {parameter_names[idx]}: {var_importance[idx]:.4f}")
    
    print("\nTraining complete!")
    
    return detector, results

# Run the main function
if __name__ == "__main__":
    detector, results = main()