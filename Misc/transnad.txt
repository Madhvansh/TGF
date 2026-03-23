import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import optuna
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ==================== Water Quality Parameter Ranges ====================

class WaterQualityRanges:
    """Acceptable ranges for cooling tower water parameters"""
    
    PARAMETER_RANGES = {
        'pH': {'min': 7.5, 'max': 8.0, 'critical': True},
        'Turbidity': {'min': 0, 'max': 20, 'critical': False},
        'TSS': {'min': 0, 'max': 20, 'critical': False},
        'FRC': {'min': 0.2, 'max': 0.5, 'critical': True},
        'Conductivity': {'min': 0, 'max': 3000, 'critical': True},  # Typical range
        'TDS': {'min': 0, 'max': 2100, 'critical': True},
        'Total Hardness': {'min': 0, 'max': 1200, 'critical': True},
        'Calcium Hardness': {'min': 0, 'max': 800, 'critical': True},  # Typical range
        'Magnesium Hardness': {'min': 0, 'max': 400, 'critical': False},  # Typical range
        'Chlorides': {'min': 0, 'max': 500, 'critical': True},
        'Ortho PO4': {'min': 6.0, 'max': 8.0, 'critical': True},
        'Total Alkalinity': {'min': 0, 'max': 200, 'critical': True},
        'P Alkalinity': {'min': 0, 'max': 0, 'critical': False},
        'Total Iron': {'min': 0, 'max': 2, 'critical': True},
        'SS': {'min': 0, 'max': 50, 'critical': False},  # Typical range
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
        
        Args:
            data: DataFrame containing the cooling tower data
            window_size: Size of the sliding window
            stride: Stride for sliding window
            mode: 'train' or 'test' mode
            use_ranges: Whether to use range-based anomaly labels
            scaler: Optional pre-fitted scaler (required for test mode)
        """
        self.data = data
        self.window_size = window_size
        self.stride = stride
        self.mode = mode
        self.use_ranges = use_ranges
        self.scaler = scaler
        
        # Map column names to standardized names
        self.column_mapping = {
            'pH': 'pH',
            'Turbidity': 'Turbidity',
            'TSS': 'TSS',
            'FRC': 'FRC',
            'Conductivity': 'Conductivity',
            'COND': 'Conductivity',
            'TDS': 'TDS',
            'Total Hardness': 'Total Hardness',
            'TH': 'Total Hardness',
            'Calcium Hardness': 'Calcium Hardness',
            'CaH': 'Calcium Hardness',
            'Magnesium Hardness': 'Magnesium Hardness',
            'MgH': 'Magnesium Hardness',
            'Chlorides': 'Chlorides',
            'Cl': 'Chlorides',
            'Ortho PO4': 'Ortho PO4',
            'ORTHO PO4': 'Ortho PO4',
            'Total Alkalinity': 'Total Alkalinity',
            'T. Alk.': 'Total Alkalinity',
            'P Alkalinity': 'P Alkalinity',
            'P. Alk.': 'P Alkalinity',
            'Total Iron': 'Total Iron',
            'SS': 'SS',
            'Sulphate': 'Sulphate',
            'Silica': 'Silica',
            'SiO2': 'Silica'
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
        
        # Use enumerate to get sequential indices instead of DataFrame indices
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
            # Create and fit a new scaler for training data
            if self.scaler is None:
                self.scaler = StandardScaler()
            normalized_data = self.scaler.fit_transform(self.data[self.parameters].values)
        else:
            # Use the provided scaler for test/validation data
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

# ==================== Enhanced TransNAS-TSAD Model ====================

class EnhancedTransNAS_TSAD(nn.Module):
    """Enhanced model with dual pathway for reconstruction and classification"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Input dimensions
        self.input_dim = config['input_dim']
        self.window_size = config['window_size']
        self.d_model = config['dim_feedforward']
        
        # Embedding
        self.embedding = nn.Linear(self.input_dim, self.d_model)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding()
        
        # Transformer encoder
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(
                d_model=self.d_model,
                n_heads=config['num_attention_heads'],
                d_ff=config['dim_feedforward'] * 2,
                dropout=config['dropout_rate'],
                activation=config['activation_function']
            ) for _ in range(config['encoder_layers'])
        ])
        
        # Dual pathway: reconstruction and classification
        # Reconstruction decoder
        self.decoder_layers = nn.ModuleList([
            TransformerBlock(
                d_model=self.d_model,
                n_heads=config['num_attention_heads'],
                d_ff=config['dim_feedforward'] * 2,
                dropout=config['dropout_rate'],
                activation=config['activation_function']
            ) for _ in range(config['decoder_layers'])
        ])
        
        # Output layers
        self.reconstruction_output = nn.Linear(self.d_model, self.input_dim)
        
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
    
    def _create_positional_encoding(self):
        """Create positional encoding"""
        pe = torch.zeros(self.window_size, self.d_model)
        position = torch.arange(0, self.window_size).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * 
                            -(np.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def forward(self, x, return_both=False):
        batch_size, seq_len, _ = x.size()
        
        # Embedding
        x_embedded = self.embedding(x)
        x_embedded = x_embedded + self.pos_encoding[:, :seq_len, :]
        
        # Encoder
        encoder_output = x_embedded
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output)
        
        # Reconstruction pathway
        decoder_output = encoder_output
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output)
        reconstruction = self.reconstruction_output(decoder_output)
        
        # Classification pathway
        classifier_input = encoder_output.reshape(batch_size, -1)
        anomaly_score = self.classifier(classifier_input)
        
        if return_both:
            return reconstruction, anomaly_score
        return reconstruction

class TransformerBlock(nn.Module):
    """Transformer block with multi-head attention and feedforward"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, 
                 dropout: float = 0.1, activation: str = 'relu'):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feedforward network
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
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feedforward
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

# ==================== Water Quality Monitoring System ====================

class CoolingTowerMonitoringSystem:
    """Complete monitoring system for cooling tower water quality"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.model = None
        self.training_history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        self.scaler = None  # Store the scaler from training
        
    def _get_default_config(self) -> Dict:
        """Get default configuration optimized for water quality monitoring"""
        return {
            'learning_rate': 0.001,
            'dropout_rate': 0.2,
            'batch_size': 32,
            'window_size': 24,  # 24 hours of hourly data
            'dim_feedforward': 128,
            'encoder_layers': 2,
            'decoder_layers': 2,
            'activation_function': 'relu',
            'num_attention_heads': 8,
            'epochs': 100,
            'patience': 15
        }
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess cooling tower data"""
        # Load data
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
    
    def analyze_data_quality(self, data: pd.DataFrame) -> Dict:
        """Analyze data quality and parameter violations"""
        analysis = {
            'total_samples': len(data),
            'parameter_violations': {},
            'missing_data': {},
            'recommendations': []
        }
        
        # Create dataset to standardize columns
        temp_dataset = CoolingTowerDataset(data.copy(), window_size=1, use_ranges=True)
        standardized_data = temp_dataset.data
        
        for param in temp_dataset.parameters:
            if param in standardized_data.columns:
                # Check violations
                violations = []
                for idx, value in standardized_data[param].items():
                    check = WaterQualityRanges.check_parameter(param, value)
                    if check['severity'] > 1:  # Warning or critical
                        violations.append({
                            'index': idx,
                            'value': value,
                            'status': check['status'],
                            'severity': check['severity']
                        })
                
                if violations:
                    analysis['parameter_violations'][param] = {
                        'count': len(violations),
                        'percentage': (len(violations) / len(data)) * 100,
                        'samples': violations[:5]  # First 5 violations
                    }
                
                # Check missing data
                missing = standardized_data[param].isna().sum()
                if missing > 0:
                    analysis['missing_data'][param] = {
                        'count': missing,
                        'percentage': (missing / len(data)) * 100
                    }
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis['parameter_violations'])
        
        return analysis
    
    def _generate_recommendations(self, violations: Dict) -> List[str]:
        """Generate operational recommendations based on violations"""
        recommendations = []
        
        for param, violation_data in violations.items():
            if violation_data['percentage'] > 10:
                if param == 'pH':
                    recommendations.append(
                        f"Critical: pH violations in {violation_data['percentage']:.1f}% of samples. "
                        "Adjust acid/alkali dosing immediately to prevent scaling/corrosion."
                    )
                elif param == 'TDS':
                    recommendations.append(
                        f"High TDS in {violation_data['percentage']:.1f}% of samples. "
                        "Increase blowdown frequency to reduce concentration cycles."
                    )
                elif param == 'Total Hardness':
                    recommendations.append(
                        f"Hardness exceeding limits in {violation_data['percentage']:.1f}% of samples. "
                        "Review scale inhibitor dosing and consider softening makeup water."
                    )
                elif param == 'FRC':
                    recommendations.append(
                        f"Free residual chlorine outside range in {violation_data['percentage']:.1f}% of samples. "
                        "Adjust biocide program to maintain 0.2-0.5 ppm FRC."
                    )
                elif param == 'Ortho PO4':
                    recommendations.append(
                        f"Orthophosphate outside 6-8 ppm range in {violation_data['percentage']:.1f}% of samples. "
                        "Adjust corrosion inhibitor dosing."
                    )
        
        if not recommendations:
            recommendations.append("System operating within normal parameters. Continue routine monitoring.")
        
        return recommendations
    
    def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame):
        """Train the enhanced TransNAS-TSAD model"""
        
        # Create training dataset first
        train_dataset = CoolingTowerDataset(
            train_data,
            window_size=self.config['window_size'],
            mode='train',
            use_ranges=True
        )
        
        # Store the scaler from training dataset
        self.scaler = train_dataset.scaler
        
        # Create validation dataset using the same scaler
        val_dataset = CoolingTowerDataset(
            val_data,
            window_size=self.config['window_size'],
            mode='test',
            use_ranges=True,
            scaler=self.scaler  # Pass the scaler from training
        )
        
        # Update config with input dimensions
        self.config['input_dim'] = len(train_dataset.parameters)
        
        print(f"Training with {len(train_dataset.parameters)} parameters: {train_dataset.parameters}")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False)
        
        # Initialize model
        self.model = EnhancedTransNAS_TSAD(self.config)
        
        # Loss functions
        reconstruction_criterion = nn.MSELoss()
        classification_criterion = nn.BCELoss()
        
        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            # Training
            self.model.train()
            train_losses = []
            
            for batch_input, batch_labels in train_loader:
                optimizer.zero_grad()
                
                reconstruction, anomaly_scores = self.model(batch_input, return_both=True)
                
                # Combined loss
                recon_loss = reconstruction_criterion(reconstruction, batch_input)
                class_loss = classification_criterion(anomaly_scores, batch_labels)
                total_loss = recon_loss + class_loss
                
                total_loss.backward()
                optimizer.step()
                train_losses.append(total_loss.item())
            
            # Validation
            self.model.eval()
            val_losses = []
            val_predictions = []
            val_labels = []
            
            with torch.no_grad():
                for batch_input, batch_labels in val_loader:
                    reconstruction, anomaly_scores = self.model(batch_input, return_both=True)
                    
                    recon_loss = reconstruction_criterion(reconstruction, batch_input)
                    class_loss = classification_criterion(anomaly_scores, batch_labels)
                    total_loss = recon_loss + class_loss
                    
                    val_losses.append(total_loss.item())
                    val_predictions.extend((anomaly_scores > 0.5).float().cpu().numpy())
                    val_labels.extend(batch_labels.cpu().numpy())
            
            # Calculate metrics
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            val_accuracy = np.mean(np.array(val_predictions) == np.array(val_labels))
            
            # Update history
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(avg_val_loss)
            self.training_history['val_accuracy'].append(val_accuracy)
            
            # Update scheduler
            scheduler.step(avg_val_loss)
            
            print(f"Epoch {epoch+1}/{self.config['epochs']}: "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, "
                  f"Val Accuracy: {val_accuracy:.2%}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_cooling_tower_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= self.config['patience']:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_cooling_tower_model.pth'))
        print("Training completed!")
    
    def detect_anomalies(self, test_data: pd.DataFrame) -> Dict:
        """Detect anomalies in test data"""
        
        # Create test dataset using the stored scaler
        test_dataset = CoolingTowerDataset(
            test_data,
            window_size=self.config['window_size'],
            mode='test',
            use_ranges=True,
            scaler=self.scaler  # Use the scaler from training
        )
        
        test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'], shuffle=False)
        
        # Get predictions
        self.model.eval()
        all_reconstructions = []
        all_anomaly_scores = []
        all_labels = []
        
        with torch.no_grad():
            for batch_input, batch_labels in test_loader:
                reconstruction, anomaly_scores = self.model(batch_input, return_both=True)
                
                # Calculate reconstruction errors
                recon_errors = torch.mean((batch_input - reconstruction) ** 2, dim=(1, 2))
                
                all_reconstructions.extend(recon_errors.cpu().numpy())
                all_anomaly_scores.extend(anomaly_scores.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        # Combine scores
        combined_scores = np.array(all_reconstructions) * 0.5 + np.array(all_anomaly_scores).flatten() * 0.5
        
        # Determine threshold (using 95th percentile of training data reconstruction errors)
        threshold = np.percentile(combined_scores, 95)
        
        # Detect anomalies
        anomalies = combined_scores > threshold
        
        # Get parameter-specific violations
        parameter_violations = self._analyze_parameter_violations(test_data)
        
        results = {
            'anomaly_scores': combined_scores,
            'anomalies': anomalies,
            'anomaly_rate': np.mean(anomalies),
            'ground_truth_labels': np.array(all_labels),
            'accuracy': np.mean(anomalies == (np.array(all_labels) > 0)),
            'parameter_violations': parameter_violations,
            'recommendations': self._generate_operational_recommendations(parameter_violations, anomalies)
        }
        
        return results
    
    def _analyze_parameter_violations(self, data: pd.DataFrame) -> Dict:
        """Analyze which parameters are violating their ranges"""
        violations = {}
        
        temp_dataset = CoolingTowerDataset(data.copy(), window_size=1, use_ranges=True)
        
        for param in temp_dataset.parameters:
            param_violations = []
            for idx, value in temp_dataset.data[param].items():
                check = WaterQualityRanges.check_parameter(param, value)
                if check['severity'] > 0:
                    param_violations.append({
                        'index': idx,
                        'value': value,
                        'severity': check['severity'],
                        'status': check['status']
                    })
            
            if param_violations:
                violations[param] = param_violations
        
        return violations
    
    def _generate_operational_recommendations(self, violations: Dict, anomalies: np.ndarray) -> List[str]:
        """Generate specific operational recommendations"""
        recommendations = []
        
        # Count critical violations
        critical_params = []
        for param, viol_list in violations.items():
            critical_count = sum(1 for v in viol_list if v['severity'] >= 2)
            if critical_count > 0:
                critical_params.append((param, critical_count))
        
        # Sort by criticality
        critical_params.sort(key=lambda x: x[1], reverse=True)
        
        # Generate recommendations
        for param, count in critical_params[:5]:  # Top 5 critical parameters
            if param == 'pH':
                recommendations.append(f"⚠️ pH violations detected ({count} instances). Adjust acid/alkali dosing.")
            elif param == 'TDS':
                recommendations.append(f"⚠️ High TDS detected ({count} instances). Increase blowdown rate.")
            elif param == 'Total Hardness':
                recommendations.append(f"⚠️ Hardness exceeding limits ({count} instances). Check scale inhibitor dosing.")
            elif param == 'Chlorides':
                recommendations.append(f"⚠️ High chloride levels ({count} instances). Risk of pitting corrosion.")
            elif param == 'FRC':
                recommendations.append(f"⚠️ Biocide levels outside range ({count} instances). Adjust biocide program.")
        
        # Overall system health
        anomaly_rate = np.mean(anomalies)
        if anomaly_rate > 0.3:
            recommendations.insert(0, "🚨 CRITICAL: System showing >30% anomalous readings. Immediate attention required.")
        elif anomaly_rate > 0.15:
            recommendations.insert(0, "⚠️ WARNING: System showing elevated anomaly rate. Review all chemical programs.")
        else:
            recommendations.insert(0, "✅ System operating within acceptable parameters.")
        
        return recommendations
    
    def visualize_results(self, data: pd.DataFrame, results: Dict):
        """Visualize monitoring results"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # 1. Anomaly scores over time
        axes[0, 0].plot(results['anomaly_scores'], label='Anomaly Score', alpha=0.7)
        axes[0, 0].axhline(y=np.percentile(results['anomaly_scores'], 95), 
                          color='r', linestyle='--', label='Threshold')
        axes[0, 0].scatter(np.where(results['anomalies'])[0], 
                          results['anomaly_scores'][results['anomalies']], 
                          color='red', s=20, label='Detected Anomalies')
        axes[0, 0].set_title('Anomaly Detection Results')
        axes[0, 0].set_xlabel('Sample Index')
        axes[0, 0].set_ylabel('Anomaly Score')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Parameter violations heatmap
        temp_dataset = CoolingTowerDataset(data.copy(), window_size=1, use_ranges=True)
        params_to_plot = temp_dataset.parameters[:8]  # Top 8 parameters
        
        violation_matrix = np.zeros((len(params_to_plot), min(100, len(data))))
        for i, param in enumerate(params_to_plot):
            for j in range(min(100, len(data))):
                if param in temp_dataset.data.columns:
                    check = WaterQualityRanges.check_parameter(param, temp_dataset.data[param].iloc[j])
                    violation_matrix[i, j] = check['severity']
        
        im = axes[0, 1].imshow(violation_matrix, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=3)
        axes[0, 1].set_yticks(range(len(params_to_plot)))
        axes[0, 1].set_yticklabels(params_to_plot)
        axes[0, 1].set_title('Parameter Violation Severity (First 100 samples)')
        axes[0, 1].set_xlabel('Sample Index')
        plt.colorbar(im, ax=axes[0, 1], label='Severity (0=Normal, 3=Critical)')
        
        # 3. Critical parameters bar chart
        violation_counts = {}
        for param, viol_list in results['parameter_violations'].items():
            violation_counts[param] = len(viol_list)
        
        if violation_counts:
            sorted_params = sorted(violation_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            params, counts = zip(*sorted_params)
            
            axes[1, 0].bar(range(len(params)), counts, color='coral')
            axes[1, 0].set_xticks(range(len(params)))
            axes[1, 0].set_xticklabels(params, rotation=45, ha='right')
            axes[1, 0].set_title('Top 10 Parameters with Violations')
            axes[1, 0].set_ylabel('Number of Violations')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Training history (if available)
        if hasattr(self, 'training_history') and self.training_history['train_loss']:
            axes[1, 1].plot(self.training_history['train_loss'], label='Train Loss', alpha=0.7)
            axes[1, 1].plot(self.training_history['val_loss'], label='Val Loss', alpha=0.7)
            axes[1, 1].set_title('Training History')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # 5. pH and TDS trends (critical parameters)
        if 'pH' in temp_dataset.data.columns:
            axes[2, 0].plot(temp_dataset.data['pH'].iloc[:200], label='pH', color='blue', alpha=0.7)
            axes[2, 0].axhline(y=7.5, color='g', linestyle='--', alpha=0.5, label='Min Limit')
            axes[2, 0].axhline(y=8.0, color='r', linestyle='--', alpha=0.5, label='Max Limit')
            axes[2, 0].set_title('pH Trend (First 200 samples)')
            axes[2, 0].set_xlabel('Sample Index')
            axes[2, 0].set_ylabel('pH')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
        
        if 'TDS' in temp_dataset.data.columns:
            axes[2, 1].plot(temp_dataset.data['TDS'].iloc[:200], label='TDS', color='brown', alpha=0.7)
            axes[2, 1].axhline(y=2100, color='r', linestyle='--', alpha=0.5, label='Max Limit')
            axes[2, 1].set_title('TDS Trend (First 200 samples)')
            axes[2, 1].set_xlabel('Sample Index')
            axes[2, 1].set_ylabel('TDS (ppm)')
            axes[2, 1].legend()
            axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, data: pd.DataFrame, results: Dict) -> str:
        """Generate comprehensive monitoring report"""
        report = """
================================================================================
                    COOLING TOWER WATER QUALITY MONITORING REPORT
================================================================================

EXECUTIVE SUMMARY
-----------------
Analysis Date: {}
Total Samples Analyzed: {}
Anomaly Detection Rate: {:.2%}
Model Accuracy: {:.2%}

SYSTEM STATUS
-------------
{}

CRITICAL PARAMETERS REQUIRING ATTENTION
----------------------------------------
""".format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            len(results['anomaly_scores']),
            results['anomaly_rate'],
            results.get('accuracy', 0),
            results['recommendations'][0] if results['recommendations'] else "System Status Unknown"
        )
        
        # Add parameter violations
        for param, violations in results['parameter_violations'].items():
            if len(violations) > 5:  # Only show parameters with significant violations
                severity_counts = {'1': 0, '2': 0, '3': 0}
                for v in violations:
                    severity_counts[str(v['severity'])] = severity_counts.get(str(v['severity']), 0) + 1
                
                report += f"\n{param}:"
                report += f"\n  - Total Violations: {len(violations)}"
                report += f"\n  - Critical: {severity_counts['3']}, Warning: {severity_counts['2']}, Minor: {severity_counts['1']}"
        
        report += "\n\nOPERATIONAL RECOMMENDATIONS\n"
        report += "-" * 30 + "\n"
        
        for i, rec in enumerate(results['recommendations'], 1):
            report += f"{i}. {rec}\n"
        
        report += "\n" + "=" * 80 + "\n"
        
        return report

# ==================== Main Execution ====================

def main():
    """Main execution function"""
    
    # Initialize system
    system = CoolingTowerMonitoringSystem()
    
    # Load your data
    # Replace with your actual file path
    data = system.load_data('Parameters_1.csv')  # Update with your file path
    
    # Analyze data quality
    print("\nAnalyzing data quality...")
    quality_analysis = system.analyze_data_quality(data)
    
    print(f"\nData Quality Analysis:")
    print(f"Total samples: {quality_analysis['total_samples']}")
    print(f"Parameters with violations: {len(quality_analysis['parameter_violations'])}")
    
    for param, info in quality_analysis['parameter_violations'].items():
        print(f"  - {param}: {info['count']} violations ({info['percentage']:.1f}%)")
    
    print("\nInitial Recommendations:")
    for rec in quality_analysis['recommendations']:
        print(f"  • {rec}")
    
    # Split data for training/validation/testing
    train_size = int(0.7 * len(data))
    val_size = int(0.15 * len(data))
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    print(f"\nData split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Train model
    print("\nTraining model...")
    system.train(train_data, val_data)
    
    # Detect anomalies
    print("\nDetecting anomalies in test data...")
    results = system.detect_anomalies(test_data)
    
    # Generate report
    report = system.generate_report(test_data, results)
    print(report)
    
    # Save report
    with open('cooling_tower_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Visualize results
    system.visualize_results(test_data, results)
    
    print("\nAnalysis complete! Report saved as 'cooling_tower_report.txt'")

if __name__ == "__main__":
    main()