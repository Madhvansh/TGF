import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import math
import io

# --- 1. Configuration and Anomaly Thresholds ---
# These settings can be tuned based on performance
class Config:
    # Data and Preprocessing
    FILE_PATH = 'Parameters_1.csv'
    # Based on the screenshot and TGF study, these are the key features
    FEATURES_TO_USE = [
        'pH', 'Conductivity', 'TDS', 'Total Hardness', 'Calcium Hardness',
        'Chlorides', 'Total Alkalinity', 'Silica', 'Total Iron'
    ]
    SEQUENCE_LENGTH = 10  # How many past time steps the model looks at
    
    # Model Hyperparameters
    INPUT_FEATURES = len(FEATURES_TO_USE)
    MODEL_DIM = 64        # Embedding dimension
    NUM_HEADS = 4         # Number of attention heads
    NUM_LAYERS = 3        # Number of VTT blocks
    DIM_FEEDFORWARD = 128 # Dimension of the feed-forward layer
    DROPOUT = 0.1

    # Training Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50       # Increase for better performance on larger datasets
    
    # Anomaly Detection
    # These ranges are from the provided screenshot
    ANOMALY_RANGES = {
        'pH': (8.3, 9.2),
        'Conductivity': (None, 3500),
        'TDS': (None, 2275),
        'Total Hardness': (None, 1000),
        'Calcium Hardness': (None, 800),
        'Chlorides': (None, 500),
        'Total Alkalinity': (None, 500),
        'Silica': (None, 150),
        'Total Iron': (None, 2.0)
    }

# --- 2. Data Loading and Preprocessing ---
def load_and_preprocess_data(config):
    """
    Loads the dataset, preprocesses it, and creates sequences for the model.
    """
    print("Loading and preprocessing data...")
    try:
        df = pd.read_csv(config.FILE_PATH)
    except FileNotFoundError:
        # Create a dummy CSV for demonstration if not found
        print(f"Warning: '{config.FILE_PATH}' not found. Creating a dummy dataset.")
        data = {
            'Date': pd.to_datetime(pd.date_range(start='2024-01-01', periods=500, freq='D')),
            'Tower': np.random.choice(['NEW CT', 'OLD CT', 'CT 850 TPD'], 500),
            'pH': np.random.normal(8.7, 0.5, 500),
            'Turbidity': np.random.uniform(10, 50, 500),
            'TSS': np.random.uniform(0, 5, 500),
            'FRC': np.random.uniform(0, 0.5, 500),
            'Conductivity': np.random.normal(3000, 500, 500),
            'TDS': np.random.normal(1950, 300, 500),
            'Total Hardness': np.random.normal(800, 200, 500),
            'Calcium Hardness': np.random.normal(600, 150, 500),
            'Magnesium Hardness': np.random.normal(200, 50, 500),
            'Chlorides': np.random.normal(400, 100, 500),
            'Ortho PO4': np.random.uniform(2, 5, 500),
            'Total Alkalinity': np.random.normal(400, 100, 500),
            'P Alkalinity': np.zeros(500),
            'Total Iron': np.random.normal(1.0, 0.5, 500),
            'SS': np.zeros(500),
            'Sulphate': np.random.uniform(100, 300, 500),
            'Silica': np.random.normal(120, 30, 500),
        }
        # Inject some anomalies
        anomaly_indices = np.random.choice(500, 50, replace=False)
        data['pH'][anomaly_indices] = np.random.uniform(9.5, 10.5, 50)
        data['Conductivity'][anomaly_indices] = np.random.uniform(3600, 4500, 50)
        df = pd.DataFrame(data)

    # --- Data Cleaning and Feature Selection ---
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values(by=['Tower', 'Date']).reset_index(drop=True)
    df_features = df[config.FEATURES_TO_USE]

    # Fill missing values using a simple forward fill
    df_features = df_features.ffill().bfill()
    df_features = df_features.astype('float64')

    # --- Data Scaling ---
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df_features)

    # --- Create Sequences ---
    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:i + seq_length]
            y = data[i + seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    X, y = create_sequences(data_scaled, config.SEQUENCE_LENGTH)

    # --- Split Data (80% train, 20% test) ---
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Store original unscaled test data for anomaly labeling and visualization
    original_data_for_labeling = df_features.iloc[split_idx + config.SEQUENCE_LENGTH:].reset_index(drop=True)

    print(f"Data ready. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test, scaler, original_data_for_labeling

# --- 3. Variable Temporal Transformer (VTT) Model Definition ---
class PositionalEncoding(nn.Module):
    """Injects positional information into the input embeddings."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class VTTBlock(nn.Module):
    """
    A single Variable Temporal Transformer block.
    As described in the paper, it consists of Temporal and Variable Attention.
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout, seq_len):
        super(VTTBlock, self).__init__()
        self.temporal_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # For variable attention, we transpose dimensions. This makes the sequence length
        # the new embedding dimension. We also need to ensure num_heads is a divisor of seq_len.
        var_nhead = nhead
        while seq_len % var_nhead != 0 and var_nhead > 1:
            var_nhead -= 1
        if seq_len % var_nhead != 0: # If no divisor > 1 is found, use 1
             var_nhead = 1

        self.variable_attn = nn.MultiheadAttention(embed_dim=seq_len, num_heads=var_nhead, dropout=dropout, batch_first=True)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # --- Temporal Attention ---
        # Attends across the time steps (sequence length)
        src_temporal, _ = self.temporal_attn(src, src, src)
        src = src + self.dropout(src_temporal)
        src = self.norm1(src)

        # --- Variable Attention ---
        # Attends across the features/variables
        # We permute to make the variable dimension the sequence dimension for attention
        src_permuted = src.permute(0, 2, 1) # (Batch, SeqLen, ModelDim) -> (Batch, ModelDim, SeqLen)
        src_variable, _ = self.variable_attn(src_permuted, src_permuted, src_permuted)
        src_variable = src_variable.permute(0, 2, 1) # Permute back
        src = src + self.dropout(src_variable)
        src = self.norm2(src)
        
        # --- Feed Forward Network ---
        src_ff = self.feed_forward(src)
        src = src + self.dropout(src_ff)
        src = self.norm3(src)
        
        return src

class VTT(nn.Module):
    """The main Variable Temporal Transformer model."""
    def __init__(self, input_features, d_model, nhead, num_layers, dim_feedforward, dropout, seq_len):
        super(VTT, self).__init__()
        self.d_model = d_model
        
        # Input embedding layer
        self.encoder = nn.Linear(input_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Stack of VTT blocks
        self.transformer_layers = nn.ModuleList([
            VTTBlock(d_model, nhead, dim_feedforward, dropout, seq_len) for _ in range(num_layers)
        ])
        
        # Output layer for reconstruction
        self.decoder = nn.Linear(d_model, input_features)

    def forward(self, src):
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        for layer in self.transformer_layers:
            src = layer(src)
            
        # We only need the output of the last time step for reconstruction of the next
        output = self.decoder(src[:, -1, :])
        return output

# --- 4. Training Function ---
def train_model(model, data_loader, config):
    """Trains the VTT model for one epoch."""
    model.train()
    total_loss = 0
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    for X_batch, y_batch in data_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(data_loader)

# --- 5. Anomaly Detection and Evaluation ---
def get_anomaly_scores(model, data_loader):
    """Calculates reconstruction error as anomaly score."""
    model.eval()
    reconstruction_errors = []
    with torch.no_grad():
        for X_batch, _ in data_loader:
            reconstructed = model(X_batch)
            error = torch.mean((X_batch[:, -1, :] - reconstructed)**2, dim=1)
            reconstruction_errors.extend(error.cpu().numpy())
    return np.array(reconstruction_errors)

def label_anomalies(original_df, config):
    """Creates ground truth labels based on operational ranges."""
    labels = pd.Series(0, index=original_df.index)
    for param, (lower, upper) in config.ANOMALY_RANGES.items():
        if lower is not None:
            labels[original_df[param] < lower] = 1
        if upper is not None:
            labels[original_df[param] > upper] = 1
    return labels.values

def plot_results(scores, labels, threshold):
    """Visualizes anomaly scores and results."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
    
    # Plot 1: Anomaly Scores
    ax1.plot(scores, label='Anomaly Score', color='blue', alpha=0.8)
    ax1.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
    ax1.set_title('Anomaly Scores vs. Time')
    ax1.set_ylabel('Reconstruction Error')
    
    # Highlight detected anomalies
    anomalies_detected = np.where(scores > threshold)[0]
    ax1.scatter(anomalies_detected, scores[anomalies_detected], color='red', label='Detected Anomaly')
    ax1.legend()

    # Plot 2: Ground Truth vs. Detections
    ax2.plot(labels, label='Ground Truth Anomaly', color='green', linestyle='-', marker='o', markersize=4, alpha=0.6)
    predictions = (scores > threshold).astype(int)
    ax2.plot(predictions, label='Model Prediction', color='red', linestyle='--', marker='x', markersize=4, alpha=0.6)
    ax2.set_title('Ground Truth vs. Model Predictions')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Anomaly (1) / Normal (0)')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# --- 6. Main Execution Block ---
def main():
    """Main function to run the entire pipeline."""
    config = Config()
    
    # Load data
    X_train, y_train, X_test, y_test, scaler, original_test_df = load_and_preprocess_data(config)
    
    # Create DataLoaders
    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    test_dataset = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Initialize and train model
    model = VTT(
        input_features=config.INPUT_FEATURES,
        d_model=config.MODEL_DIM,
        nhead=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT,
        seq_len=config.SEQUENCE_LENGTH
    )
    
    print("\nStarting model training...")
    for epoch in range(config.NUM_EPOCHS):
        loss = train_model(model, train_loader, config)
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{config.NUM_EPOCHS}, Loss: {loss:.6f}')
    print("Training complete.\n")
    
    # --- Anomaly Detection on Test Set ---
    # 1. Get anomaly scores from the training set to find a threshold
    train_eval_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    train_scores = get_anomaly_scores(model, train_eval_loader)
    
    # 2. Set threshold (e.g., 95th percentile of training scores)
    threshold = np.percentile(train_scores, 95)
    print(f"Calculated Anomaly Threshold: {threshold:.6f}")

    # 3. Get anomaly scores on the test set
    test_scores = get_anomaly_scores(model, test_loader)
    
    # 4. Generate predictions based on the threshold
    predictions = (test_scores > threshold).astype(int)
    
    # 5. Get ground truth labels
    ground_truth_labels = label_anomalies(original_test_df, config)
    
    # --- Evaluation ---
    print("\n--- Evaluation on Test Set ---")
    precision, recall, f1, _ = precision_recall_fscore_support(ground_truth_labels, predictions, average='binary')
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(ground_truth_labels, predictions)
    print(pd.DataFrame(cm, index=['Actual Normal', 'Actual Anomaly'], columns=['Predicted Normal', 'Predicted Anomaly']))
    
    # --- Visualization ---
    print("\nPlotting results...")
    plot_results(test_scores, ground_truth_labels, threshold)

if __name__ == '__main__':
    main()

