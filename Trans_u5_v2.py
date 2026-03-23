import torch
import torch.nn as nn
import math
import optuna
from optuna.samplers import NSGAIISampler
import time
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import pandas as pd
import optuna.visualization as vis
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import argparse



class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding."""
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

class FourierEncoding(nn.Module):
    """Fourier Positional Encoding."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(FourierEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Implementation for Fourier encoding can be added here if needed.
        # For simplicity, we will use a learnable embedding as a proxy for complex Fourier features.
        self.position_embedding = nn.Parameter(torch.randn(max_len, 1, d_model))

    def forward(self, x):
        x = x + self.position_embedding[:x.size(0), :]
        return self.dropout(x)

class TransNAS_TSAD_Model(nn.Module):
    def __init__(self, config):
        super(TransNAS_TSAD_Model, self).__init__()
        self.config = config
        self.feature_dim = config['feature_dim']
        self.d_model = config['d_model']
        self.window_size = config['window_size']

        # Optional Linear Embedding
        if config.get('use_linear_embedding', False):
            self.embedding = nn.Linear(self.feature_dim, self.d_model)
        else:
            # If no embedding, d_model must match feature_dim
            self.d_model = self.feature_dim
            self.embedding = nn.Identity()

        # Positional Encoding
        if config['pos_encoding_type'] == 'sinusoidal':
            self.pos_encoder = PositionalEncoding(self.d_model, config['dropout'], self.window_size)
        else: # fourier
            self.pos_encoder = FourierEncoding(self.d_model, config['dropout'], self.window_size)

        # Activation Function
        activation_map = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh()
        }
        activation_fn = activation_map[config['activation_function']]

        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config['n_heads'],
            dim_feedforward=config['d_ff'],
            dropout=config['dropout'],
            activation=activation_fn,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=config['encoder_layers'])

        # Transformer Decoder (Two for adversarial training)
        decoder_layers = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=config['n_heads'],
            dim_feedforward=config['d_ff'],
            dropout=config['dropout'],
            activation=activation_fn,
            batch_first=True
        )
        self.transformer_decoder1 = nn.TransformerDecoder(decoder_layers, num_layers=config['decoder_layers'])
        self.transformer_decoder2 = nn.TransformerDecoder(decoder_layers, num_layers=config['decoder_layers'])

        # Final output layer
        self.fc_out = nn.Linear(self.d_model, self.feature_dim)

    def forward(self, src):
        # src shape: (batch_size, window_size, feature_dim)
        
        # Embedding and Positional Encoding
        embedded_src = self.embedding(src)
        pos_encoded_src = self.pos_encoder(embedded_src.transpose(0, 1)).transpose(0, 1)

        # Encoder
        memory = self.transformer_encoder(pos_encoded_src)

        # Decoders
        # Decoder input is typically a shifted version of the target or a start token.
        # For reconstruction, we can use the encoded output itself as a starting point.
        # A simple approach is to use a zero-initialized tensor as the decoder input (tgt).
        tgt = torch.zeros_like(embedded_src)
        pos_encoded_tgt = self.pos_encoder(tgt.transpose(0,1)).transpose(0,1)

        output1 = self.transformer_decoder1(pos_encoded_tgt, memory)
        output2 = self.transformer_decoder2(pos_encoded_tgt, memory)

        # Final linear layer to project back to feature dimension
        reconstruction1 = self.fc_out(output1)
        reconstruction2 = self.fc_out(output2)

        return reconstruction1, reconstruction2, memory

# Placeholder for the AdversarialTrainer class, to be defined in the next section
class AdversarialTrainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self):
        # This is a simplified placeholder for the complex training logic.
        # The full implementation will be in the next section.
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        criterion = nn.MSELoss()
        
        self.model.train()
        for epoch in range(self.config.get('epochs', 3)): # Use a small number of epochs for NAS trials
            for batch in self.train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                recon1, recon2, _ = self.model(batch)
                loss1 = criterion(recon1, batch)
                loss2 = criterion(recon2, batch)
                loss = loss1 + loss2 # Simplified loss for placeholder
                loss.backward()
                optimizer.step()
        return self.evaluate()

    def evaluate(self):
        # Simplified evaluation placeholder
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                recon1, recon2, _ = self.model(batch)
                loss1 = torch.mean((recon1 - batch) ** 2, dim=(1, 2))
                loss2 = torch.mean((recon2 - batch) ** 2, dim=(1, 2))
                total_loss += torch.mean(loss1 + loss2).item()
        
        # In a real scenario, you'd calculate F1 score based on anomaly labels.
        # For this example, we'll use a proxy: inverse of validation loss.
        # This assumes lower reconstruction error correlates with better anomaly detection.
        # A proper implementation requires anomaly labels in the validation set.
        # We will simulate this for the NAS process.
        
        # Simulate F1 score based on loss
        simulated_f1 = 1 / (1 + total_loss) if total_loss > 0 else 1.0
        return simulated_f1

def objective(trial, train_loader, val_loader, feature_dim):
    """Optuna objective function for NAS."""
    config = {
        'feature_dim': feature_dim,
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'dropout': trial.suggest_float('dropout_rate', 0.1, 0.5),
        'batch_size': trial.suggest_categorical('batch_size', ),
        'window_size': trial.suggest_int('window_size', 10, 30),
        'pos_encoding_type': trial.suggest_categorical('pos_encoding_type', ['sinusoidal', 'fourier']),
        'd_ff': trial.suggest_int('d_ff', 8, 128, log=True),
        'encoder_layers': trial.suggest_int('encoder_layers', 1, 3),
        'decoder_layers': trial.suggest_int('decoder_layers', 1, 3),
        'activation_function': trial.suggest_categorical('activation_function', ['relu', 'leaky_relu', 'sigmoid', 'tanh']),
        'n_heads': trial.suggest_categorical('n_heads', ),
        'use_linear_embedding': trial.suggest_categorical('use_linear_embedding',),
        'phase_type': trial.suggest_categorical('phase_type', ['2phase', 'iterative']),
        # d_model depends on use_linear_embedding
    }
    
    # d_model must be divisible by n_heads. We can enforce this.
    # Let's define d_model as a multiple of n_heads.
    d_model_options = [h * d for h in  for d in ]
    d_model_options = sorted(list(set(d_model_options)))
    config['d_model'] = trial.suggest_categorical('d_model', d_model_options)
    
    # Ensure d_model is compatible with n_heads
    if config['d_model'] % config['n_heads']!= 0:
        # Prune trial if config is invalid
        raise optuna.exceptions.TrialPruned()

    # Instantiate model
    model = TransNAS_TSAD_Model(config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Train and evaluate
    start_time = time.time()
    trainer = AdversarialTrainer(model, train_loader, val_loader, config)
    f1_score = trainer.train() # This is a simulated F1 for now
    training_time = time.time() - start_time
    
    trial.set_user_attr("training_time", training_time)
    trial.set_user_attr("f1_score", f1_score)

    # Objectives: Maximize F1, Minimize Parameters
    return f1_score, n_params

def run_nas_study(train_loader, val_loader, feature_dim, n_trials=100):
    """Sets up and runs the Optuna NAS study."""
    sampler = NSGAIISampler()
    study = optuna.create_study(
        directions=['maximize', 'minimize'],
        sampler=sampler,
        study_name="TransNAS-TSAD_Optimization"
    )
    
    # The objective function needs access to data loaders and feature_dim
    objective_func = lambda trial: objective(trial, train_loader, val_loader, feature_dim)
    
    study.optimize(objective_func, n_trials=n_trials)
    
    print("Number of finished trials: ", len(study.trials))
    
    pareto_front = study.best_trials
    print("Pareto front contains {} trials:".format(len(pareto_front)))
    for trial in pareto_front:
        print(f"  Trial {trial.number}: F1={trial.values:.4f}, Params={trial.values}")
        
    return study

class AdversarialTrainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.criterion = nn.MSELoss(reduction='none') # Use 'none' for per-element loss
        self.phase_type = config['phase_type']
        self.epochs = config.get('epochs', 10) # A reasonable number for final training

    def _train_2phase_epoch(self):
        self.model.train()
        total_loss = 0
        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            # Phase 1: Preliminary Reconstruction
            recon1, recon2, _ = self.model(batch)
            loss1 = torch.mean(self.criterion(recon1, batch))
            loss2 = torch.mean(self.criterion(recon2, batch))
            
            # L_focus from paper (using recon1 for focus)
            l_focus = torch.mean(self.criterion(recon1, batch))
            
            # Phase 2: Adversarial Reconstruction
            # Decoder 1 minimizes reconstruction error
            l_adv1 = torch.mean(self.criterion(recon1, batch))
            # Decoder 2 maximizes reconstruction error (negative loss)
            l_adv2 = -torch.mean(self.criterion(recon2, batch))
            
            # Combine losses
            loss = l_focus + l_adv1 + l_adv2
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)

    def _train_iterative_epoch(self):
        self.model.train()
        total_loss = 0
        epsilon = 1e-5 # Convergence threshold
        
        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            # Iterative refinement using the first decoder/reconstruction path
            l_iteration_prev = torch.inf
            l_iteration_current = 0
            
            recon1, _, _ = self.model(batch)
            l_iteration_current = torch.mean(self.criterion(recon1, batch))
            
            # Simplified iterative loop for training context
            # A full implementation would involve more complex feedback
            # For training, we can model the objective directly
            
            # L_self-adv aims to minimize the change between iterations
            # For a single training step, we can approximate this by minimizing the reconstruction error
            # while regularizing. Here, we focus on the core reconstruction loss.
            loss = l_iteration_current
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def train(self):
        """Main training loop that selects the phase type."""
        print(f"Starting training with phase type: {self.phase_type}")
        for epoch in range(self.epochs):
            if self.phase_type == '2phase':
                epoch_loss = self._train_2phase_epoch()
            elif self.phase_type == 'iterative':
                epoch_loss = self._train_iterative_epoch()
            else: # Fallback to 2phase
                epoch_loss = self._train_2phase_epoch()

            val_f1 = self.evaluate() # Using simulated F1
            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {epoch_loss:.6f}, Val F1 (simulated): {val_f1:.4f}")
        
        # Return final validation F1 for the objective function
        return self.evaluate()

    def evaluate(self, test_labels=None, test_scores=None):
        """Evaluate model and return F1 score."""
        self.model.eval()
        
        # If true labels and scores are provided, calculate real F1
        if test_labels is not None and test_scores is not None:
            # This part is used for final evaluation, not during NAS
            detector = AnomalyDetector(self.model)
            scores = detector.compute_scores(self.val_loader)
            
            # Find threshold on validation scores (assuming val is normal)
            threshold = detector.find_pot_threshold(scores, q=0.99)
            
            # Predict on test scores
            preds = (test_scores > threshold).astype(int)
            return f1_score(test_labels, preds)

        # During NAS, we use a simulated F1 based on reconstruction loss
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                recon1, recon2, _ = self.model(batch)
                
                # Anomaly score from Eq. 7
                error1 = torch.mean(self.criterion(recon1, batch), dim=(1,2))
                error2 = torch.mean(self.criterion(recon2, batch), dim=(1,2))
                scores = 0.5 * error1 + 0.5 * error2
                
                total_loss += torch.mean(scores).item()
        
        avg_loss = total_loss / len(self.val_loader)
        simulated_f1 = 1 / (1 + avg_loss) if avg_loss > 0 else 1.0
        return simulated_f1
    
import pandas as pd

class AnomalyDetector:
    def __init__(self, model):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = nn.MSELoss(reduction='none')

    def compute_scores(self, data_loader):
        """Compute anomaly scores for a given dataset."""
        self.model.eval()
        scores =
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                recon1, recon2, _ = self.model(batch)
                
                error1 = torch.mean(self.criterion(recon1, batch), dim=(1, 2))
                error2 = torch.mean(self.criterion(recon2, batch), dim=(1, 2))
                
                # Anomaly score from Eq. 7
                batch_scores = 0.5 * error1 + 0.5 * error2
                scores.extend(batch_scores.cpu().numpy())
        return np.array(scores)

    def find_pot_threshold(self, scores, q=0.999):
        """Find the base POT threshold using a high quantile."""
        return np.quantile(scores, q)

    def find_dynamic_thresholds(self, scores, base_threshold, alpha=0.1, window_size=50):
        """
        Compute dynamic mPOT thresholds for a series of scores.
        mPOT(x) = POT(x) + alpha * recent_deviation(x)
        """
        scores_series = pd.Series(scores)
        # Calculate recent deviation (rolling standard deviation)
        recent_deviation = scores_series.rolling(window=window_size, min_periods=1).std().fillna(0)
        dynamic_thresholds = base_threshold + alpha * recent_deviation.values
        return dynamic_thresholds

    def predict_anomalies(self, data_loader, train_scores, q=0.999, alpha=0.1, window_size=50):
        """Predict anomalies using the full mPOT pipeline."""
        test_scores = self.compute_scores(data_loader)
        base_threshold = self.find_pot_threshold(train_scores, q=q)
        dynamic_thresholds = self.find_dynamic_thresholds(test_scores, base_threshold, alpha, window_size)
        
        anomalies = (test_scores > dynamic_thresholds).astype(int)
        return anomalies, test_scores, dynamic_thresholds
    
class AnomalyDetector:
    def __init__(self, model):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = nn.MSELoss(reduction='none')

    def compute_scores(self, data_loader):
        """Compute anomaly scores for a given dataset."""
        self.model.eval()
        scores =
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                recon1, recon2, _ = self.model(batch)
                
                error1 = torch.mean(self.criterion(recon1, batch), dim=(1, 2))
                error2 = torch.mean(self.criterion(recon2, batch), dim=(1, 2))
                
                # Anomaly score from Eq. 7
                batch_scores = 0.5 * error1 + 0.5 * error2
                scores.extend(batch_scores.cpu().numpy())
        return np.array(scores)

    def find_pot_threshold(self, scores, q=0.999):
        """Find the base POT threshold using a high quantile."""
        return np.quantile(scores, q)

    def find_dynamic_thresholds(self, scores, base_threshold, alpha=0.1, window_size=50):
        """
        Compute dynamic mPOT thresholds for a series of scores.
        mPOT(x) = POT(x) + alpha * recent_deviation(x)
        """
        scores_series = pd.Series(scores)
        # Calculate recent deviation (rolling standard deviation)
        recent_deviation = scores_series.rolling(window=window_size, min_periods=1).std().fillna(0)
        dynamic_thresholds = base_threshold + alpha * recent_deviation.values
        return dynamic_thresholds

    def predict_anomalies(self, data_loader, train_scores, q=0.999, alpha=0.1, window_size=50):
        """Predict anomalies using the full mPOT pipeline."""
        test_scores = self.compute_scores(data_loader)
        base_threshold = self.find_pot_threshold(train_scores, q=q)
        dynamic_thresholds = self.find_dynamic_thresholds(test_scores, base_threshold, alpha, window_size)
        
        anomalies = (test_scores > dynamic_thresholds).astype(int)
        return anomalies, test_scores, dynamic_thresholds
    


def calculate_eacs(f1, train_time, params, f1_max, t_max, p_max, w_a=0.4, w_t=0.4, w_p=0.2):
    """
    Calculates the Efficiency-Accuracy-Complexity Score (EACS) from Eq. 11.
    """
    f1_norm = f1 / f1_max
    t_norm = 1 - (train_time / t_max)
    p_norm = 1 - (params / p_max)
    
    eacs = (w_a * f1_norm) + (w_t * t_norm) + (w_p * p_norm)
    return eacs

def analyze_pareto_front(study):
    """Analyzes the Pareto front, calculates EACS, and returns the best trial."""
    pareto_front = study.best_trials
    
    if not pareto_front:
        print("No trials in Pareto front. Something went wrong.")
        return None

    results =
    for trial in pareto_front:
        results.append({
            'trial_number': trial.number,
            'f1_score': trial.values,
            'params': trial.values,
            'training_time': trial.user_attrs['training_time'],
            'hyperparams': trial.params
        })
    
    df_results = pd.DataFrame(results)
    
    # Normalize for EACS calculation
    f1_max = df_results['f1_score'].max()
    t_max = df_results['training_time'].max()
    p_max = df_results['params'].max()
    
    df_results['eacs'] = df_results.apply(
        lambda row: calculate_eacs(
            row['f1_score'], row['training_time'], row['params'],
            f1_max, t_max, p_max
        ), axis=1
    )
    
    df_results = df_results.sort_values(by='eacs', ascending=False).reset_index(drop=True)
    
    print("\nPareto Front Analysis:")
    print(df_results[['trial_number', 'f1_score', 'params', 'training_time', 'eacs']])
    
    # Plot Pareto front
    fig = vis.plot_pareto_front(study)
    fig.show()
    
    best_trial_config = df_results.iloc['hyperparams']
    print(f"\nBest model selected based on EACS (Trial #{df_results.iloc['trial_number']})")
    
    return best_trial_config


def preprocess_data(file_path, tower_id, window_size, batch_size):
    """Complete data preprocessing pipeline."""
    df = pd.read_csv(file_path, parse_dates=, index_col='Date')
    
    # 1. Filter by Tower ID
    df_tower = df == tower_id].copy()
    
    # 2. Select numeric features and handle invalid zeros
    numeric_cols = df_tower.select_dtypes(include=np.number).columns.tolist()
    df_numeric = df_tower[numeric_cols]
    df_numeric.replace(0, np.nan, inplace=True)
    
    # 3. Interpolate missing values
    df_numeric.interpolate(method='linear', limit_direction='both', inplace=True)
    
    # 4. Normalize data
    scaler = MinMaxScaler()
    
    # Split data (e.g., 80% train, 20% test)
    train_size = int(len(df_numeric) * 0.8)
    train_df, test_df = df_numeric.iloc[:train_size], df_numeric.iloc[train_size:]
    
    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)
    
    # 5. Create sliding windows
    def create_windows(data, ws):
        sequences =
        for i in range(len(data) - ws + 1):
            sequences.append(data[i:i+ws])
        return np.array(sequences)

    X_train = create_windows(train_scaled, window_size)
    X_test = create_windows(test_scaled, window_size)
    
    # Create DataLoaders
    train_dataset = TensorDataset(torch.from_numpy(X_train).float())
    test_dataset = TensorDataset(torch.from_numpy(X_test).float())
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, train_scaled, test_scaled, scaler, df_numeric.columns

def main(args):
    # For NAS, we need a validation set. We'll split the training data.
    # The final model will be trained on all training data.
    # This is a simplified setup for demonstration.
    # A more robust setup would have a dedicated validation set.
    
    # Note: The window size is a hyperparameter. For the initial data loading,
    # we need a placeholder. The NAS will optimize this. We'll create loaders
    # inside the objective function based on the trial's window size.
    # For simplicity here, we'll fix it for the main script flow.
    
    # Let's assume a placeholder window size for initial data prep
    # In a full pipeline, the data would be re-windowed inside the objective
    
    # --- 1. Data Preprocessing ---
    # This part would be more complex in a real run, adapting to window size
    # For now, let's prepare the raw data first.
    df = pd.read_csv(args.file_path, parse_dates=, index_col='Date')
    df_tower = df == args.tower_id].copy()
    numeric_cols = df_tower.select_dtypes(include=np.number).columns.tolist()
    df_numeric = df_tower[numeric_cols]
    df_numeric.replace(0, np.nan, inplace=True)
    df_numeric.interpolate(method='linear', limit_direction='both', inplace=True)
    
    train_size = int(len(df_numeric) * 0.8)
    train_df, test_df = df_numeric.iloc[:train_size], df_numeric.iloc[train_size-50:train_size], # val set
    
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df)
    
    # Create a dummy validation set for NAS
    val_size = int(len(train_scaled) * 0.2)
    train_data_for_nas = train_scaled[:-val_size]
    val_data_for_nas = train_scaled[-val_size:]

    # --- 2. Run NAS ---
    print("--- Starting Neural Architecture Search ---")
    # A simplified data loading for NAS trials
    # In a real scenario, you'd handle windowing inside the objective
    # For now, we'll use a fixed window size for demonstration loaders
    ws = 20
    bs = 64
    train_windows_nas = create_windows(train_data_for_nas, ws)
    val_windows_nas = create_windows(val_data_for_nas, ws)
    train_loader_nas = DataLoader(TensorDataset(torch.from_numpy(train_windows_nas).float()), batch_size=bs)
    val_loader_nas = DataLoader(TensorDataset(torch.from_numpy(val_windows_nas).float()), batch_size=bs)
    
    study = run_nas_study(train_loader_nas, val_loader_nas, feature_dim=train_scaled.shape, n_trials=args.n_trials)
    
    # --- 3. Analyze Pareto Front and Select Best Model ---
    best_config = analyze_pareto_front(study)
    if best_config is None:
        return
        
    # --- 4. Train Final Model with Best Hyperparameters ---
    print("\n--- Training Final Model with Best Hyperparameters ---")
    final_window_size = best_config['window_size']
    final_batch_size = best_config['batch_size']
    
    X_train_final = create_windows(train_scaled, final_window_size)
    train_loader_final = DataLoader(TensorDataset(torch.from_numpy(X_train_final).float()), batch_size=final_batch_size, shuffle=True)
    
    # Use the same validation loader for consistency during final training printouts
    val_windows_final = create_windows(val_data_for_nas, final_window_size)
    val_loader_final = DataLoader(TensorDataset(torch.from_numpy(val_windows_final).float()), batch_size=final_batch_size)
    
    best_config['feature_dim'] = train_scaled.shape
    best_config['epochs'] = 25 # Train longer for the final model
    
    final_model = TransNAS_TSAD_Model(best_config)
    final_trainer = AdversarialTrainer(final_model, train_loader_final, val_loader_final, best_config)
    final_trainer.train()
    
    # --- 5. Evaluate on Test Set ---
    print("\n--- Evaluating on Test Set ---")
    test_scaled = scaler.transform(test_df)
    X_test_final = create_windows(test_scaled, final_window_size)
    test_loader_final = DataLoader(TensorDataset(torch.from_numpy(X_test_final).float()), batch_size=final_batch_size)
    
    detector = AnomalyDetector(final_model)
    
    # Get scores on training data to set POT threshold
    train_scores_final = detector.compute_scores(train_loader_final)
    
    # Predict anomalies on the test set
    anomalies, test_scores, dynamic_thresholds = detector.predict_anomalies(
        test_loader_final,
        train_scores=train_scores_final
    )
    
    print(f"Detected {np.sum(anomalies)} anomalies in the test set.")
    
    # --- 6. Report Anomalies ---
    anomaly_indices = np.where(anomalies == 1)
    print("\n--- Detected Anomaly Timestamps and Details ---")
    for idx in anomaly_indices[:10]: # Print first 10 anomalies
        # Adjust index to match original dataframe
        timestamp = test_df.index[idx + final_window_size - 1]
        print(f"Timestamp: {timestamp}, Anomaly Score: {test_scores[idx]:.4f}, Threshold: {dynamic_thresholds[idx]:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TransNAS-TSAD Implementation")
    parser.add_argument('--file_path', type=str, default='Parameters_1.csv', help='Path to the dataset CSV file')
    parser.add_argument('--tower_id', type=str, default='NEW CT', help='ID of the tower to analyze')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of trials for the NAS')
    args = parser.parse_args()
    main(args)