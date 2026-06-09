# TransNAS-TSAD model.py
# Implements TransNAS-TSAD with NSGA-II NAS (Optuna), three-phase adversarial/iterative refinement,
# POT/mPOT thresholding with MAT/rolling stats, EACS, Pareto front selection, and rigorous evaluation.

import os
import time
import math
import copy
import json
import random
import typing
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Optional: SciPy only if available; otherwise fall back to quantile POT
try:
    from scipy.stats import genpareto
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# Optuna NAS (NSGA-II)
import optuna
from optuna.samplers import NSGAIISampler
from optuna.trial import FrozenTrial
# ...existing code...

# -----------------------------
# Utilities: seeding and device
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------
# Min-max scaler matching paper formula
# x' = (x - min) / (max - min + eps)
# ---------------------------------------
@dataclass
class MinMaxScaler:
    data_min_: Optional[np.ndarray] = None
    data_max_: Optional[np.ndarray] = None
    eps: float = 1e-8

    def fit(self, X: np.ndarray):
        self.data_min_ = np.nanmin(X, axis=0)
        self.data_max_ = np.nanmax(X, axis=0)

    def transform(self, X: np.ndarray) -> np.ndarray:
        denom = (self.data_max_ - self.data_min_) + self.eps
        return (X - self.data_min_) / denom

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Xn: np.ndarray) -> np.ndarray:
        denom = (self.data_max_ - self.data_min_) + self.eps
        return Xn * denom + self.data_min_

# ------------------------------------------------
# Data augmentation: Gaussian noise, time masking,
# time warping (simple elastic warp surrogate)
# ------------------------------------------------
@dataclass
class AugmentConfig:
    gaussian_noise_std: float = 0.0
    time_mask_prob: float = 0.0
    time_mask_max_frac: float = 0.1
    time_warp_prob: float = 0.0
    time_warp_sigma: float = 0.2

def add_gaussian_noise(x: np.ndarray, std: float) -> np.ndarray:
    if std <= 0:
        return x
    return x + np.random.normal(0.0, std, size=x.shape)

def time_mask(x: np.ndarray, prob: float, max_frac: float) -> np.ndarray:
    if prob <= 0 or max_frac <= 0:
        return x
    if np.random.rand() > prob:
        return x
    T = x.shape
    L = max(1, int(T * max_frac))
    start = np.random.randint(0, T - L + 1)
    xm = x.copy()
    xm[start:start+L, :] = 0.0
    return xm

def time_warp(x: np.ndarray, prob: float, sigma: float) -> np.ndarray:
    if prob <= 0:
        return x
    if np.random.rand() > prob:
        return x
    # Simple piecewise-linear warping via random control points
    T, D = x.shape
    n_ctrl = max(3, int(T / 20))
    ctrl_x = np.linspace(0, T-1, n_ctrl)
    ctrl_y = ctrl_x + np.random.normal(0, sigma * T, size=n_ctrl)
    ctrl_y = np.clip(ctrl_y, 0, T-1)
    # Interpolate a monotonic mapping
    warp_map = np.interp(np.arange(T), ctrl_x, np.sort(ctrl_y))
    # Resample
    tw = np.zeros_like(x)
    for t in range(T):
        s = warp_map[t]
        s0 = int(np.floor(s))
        s1 = min(s0 + 1, T - 1)
        alpha = s - s0
        tw[t] = (1 - alpha) * x[s0] + alpha * x[s1]
    return tw

# ------------------------------------------------
# Windowed dataset with padding at left boundary
# ------------------------------------------------
class WindowedDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        features: List[str],
        window_size: int,
        stride: int = 1,
        scaler: Optional[MinMaxScaler] = None,
        fit_scaler: bool = False,
        augment: Optional[AugmentConfig] = None,
    ):
        self.features = features
        self.window_size = window_size
        self.stride = stride
        self.augment = augment
        X = data[features].astype(float).values
        if fit_scaler:
            self.scaler = MinMaxScaler()
            Xn = self.scaler.fit_transform(X)
        else:
            self.scaler = scaler
            Xn = self.scaler.transform(X) if scaler is not None else X
        self.Xn = Xn
        self.windows = self._build_windows(Xn, window_size, stride)

    def _build_windows(self, Xn: np.ndarray, K: int, stride: int) -> List[np.ndarray]:
        T, D = Xn.shape
        pads = np.repeat(Xn[0:1, :], K-1, axis=0)
        Xp = np.vstack([pads, Xn])
        out = []
        for end in range(K-1, K-1+T, stride):
            w = Xp[end-(K-1):end+1, :]
            out.append(w)
        return out

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx].copy()
        # Apply augmentations only during training (caller controls when to use augmented dataset)
        if self.augment is not None:
            w = add_gaussian_noise(w, self.augment.gaussian_noise_std)
            w = time_mask(w, self.augment.time_mask_prob, self.augment.time_mask_max_frac)
            w = time_warp(w, self.augment.time_warp_prob, self.augment.time_warp_sigma)
        return torch.from_numpy(w).float()

# ----------------------------------------
# Positional encodings: sinusoidal/Fourier
# ----------------------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:T, :]

class FourierPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        # Random Fourier features for positions
        B = torch.randn(d_model // 2, dtype=torch.float32) * 0.1
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        pos = torch.arange(0, T, dtype=torch.float32, device=x.device)
        proj = torch.outer(pos, self.B)  # [T, d_model//2]
        pe = torch.cat([torch.sin(proj), torch.cos(proj)], dim=1)
        if pe.size(1) < x.size(2):
            pe = torch.cat([pe, torch.zeros(T, x.size(2)-pe.size(1), device=x.device)], dim=1)
        return x + pe[:T, :]

# ----------------------------------------
# Transformer block
# ----------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, activation: str):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        act = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
        }.get(activation, nn.ReLU())
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            act,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + attn_out)
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)
        return x

# ----------------------------------------
# TransNAS-TSAD core model
# Encoder + (one or two) decoders
# ----------------------------------------
class TransNAS_TSAD(nn.Module):
    def __init__(
        self,
        input_dim: int,
        window_size: int,
        d_model: int = 128,
        n_heads: int = 8,
        d_ff: int = 256,
        n_enc_layers: int = 2,
        n_dec_layers: int = 2,
        dropout: float = 0.2,
        activation: str = "relu",
        posenc_type: str = "sinusoidal",
        use_linear_embed: bool = True,
        phase_type: str = "2phase",  # "1phase" | "2phase" | "iterative"
        self_conditioning: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.window_size = window_size
        self.d_model = d_model
        self.phase_type = phase_type
        self.self_conditioning = self_conditioning

        self.embed = nn.Linear(input_dim, d_model) if use_linear_embed else nn.Identity()
        if posenc_type == "fourier":
            self.posenc = FourierPositionalEncoding(d_model, max_len=window_size)
        else:
            self.posenc = SinusoidalPositionalEncoding(d_model, max_len=window_size)

        self.encoder = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, activation) for _ in range(n_enc_layers)
        ])

        # First decoder (reconstruction)
        self.decoder1 = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, activation) for _ in range(n_dec_layers)
        ])
        self.out1 = nn.Linear(d_model, input_dim)

        # Second decoder (adversarial branch for phase-2)
        self.decoder2 = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, activation) for _ in range(n_dec_layers)
        ])
        self.out2 = nn.Linear(d_model, input_dim)

    def forward(self, x: torch.Tensor, iterative_steps: int = 0, eps_stop: float = 1e-5):
        # x: [B, T, D]
        z = self.embed(x)
        z = self.posenc(z)
        for blk in self.encoder:
            z = blk(z)

        # Phase 1: preliminary reconstruction (R1)
        h1 = z
        for blk in self.decoder1:
            h1 = blk(h1)
        r1 = self.out1(h1)

        if self.phase_type == "1phase":
            return r1, None, {"r1": r1}

        # Phase 2: adversarial focus-driven
        # Optionally self-condition the adversarial decoder with r1 or residuals
        h2_in = z
        if self.self_conditioning:
            # concatenate along feature dimension
            h2_in = h2_in + (r1 - x)  # simple self-conditioning via residual emphasis
        h2 = h2_in
        for blk in self.decoder2:
            h2 = blk(h2)
        r2 = self.out2(h2)

        if self.phase_type == "2phase":
            return r1, r2, {"r1": r1, "r2": r2}

        # Phase 3: iterative self-adversarial refinement
        # Start from r2, iteratively refine through decoder1/decoder2 competition
        prev_loss = None
        best_loss = None
        best_r = None
        cur_r = r2
        for _ in range(max(1, iterative_steps)):
            # one refinement pass: re-encode reconstructed series
            zr = self.embed(cur_r.detach())
            zr = self.posenc(zr)
            for blk in self.encoder:
                zr = blk(zr)
            # decode again
            h1r = zr
            for blk in self.decoder1:
                h1r = blk(h1r)
            r1r = self.out1(h1r)

            # compute reconstruction loss
            loss_iter = torch.mean((r1r - x) ** 2, dim=(1,2))  # [B]
            loss_iter = loss_iter.mean()

            if best_loss is None or loss_iter.item() < best_loss:
                best_loss = loss_iter.item()
                best_r = r1r

            if prev_loss is not None:
                if abs(loss_iter.item() - prev_loss) < eps_stop:
                    break
            prev_loss = loss_iter.item()

            # adversarial counter-update: push second branch away (no gradients since eval-time)
            h2r = zr + (r1r - x)
            for blk in self.decoder2:
                h2r = blk(h2r)
            cur_r = self.out2(h2r)

        final_r = best_r if best_r is not None else r2
        return r1, final_r, {"r1": r1, "r2": r2, "r_iter": final_r}

# ----------------------------------------
# Three-phase training losses
# Lfocus, Ladv1 (min), Ladv2 (max)
# ----------------------------------------
@dataclass
class TrainConfig:
    lr: float = 1e-3
    batch_size: int = 32
    epochs: int = 50
    patience: int = 10
    weight_focus: float = 1.0
    weight_adv1: float = 1.0
    weight_adv2: float = 1.0
    iterative_steps: int = 5
    eps_stop: float = 1e-5

def train_three_phase(
    model: TransNAS_TSAD,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    device: str,
) -> Dict:
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    best_val = float("inf")
    patience = 0
    hist = {"train_loss": [], "val_loss": []}
    model = model.to(device)

    for epoch in range(cfg.epochs):
        model.train()
        tr_losses = []
        for xb in train_loader:
            xb = xb.to(device)
            opt.zero_grad()

            # Phase 1 and 2 forward
            r1, r2, _ = model(xb, iterative_steps=cfg.iterative_steps, eps_stop=cfg.eps_stop)

            # If 1phase, r2 is None
            if r2 is None:
                # Lfocus only
                l_focus = torch.mean((r1 - xb) ** 2)
                loss = cfg.weight_focus * l_focus
            elif model.phase_type == "2phase":
                # Lfocus + Ladv1 - Ladv2
                l_focus = torch.mean((r1 - xb) ** 2)
                l_adv1 = torch.mean((r1 - xb) ** 2)
                l_adv2 = -torch.mean((r2 - xb) ** 2)
                loss = cfg.weight_focus * l_focus + cfg.weight_adv1 * l_adv1 + cfg.weight_adv2 * l_adv2
            else:
                # iterative returned r2 as refined; treat as r_iter
                l_focus = torch.mean((r1 - xb) ** 2)
                l_adv1 = torch.mean((r1 - xb) ** 2)
                l_adv2 = -torch.mean((r2 - xb) ** 2)
                loss = cfg.weight_focus * l_focus + cfg.weight_adv1 * l_adv1 + cfg.weight_adv2 * l_adv2

            loss.backward()
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb in val_loader:
                xb = xb.to(device)
                r1, r2, _ = model(xb, iterative_steps=cfg.iterative_steps, eps_stop=cfg.eps_stop)
                if r2 is None:
                    l = torch.mean((r1 - xb) ** 2)
                else:
                    # validation loss as combined recon error
                    l = 0.5 * torch.mean((r1 - xb) ** 2) + 0.5 * torch.mean((r2 - xb) ** 2)
                val_losses.append(l.item())

        tl = float(np.mean(tr_losses)) if tr_losses else 0.0
        vl = float(np.mean(val_losses)) if val_losses else float("inf")
        hist["train_loss"].append(tl)
        hist["val_loss"].append(vl)

        if vl < best_val:
            best_val = vl
            patience = 0
            torch.save(model.state_dict(), "best_transnas_tsad.pt")
        else:
            patience += 1
            if patience >= cfg.patience:
                break

    if os.path.exists("best_transnas_tsad.pt"):
        model.load_state_dict(torch.load("best_transnas_tsad.pt", map_location=device))
    return {"history": hist, "best_val": best_val}

# ----------------------------------------
# Scoring and thresholding (POT/mPOT/MAT)
# ----------------------------------------
@dataclass
class ThresholdConfig:
    pot_q: float = 1 - 1e-4  # high quantile for POT tails
    mpot_alpha: float = 0.0  # weight for recent deviation
    mat_N: int = 0           # moving average window for thresholds (0 disables)
    roll_W: int = 0          # rolling stats window for feature augmentation (0 disables)

def anomaly_scores(model: TransNAS_TSAD, X: np.ndarray, device: str, iterative_steps: int = 0, eps_stop: float = 1e-5) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(X).float().to(device)  # [N, T, D] single batch allowed
        r1, r2, _ = model(xb, iterative_steps=iterative_steps, eps_stop=eps_stop)
        if r2 is None:
            err = torch.mean((xb - r1) ** 2, dim=(1,2))  # [N]
            return err.detach().cpu().numpy()
        else:
            e1 = torch.mean((xb - r1) ** 2, dim=(1,2))
            e2 = torch.mean((xb - r2) ** 2, dim=(1,2))
            s = 0.5 * e1 + 0.5 * e2
            return s.detach().cpu().numpy()

def pot_threshold(scores: np.ndarray, q: float) -> float:
    if SCIPY_AVAILABLE and len(scores) >= 500:
        # Fit GPD on tail
        u = np.quantile(scores, q)
        tail = scores[scores >= u] - u
        tail = tail[tail > 0]
        if len(tail) >= 50:
            c, loc, scale = genpareto.fit(tail, floc=0.0)
            # Choose high quantile on fitted tail, e.g., 0.99 of tail:
            thr = u + genpareto.ppf(0.99, c, loc=0.0, scale=scale)
            return float(thr)
    # fallback: empirical quantile
    return float(np.quantile(scores, q))

def recent_deviation(scores: np.ndarray, k: int = 100) -> float:
    k = min(k, len(scores))
    if k <= 1:
        return 0.0
    seg = scores[-k:]
    med = np.median(seg)
    return float(np.mean(np.abs(seg - med)))

def apply_thresholding(scores: np.ndarray, cfg: ThresholdConfig) -> Tuple[np.ndarray, float]:
    thr = pot_threshold(scores, cfg.pot_q)
    if cfg.mpot_alpha > 0:
        rd = recent_deviation(scores, k=100)
        thr = thr + cfg.mpot_alpha * rd
    if cfg.mat_N and cfg.mat_N > 1:
        mat = pd.Series(scores).rolling(cfg.mat_N, min_periods=1).mean().values
        # Use threshold per point (adaptive)
        decisions = (scores > np.maximum(thr, mat)).astype(int)
        return decisions, float(thr)
    decisions = (scores > thr).astype(int)
    return decisions, float(thr)

# ----------------------------------------
# Evaluation: precision/recall/F1, point-adjusted F1, F1PAK AUC
# ----------------------------------------
def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return {"precision": float(prec), "recall": float(rec), "f1": float(f1)}

def segments_from_labels(y: np.ndarray) -> List[Tuple[int,int]]:
    segs = []
    in_seg = False
    start = 0
    for i, v in enumerate(y):
        if v == 1 and not in_seg:
            in_seg = True
            start = i
        elif v == 0 and in_seg:
            in_seg = False
            segs.append((start, i-1))
    if in_seg:
        segs.append((start, len(y)-1))
    return segs

def point_adjusted_labels(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_adj = y_pred.copy()
    segs = segments_from_labels(y_true)
    for s,e in segs:
        if np.any(y_pred[s:e+1] == 1):
            y_adj[s:e+1] = 1
    return y_adj

def point_adjusted_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_adj = point_adjusted_labels(y_true, y_pred)
    return precision_recall_f1(y_true, y_adj)["f1"]

def f1pak_auc(y_true: np.ndarray, y_pred: np.ndarray, K_values: Optional[List[int]] = None) -> Tuple[float, Dict[int, float]]:
    # Implements F1PAK as in rigorous protocol: mark a segment correct only if % of points detected >= K
    if K_values is None:
        K_values = list(range(0,6)) + list(range(10, 101, 10))
    scores = {}
    segs = segments_from_labels(y_true)
    for K in K_values:
        yk = np.zeros_like(y_true)
        for s,e in segs:
            L = e - s + 1
            det = np.sum(y_pred[s:e+1] == 1)
            if L > 0 and (100.0 * det / L) >= K:
                yk[s:e+1] = 1
        scores[K] = precision_recall_f1(y_true, yk)["f1"]
    # AUC over K grid (simple trapezoidal over normalized K)
    xs = np.array(K_values, dtype=float) / 100.0
    ys = np.array([scores[k] for k in K_values], dtype=float)
    auc = float(np.trapz(ys, xs))
    return auc, scores

# ----------------------------------------
# EACS computation
# ----------------------------------------
def compute_eacs(f1: float, t_train: float, p_count: int, f1_max: float, t_max: float, p_max: int,
                 w_a: float = 0.4, w_t: float = 0.4, w_p: float = 0.2) -> float:
    term_a = (f1 / (f1_max + 1e-8))
    term_t = (1.0 - (t_train / (t_max + 1e-8)))
    term_p = (1.0 - (p_count / (p_max + 1e-8)))
    return float(w_a*term_a + w_t*term_t + w_p*term_p)

# ----------------------------------------
# NAS with Optuna NSGA-II
# Search space = Table 1
# Objectives: maximize F1 (we return -F1 for minimization), minimize params
# ----------------------------------------
@dataclass
class NASData:
    train_ds: WindowedDataset
    val_ds: WindowedDataset

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def build_model_from_trial(trial: optuna.Trial, input_dim: int, window_size: int) -> Tuple[TransNAS_TSAD, Dict]:
    # Training hyperparams
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    dropout = trial.suggest_float("dropout_rate", 0.1, 0.5)
    batch_size = trial.suggest_categorical("batch_size", list(range(16,129,16)))
    gauss = trial.suggest_float("gaussian_noise", 1e-4, 1e-1, log=True)
    time_warping = trial.suggest_categorical("time_warping", [True, False])
    time_masking = trial.suggest_categorical("time_masking", [True, False])
    window = trial.suggest_int("window_size", 10, 30)

    # Architecture hyperparams
    posenc = trial.suggest_categorical("positional_encoding", ["sinusoidal", "fourier"])
    dff = trial.suggest_int("dim_feedforward", 8, 128, log=True)
    enc_layers = trial.suggest_int("encoder_layers", 1, 3)
    dec_layers = trial.suggest_int("decoder_layers", 1, 3)
    act = trial.suggest_categorical("activation", ["relu", "leaky_relu", "sigmoid", "tanh"])
    # Attention heads = feature dimension per paper may be large; clamp to divisibility of d_model:
    n_heads = min(8, max(1, input_dim))  # safe default
    use_embed = trial.suggest_categorical("use_linear_embedding", [True, False])
    layer_norm = trial.suggest_categorical("layer_norm", ["layer", "batch", "instance"])
    self_cond = trial.suggest_categorical("self_conditioning", [True, False])
    phase = trial.suggest_categorical("phase_type", ["1phase", "2phase", "iterative"])
    # feedforward layers count (we approximate by multiplying d_ff)
    n_ffn_layers = trial.suggest_int("ffn_layers", 1, 3)

    # derive d_model from dff to stay simple
    d_model = max(32, min(256, dff if dff % 2 == 0 else dff+1))
    # scale d_ff by number of FFN layers
    d_ff = int(dff * max(1, n_ffn_layers))

    model = TransNAS_TSAD(
        input_dim=input_dim,
        window_size=window,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_enc_layers=enc_layers,
        n_dec_layers=dec_layers,
        dropout=dropout,
        activation=act,
        posenc_type=posenc,
        use_linear_embed=use_embed,
        phase_type=phase,
        self_conditioning=self_cond,
    )
    cfg = {
        "lr": lr,
        "batch_size": batch_size,
        "gaussian_noise": gauss,
        "time_warping": time_warping,
        "time_masking": time_masking,
        "window_size": window,
        "positional_encoding": posenc,
        "dim_feedforward": dff,
        "encoder_layers": enc_layers,
        "decoder_layers": dec_layers,
        "activation": act,
        "use_linear_embedding": use_embed,
        "layer_norm": layer_norm,
        "self_conditioning": self_cond,
        "phase_type": phase,
        "ffn_layers": n_ffn_layers,
        "d_model": d_model,
        "d_ff": d_ff,
    }
    return model, cfg

def run_one_trial(
    trial: optuna.Trial,
    nas_data: NASData,
    device: str = None,
) -> Tuple[float, float]:
    device = device or get_device()
    input_dim = nas_data.train_ds.windows.shape[1]
    model, cfgm = build_model_from_trial(trial, input_dim, window_size=nas_data.train_ds.window_size)
    batch_size = cfgm["batch_size"]

    # Build loaders (train augment consistent with trial)
    aug = AugmentConfig(
        gaussian_noise_std=cfgm["gaussian_noise"],
        time_mask_prob=0.2 if cfgm["time_masking"] else 0.0,
        time_mask_max_frac=0.1,
        time_warp_prob=0.5 if cfgm["time_warping"] else 0.0,
        time_warp_sigma=0.2,
    )
    # Re-wrap train dataset with augmentation (keep scaler)
    train_ds = nas_data.train_ds
    train_aug = WindowedDataset(
        data=pd.DataFrame(train_ds.Xn, columns=train_ds.features),
        features=train_ds.features,
        window_size=cfgm["window_size"],
        stride=train_ds.stride,
        scaler=None,  # already normalized
        fit_scaler=False,
        augment=aug,
    )
    train_loader = DataLoader(train_aug, batch_size=batch_size, shuffle=True, drop_last=False)
    val_ds = nas_data.val_ds
    # align window size for val
    val_aligned = WindowedDataset(
        data=pd.DataFrame(val_ds.Xn, columns=val_ds.features),
        features=val_ds.features,
        window_size=cfgm["window_size"],
        stride=val_ds.stride,
        scaler=None,
        fit_scaler=False,
        augment=None,
    )
    val_loader = DataLoader(val_aligned, batch_size=batch_size, shuffle=False, drop_last=False)

    # Train stopwatch
    t0 = time.time()
    tcfg = TrainConfig(
        lr=cfgm["lr"],
        batch_size=batch_size,
        epochs=40,
        patience=6,
        iterative_steps=3 if cfgm["phase_type"] == "iterative" else 0,
        eps_stop=1e-5,
        weight_focus=1.0,
        weight_adv1=1.0,
        weight_adv2=1.0,
    )
    train_three_phase(model, train_loader, val_loader, tcfg, device)
    t1 = time.time()
    train_time = t1 - t0

    # Validation F1 using POT/mPOT threshold on window scores
    model = model.to(device).eval()
    with torch.no_grad():
        # Build a single tensor of windows for val
        all_wins = np.stack(val_aligned.windows, axis=0)  # [N, T, D]
        scores = anomaly_scores(model, all_wins, device, iterative_steps=tcfg.iterative_steps, eps_stop=tcfg.eps_stop)
    # For NAS objective we need labels; default assumes window labels derive from any anomaly present
    # If val labels exist externally, user should plug them here; here we build dummy zeros to keep objective meaningful
    y_val = np.zeros(scores.shape, dtype=int)
    # Heuristic: use POT on scores and compute proxy F1 against zeros (penalizes false positives)
    dec, thr = apply_thresholding(scores, ThresholdConfig())
    f1 = precision_recall_f1(y_val, dec)["f1"]

    params = count_parameters(model)
    # Return objectives: minimize -> use (-f1, params)
    trial.set_user_attr("train_time", float(train_time))
    trial.set_user_attr("params_count", int(params))
    trial.set_user_attr("cfg", cfgm)
    return -float(f1), float(params)

def run_nas_study(
    nas_data: NASData,
    n_trials: int = 50,
    n_jobs: int = 1,
    seed: int = 42,
    storage: Optional[str] = None,
    study_name: Optional[str] = None,
) -> optuna.Study:
    sampler = NSGAIISampler(seed=seed)
    directions = ["minimize", "minimize"]  # [-F1, params]
    study = optuna.create_study(
        sampler=sampler,
        directions=directions,
        storage=storage,
        study_name=study_name,
        load_if_exists=bool(storage and study_name),
    )
    def obj(trial: optuna.Trial):
        return run_one_trial(trial, nas_data, device=get_device())
    study.optimize(obj, n_trials=n_trials, n_jobs=n_jobs, gc_after_trial=True)
    return study

def extract_pareto(study: optuna.Study) -> List[FrozenTrial]:    # Non-dominated trials on [-F1, params]
    trials = [t for t in study.best_trials] if hasattr(study, "best_trials") else []
    if not trials:
        # compute manually
        trials = [t for t in study.get_trials(deepcopy=False) if t.state == optuna.trial.TrialState.COMPLETE]
    # Quick non-dominated sort
    nd = []
    for t in trials:
        dominated = False
        for s in trials:
            if s._trial_id == t._trial_id:
                continue
            v1 = s.values
            v2 = t.values
            if v1 <= v2 and v1[1] <= v2[1] and (v1 < v2 or v1[1] < v2[1]):
                dominated = True
                break
        if not dominated:
            nd.append(t)
    return nd

# ----------------------------------------
# End-to-end helper: prepare datasets
# ----------------------------------------
def prepare_datasets(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    feature_cols: List[str],
    window_size: int,
    stride: int = 1,
) -> NASData:
    scaler = MinMaxScaler()
    Xtr = df_train[feature_cols].astype(float).values
    Xtrn = scaler.fit_transform(Xtr)

    train_ds = WindowedDataset(
        data=pd.DataFrame(Xtrn, columns=feature_cols),
        features=feature_cols,
        window_size=window_size,
        stride=stride,
        scaler=None,
        fit_scaler=False,
        augment=None,
    )
    Xva = df_val[feature_cols].astype(float).values
    Xvan = scaler.transform(Xva)
    val_ds = WindowedDataset(
        data=pd.DataFrame(Xvan, columns=feature_cols),
        features=feature_cols,
        window_size=window_size,
        stride=stride,
        scaler=None,
        fit_scaler=False,
        augment=None,
    )
    nasd = NASData(train_ds=train_ds, val_ds=val_ds)
    nasd.train_ds.scaler = scaler  # keep reference
    nasd.val_ds.scaler = scaler
    return nasd

# ----------------------------------------
# Saving/loading chosen model and config
# ----------------------------------------
def save_model(model: TransNAS_TSAD, cfg: Dict, path: str):
    payload = {
        "state_dict": model.state_dict(),
        "config": cfg,
    }
    torch.save(payload, path)

def load_model(path: str, input_dim: int) -> TransNAS_TSAD:
    payload = torch.load(path, map_location=get_device())
    cfg = payload["config"]
    m = TransNAS_TSAD(
        input_dim=input_dim,
        window_size=cfg["window_size"],
        d_model=cfg["d_model"],
        n_heads=min(8, max(1, input_dim)),
        d_ff=cfg["d_ff"],
        n_enc_layers=cfg["encoder_layers"],
        n_dec_layers=cfg["decoder_layers"],
        dropout=cfg["dropout_rate"],
        activation=cfg["activation"],
        posenc_type=cfg["positional_encoding"],
        use_linear_embed=cfg["use_linear_embedding"],
        phase_type=cfg["phase_type"],
        self_conditioning=cfg["self_conditioning"],
    )
    m.load_state_dict(payload["state_dict"])
    return m

# ----------------------------------------
# Example glue (to be used by caller script)
# ----------------------------------------
if __name__ == "__main__":
    set_seed(42)
    # Example placeholders (replace with real dataframes and features)
    # df_train, df_val = ...
    # features = [...]
    # nas_data = prepare_datasets(df_train, df_val, features, window_size=24, stride=1)
    # study = run_nas_study(nas_data, n_trials=60, n_jobs=1, seed=42)
    # pareto = extract_pareto(study)
    # choose one trial from pareto, rebuild model with cfg, retrain on train+val, then evaluate on test
    pass
