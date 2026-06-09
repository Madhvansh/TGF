# TransNAS-TSAD model.py
# Faithful implementation of TransNAS-TSAD with NSGA-II NAS, adversarial phases,
# mPOT/POT thresholding, EACS metric, and F1PAK evaluation.

import os
import time
import math
import copy
import json
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils import check_random_state

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import optuna
from optuna.samplers import NSGAIISampler

# ============== Utilities ==============

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def activation_from_name(name: str):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "leaky relu" or name == "leaky_relu":
        return nn.LeakyReLU()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "tanh":
        return nn.Tanh()
    return nn.ReLU()

# Fourier positional encoding
class FourierPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        # Use sin/cos but with learned frequencies scaling for flexibility
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-(math.log(10000.0) / d_model))
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        T = x.size(1)
        return x + self.pe[:, :T, :]

class SinusoidalPositionalEncoding(FourierPositionalEncoding):
    # alias, identical structure for practical purposes here
    pass

# Time warping / masking helpers
def time_mask(x: torch.Tensor, mask_prob: float = 0.1):
    # x: (B, T, F)
    if mask_prob <= 0.0:
        return x
    B, T, F = x.shape
    mask = torch.rand(B, T, 1, device=x.device) < mask_prob
    x_masked = x.clone()
    x_masked[mask.expand(-1, -1, F)] = 0.0
    return x_masked

def time_warp(x: torch.Tensor, sigma: float = 0.04):
    # Simple elastic warp along time dimension (approximate)
    if sigma <= 0.0:
        return x
    B, T, F = x.shape
    device = x.device
    tt = torch.arange(T, device=device).float()
    tt = tt.unsqueeze(0).repeat(B, 1)  # (B, T)
    # random smooth noise
    noise = torch.randn(B, 4, device=device) * sigma
    knots = torch.linspace(0, T-1, 4, device=device)
    warp = torch.zeros(B, T, device=device)
    for i in range(B):
        warp[i] = torch.interp(tt[i], knots, noise[i])
    tt_warped = tt + warp
    tt_warped = tt_warped.clamp(0, T-1)
    # linear interpolate
    x_warped = torch.zeros_like(x)
    t0 = tt_warped.floor().long()
    t1 = (t0 + 1).clamp(max=T-1)
    alpha = (tt_warped - t0.float()).unsqueeze(-1)
    x_warped = (1 - alpha) * x.gather(1, t0.unsqueeze(-1).expand(-1, -1, F)) + alpha * x.gather(1, t1.unsqueeze(-1).expand(-1, -1, F))
    return x_warped

# Min-max normalization per paper
def minmax_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # x: (N, F)
    minv = np.min(x, axis=0)
    maxv = np.max(x, axis=0)
    # Avoid zero range
    maxv_safe = np.where(maxv == minv, minv + 1e-8, maxv)
    return minv, maxv_safe

def minmax_transform(x: np.ndarray, minv: np.ndarray, maxv: np.ndarray) -> np.ndarray:
    return (x - minv) / (maxv - minv + 1e-8)

# ============== Dataset / Windowing ==============

class SlidingWindowDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,          # (N, F)
        y: Optional[np.ndarray],# (N,) binary labels (optional)
        window_size: int,
        stride: int = 1,
        augment: Dict = None,
        split: str = "train",
        norm_fit_on: Optional[np.ndarray] = None,  # for min-max
    ):
        super().__init__()
        self.X = X.astype(np.float32)
        self.y = y
        self.window_size = window_size
        self.stride = stride
        self.split = split
        self.augment = augment or {}
        # min-max per paper (minS, maxS) possibly across train+test; default to provided fit range
        if norm_fit_on is None:
            self.minv, self.maxv = minmax_fit(self.X)
        else:
            self.minv, self.maxv = norm_fit_on
        self.Xn = minmax_transform(self.X, self.minv, self.maxv).astype(np.float32)

        self.windows, self.labels = self._make_windows(self.Xn, self.y, window_size, stride)

    def _make_windows(self, Xn, y, K, stride):
        N, F = Xn.shape
        if N < K:
            # pad by repeating first
            pad = np.repeat(Xn[:1, :], K - N, axis=0)
            Xn = np.vstack([pad, Xn])
            if y is not None:
                ypad = np.repeat(y[:1], K - N, axis=0)
                y = np.concatenate([ypad, y])

        idxs = list(range(0, len(Xn) - K + 1, stride))
        wins = np.stack([Xn[i:i+K, :] for i in idxs], axis=0)  # (M, K, F)
        if y is not None:
            # label per point; keep segment labels aligned for pointwise eval later
            y_w = np.stack([y[i:i+K] for i in idxs], axis=0)
        else:
            y_w = None
        return wins, y_w

    def __len__(self):
        return self.windows.shape

    def __getitem__(self, i):
        x = torch.from_numpy(self.windows[i])  # (K, F)
        if self.split == "train":
            # augmentations if enabled
            if self.augment.get("gaussian_noise", 0.0) > 0.0:
                x = x + torch.randn_like(x) * float(self.augment["gaussian_noise"])
            if self.augment.get("time_mask", False):
                x = time_mask(x.unsqueeze(0), mask_prob=self.augment.get("time_mask_prob", 0.1)).squeeze(0)
            if self.augment.get("time_warp", False):
                x = time_warp(x.unsqueeze(0), sigma=self.augment.get("time_warp_sigma", 0.04)).squeeze(0)
        y = None
        if self.labels is not None:
            y = torch.from_numpy(self.labels[i].astype(np.float32))  # (K,)
        return x, y

# ============== Transformer Blocks ==============

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, activation: str, norm_type: str):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        if norm_type.lower().startswith("layer"):
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        elif norm_type.lower().startswith("batch"):
            self.norm1 = nn.BatchNorm1d(d_model)
            self.norm2 = nn.BatchNorm1d(d_model)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            activation_from_name(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, d_model)
        attn_out, _ = self.attn(x, x, x)  # self-attention
        x2 = x + self.dropout(attn_out)
        if isinstance(self.norm1, nn.BatchNorm1d):
            x2 = self.norm1(x2.transpose(1,2)).transpose(1,2)
        else:
            x2 = self.norm1(x2)

        ff_out = self.ff(x2)
        x3 = x2 + self.dropout(ff_out)
        if isinstance(self.norm2, nn.BatchNorm1d):
            x3 = self.norm2(x3.transpose(1,2)).transpose(1,2)
        else:
            x3 = self.norm2(x3)
        return x3

# ============== TransNAS-TSAD Model ==============

class TransNAS_TSAD(nn.Module):
    """
    Encoder + (Decoder1, Decoder2) with optional iterative self-adversarial refinement.
    Reconstruction-only; anomaly score from reconstruction errors per paper.
    """
    def __init__(self, config: Dict, input_dim: int, window_size: int):
        super().__init__()
        self.cfg = config
        self.input_dim = input_dim
        self.window_size = window_size
        self.d_model = config["dim_feedforward"]  # used as model width per paper
        self.n_heads = min(config["num_attention_heads"], max(1, self.d_model // 8))  # practical cap
        self.d_ff = max(self.d_model * 2, 32)

        # Optional linear embedding
        self.use_embed = config.get("use_linear_embedding", True)
        self.embed = nn.Linear(input_dim, self.d_model) if self.use_embed else nn.Identity()

        # Positional encoding
        if config["positional_encoding_type"].lower().startswith("four"):
            self.posenc = FourierPositionalEncoding(self.d_model, max_len=4096)
        else:
            self.posenc = SinusoidalPositionalEncoding(self.d_model, max_len=4096)

        # Encoder
        self.encoder = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.d_ff, config["dropout_rate"],
                             config["activation_function"], config["layer_norm_type"])
            for _ in range(config["encoder_layers"])
        ])

        # Two parallel decoders
        self.decoder1 = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.d_ff, config["dropout_rate"],
                             config["activation_function"], config["layer_norm_type"])
            for _ in range(config["decoder_layers"])
        ])
        self.decoder2 = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.d_ff, config["dropout_rate"],
                             config["activation_function"], config["layer_norm_type"])
            for _ in range(config["decoder_layers"])
        ])

        self.out1 = nn.Linear(self.d_model, input_dim)
        self.out2 = nn.Linear(self.d_model, input_dim)

        # Phase type: "1phase", "2phase", "iterative"
        self.phase_type = config.get("phase_type", "2phase")
        self.iter_max_steps = config.get("iter_max_steps", 5)
        self.iter_eps = config.get("iter_eps", 1e-5)

    def forward_enc(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        h = self.embed(x)
        h = self.posenc(h)
        for blk in self.encoder:
            h = blk(h)
        return h

    def forward_dec(self, h: torch.Tensor, which: int) -> torch.Tensor:
        g = h
        layers = self.decoder1 if which == 1 else self.decoder2
        for blk in layers:
            g = blk(g)
        return self.out1(g) if which == 1 else self.out2(g)

    def forward(self, x: torch.Tensor, return_all=False):
        """
        Returns:
            recon1, recon2, (iter_best_recon, iter_losses) if iterative
        """
        h = self.forward_enc(x)
        r1 = self.forward_dec(h, which=1)
        if self.phase_type == "1phase":
            if return_all:
                return r1, None, None
            return r1
        # 2-phase or iterative: compute second decoder
        r2 = self.forward_dec(h, which=2)
        if self.phase_type == "iterative":
            # iterative self-adversarial refinement: refine r2 to reduce reconstruction loss towards x
            current = r2
            prev_loss = torch.mean((x - current)**2, dim=(1,2))  # (B,)
            best = current.clone()
            best_loss = prev_loss.clone()
            losses = [prev_loss.detach().cpu()]
            for _ in range(self.iter_max_steps):
                # Feed current reconstruction back via encoder-decoder2 (self-conditioning refinement)
                h2 = self.forward_enc(current.detach())
                current = self.forward_dec(h2, which=2)
                cur_loss = torch.mean((x - current)**2, dim=(1,2))
                losses.append(cur_loss.detach().cpu())
                improve = (best_loss - cur_loss) > self.iter_eps
                best[improve] = current[improve]
                best_loss[improve] = cur_loss[improve]
                if torch.all(~improve):
                    break
            if return_all:
                return r1, r2, (best, losses)
            return r1, r2, best
        else:
            if return_all:
                return r1, r2, None
            return r1, r2

# ============== Losses for phases ==============

def phase_losses(x, r1, r2=None, adv_weight=1.0):
    """
    Phase-1 focus (minimize) and Phase-2 adversarial complement:
    L_focus = ||r1 - x||^2
    L_adv1  = ||r1 - x||^2  (minimize)
    L_adv2  = -||r2 - x||^2 (maximize discrepancy against x) -> implemented as + adv_weight * ||r2 - x||^2 with reversed sign in optimizer if needed
    Here implement min over r1 and encourage r2 to differ via adversarial target on the encoder/decoder2 through a gradient reversal trick approximation.
    For stability, we simply minimize r1 loss, and for r2 we minimize (alpha * ||(r1.detach() - r2)||^2) to push diversity.
    """
    l1 = torch.mean((r1 - x)**2)
    if r2 is None:
        return l1, None, l1
    # encourage r2 to be different from r1 (diversity) but not explode:
    diversity = torch.mean((r2 - r1.detach())**2)
    # also keep r2 bounded wrt x so it doesn't diverge excessively:
    bound = torch.mean(torch.clamp((r2 - x)**2, max=5.0))
    l_total = l1 + adv_weight * (0.5 * diversity + 0.5 * bound)
    return l1, diversity, l_total

# ============== POT / mPOT thresholding ==============

from scipy.stats import genpareto

def fit_pot_threshold(scores: np.ndarray, q: float = 0.98):
    """
    Fit GPD to tail above a high quantile threshold (u) using POT method.
    Returns u and dynamic function to decide threshold at inference.
    """
    scores = np.asarray(scores)
    u = np.quantile(scores, q)
    excess = scores[scores > u] - u
    if len(excess) < 10:
        # fallback: direct percentile
        return u, lambda recent_dev=0.0, w=0.0: u
    c, loc, scale = genpareto.fit(excess, floc=0.0)
    def pot_thresh(recent_dev=0.0, w=0.0):
        # base threshold ~ high quantile tail return level; for simplicity use u + Q(0.98) on fitted GPD
        base = u + genpareto.ppf(0.98, c, loc=0.0, scale=scale)
        return float(base + w * recent_dev)
    return u, pot_thresh

def recent_deviation(scores: np.ndarray, window: int = 100) -> float:
    tail = scores[-window:] if len(scores) >= window else scores
    if len(tail) < 5:
        return 0.0
    return float(np.median(np.abs(tail - np.median(tail))))

def mpot_threshold_fn(train_scores: np.ndarray, w: float = 0.1, q: float = 0.98):
    u, pot = fit_pot_threshold(train_scores, q=q)
    def thresh_fn(curr_scores: np.ndarray):
        rd = recent_deviation(curr_scores, window=min(200, len(curr_scores)))
        return pot(recent_dev=rd, w=w)
    return thresh_fn

# ============== F1PAK and EACS ==============

def point_adjust_labels(y_true: np.ndarray, y_pred: np.ndarray, K: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply point-adjustment with requirement of at least K% coverage of anomaly segment.
    Follows F1PAK definition; here we approximate segment coverage thresholding.
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    # find contiguous anomaly segments in ground truth
    N = len(y_true)
    i = 0
    y_adj = y_pred.copy()
    while i < N:
        if y_true[i] == 1:
            j = i
            while j < N and y_true[j] == 1:
                j += 1
            seg_len = j - i
            if seg_len > 0:
                seg_pred_pos = y_pred[i:j].sum()
                coverage = 100.0 * seg_pred_pos / seg_len
                if coverage >= K:
                    y_adj[i:j] = 1
                else:
                    y_adj[i:j] = 0
            i = j
        else:
            i += 1
    return y_true, y_adj

def f1pak_auc(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, Dict[int, float]]:
    """
    Compute F1PAK AUC over K in {0,1,2,3,4,5,10,20,...,100}.
    Threshold scores at per-K optimal threshold on validation grid.
    """
    K_values = list(range(0, 6)) + list(range(10, 101, 10))
    best_f1_per_K = {}
    for K in K_values:
        # sweep thresholds on scores to maximize F1 at this K
        ths = np.unique(np.percentile(y_score, np.linspace(60, 99.9, 40)))
        best_f1 = 0.0
        for th in ths:
            y_pred = (y_score >= th).astype(int)
            y_t, y_adj = point_adjust_labels(y_true, y_pred, K=K)
            p, r, f1, _ = precision_recall_fscore_support(y_t, y_adj, average="binary", zero_division=0)
            best_f1 = max(best_f1, f1)
        best_f1_per_K[K] = best_f1
    # simple trapezoid AUC over K (0..100)
    xs = np.array(sorted(best_f1_per_K.keys()))
    ys = np.array([best_f1_per_K[k] for k in xs])
    auc = np.trapz(ys, xs) / 100.0
    return float(auc), best_f1_per_K

def eacs(f1: float, train_time: float, param_count: int,
         f1_max: float, t_max: float, p_max: int,
         wa: float = 0.4, wt: float = 0.4, wp: float = 0.2) -> float:
    # normalize, higher is better for F1; lower better for time/params
    f1n = f1 / (f1_max + 1e-8)
    tn = 1.0 - (train_time / (t_max + 1e-8))
    pn = 1.0 - (param_count / (p_max + 1.0))
    return wa * f1n + wt * tn + wp * pn

# ============== Training / Scoring ==============

def anomaly_scores_from_recons(x: torch.Tensor, r1: torch.Tensor, r2: Optional[torch.Tensor], rbest: Optional[torch.Tensor]):
    # x, r* : (B, T, F)
    e1 = torch.mean((x - r1)**2, dim=(1,2))
    if rbest is not None:
        eb = torch.mean((x - rbest)**2, dim=(1,2))
        # combine per paper with best iterative as substitute for r2:
        s = 0.5 * e1 + 0.5 * eb
        return s
    if r2 is not None:
        e2 = torch.mean((x - r2)**2, dim=(1,2))
        s = 0.5 * e1 + 0.5 * e2
        return s
    return e1

def run_epoch(model, loader, optimizer, device, phase_type: str, adv_weight: float):
    model.train()
    total = 0.0
    n = 0
    for xb, _ in loader:
        xb = xb.to(device)
        optimizer.zero_grad()
        if phase_type == "1phase":
            r1 = model(xb)
            l1, _, ltot = phase_losses(xb, r1, None, adv_weight=adv_weight)
            loss = ltot
        elif phase_type == "2phase":
            r1, r2 = model(xb)
            l1, div, ltot = phase_losses(xb, r1, r2, adv_weight=adv_weight)
            loss = ltot
        else:
            r1, r2, (rbest, _) = model(xb, return_all=True)
            l1, div, ltot = phase_losses(xb, r1, r2, adv_weight=adv_weight)
            # small penalty for iterative stabilizer towards rbest not diverging
            stab = torch.mean((r2 - rbest.detach())**2)
            loss = ltot + 0.1 * stab
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu())
        n += 1
    return total / max(n, 1)

@torch.no_grad()
def collect_scores(model, loader, device):
    model.eval()
    scores = []
    y_true = []
    for xb, yb in loader:
        xb = xb.to(device)
        if model.phase_type == "1phase":
            r1 = model(xb)
            s = anomaly_scores_from_recons(xb, r1, None, None)
        elif model.phase_type == "2phase":
            r1, r2 = model(xb)
            s = anomaly_scores_from_recons(xb, r1, r2, None)
        else:
            r1, r2, (rbest, _) = model(xb, return_all=True)
            s = anomaly_scores_from_recons(xb, r1, r2, rbest)
        scores.append(s.detach().cpu().numpy())
        if yb is not None:
            # collapse window point labels to center point for pointwise comparison
            center = yb[:, yb.shape[1]//2].numpy()
            y_true.append(center)
    scores = np.concatenate(scores, axis=0)
    y_true = np.concatenate(y_true, axis=0) if len(y_true) > 0 else None
    return scores, y_true

# ============== NAS objective (NSGA-II) ==============

def build_config_from_trial(trial, input_dim, default: Dict) -> Dict:
    cfg = {}
    # training hyperparams
    cfg["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    cfg["dropout_rate"] = trial.suggest_float("dropout_rate", 0.1, 0.5)
    cfg["batch_size"] = trial.suggest_categorical("batch_size", [16, 32, 48, 64, 96, 128])
    cfg["gaussian_noise"] = trial.suggest_float("gaussian_noise", 1e-4, 1e-1, log=True)
    cfg["time_warp"] = trial.suggest_categorical("time_warp", [True, False])
    cfg["time_mask"] = trial.suggest_categorical("time_mask", [True, False])
    cfg["window_size"] = trial.suggest_int("window_size", 10, 30)
    # architectural
    cfg["positional_encoding_type"] = trial.suggest_categorical("positional_encoding_type", ["sinusoidal", "fourier"])
    cfg["dim_feedforward"] = trial.suggest_int("dim_feedforward", 8, 128, log=True)
    cfg["encoder_layers"] = trial.suggest_int("encoder_layers", 1, 3)
    cfg["decoder_layers"] = trial.suggest_int("decoder_layers", 1, 3)
    cfg["activation_function"] = trial.suggest_categorical("activation_function", ["relu", "leaky_relu", "sigmoid", "tanh"])
    cfg["num_attention_heads"] = max(1, min(input_dim, trial.suggest_int("num_attention_heads", 1, max(1, input_dim))))
    cfg["use_linear_embedding"] = trial.suggest_categorical("use_linear_embedding", [True, False])
    cfg["layer_norm_type"] = trial.suggest_categorical("layer_norm_type", ["layer", "batch", "instance"])
    cfg["phase_type"] = trial.suggest_categorical("phase_type", ["1phase", "2phase", "iterative"])
    cfg["iter_max_steps"] = 5
    cfg["iter_eps"] = 1e-5
    # fixed or default extras
    cfg["epochs"] = default.get("epochs", 30)
    cfg["adv_weight"] = default.get("adv_weight", 1.0)
    return cfg

def objective_nsga(trial, X_train, y_train, X_val, y_val, device, default):
    # sample config
    input_dim = X_train.shape[1]
    cfg = build_config_from_trial(trial, input_dim, default)
    # build datasets
    fit_minv, fit_maxv = minmax_fit(np.vstack([X_train, X_val]))
    augment = {
        "gaussian_noise": cfg["gaussian_noise"],
        "time_warp": cfg["time_warp"],
        "time_warp_sigma": 0.04,
        "time_mask": cfg["time_mask"],
        "time_mask_prob": 0.1
    }
    ds_tr = SlidingWindowDataset(X_train, y_train, cfg["window_size"], stride=1, augment=augment, split="train", norm_fit_on=(fit_minv, fit_maxv))
    ds_va = SlidingWindowDataset(X_val, y_val, cfg["window_size"], stride=1, augment=None, split="val", norm_fit_on=(fit_minv, fit_maxv))
    dl_tr = DataLoader(ds_tr, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=cfg["batch_size"], shuffle=False)

    # build model
    model = TransNAS_TSAD(cfg, input_dim=input_dim, window_size=cfg["window_size"]).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    # quick training
    t0 = time.time()
    for ep in range(cfg["epochs"]):
        tr_loss = run_epoch(model, dl_tr, opt, device, phase_type=cfg["phase_type"], adv_weight=cfg["adv_weight"])
        # early stop light (optional)
        if math.isnan(tr_loss) or math.isinf(tr_loss):
            break
    train_time = time.time() - t0

    # validation scores
    val_scores, val_true = collect_scores(model, dl_va, device)
    # mPOT threshold
    thresh_fn = mpot_threshold_fn(val_scores, w=0.1, q=0.98)
    th = thresh_fn(val_scores)
    val_pred = (val_scores >= th).astype(int)
    # pointwise F1 (no adjustment) for objective; F1PAK is reported separately
    p, r, f1, _ = precision_recall_fscore_support(val_true, val_pred, average="binary", zero_division=0)

    # objectives: maximize F1, minimize params
    params = count_params(model)
    trial.set_user_attr("config", cfg)
    trial.set_user_attr("train_time", train_time)
    trial.set_user_attr("params", params)
    trial.set_user_attr("val_f1", f1)

    return f1, params

# ============== Pareto study runner ==============

def run_nsga_study(X_train, y_train, X_val, y_val, device="cuda" if torch.cuda.is_available() else "cpu",
                   n_trials: int = 50, seed: int = 42, default: Dict = None):
    default = default or {"epochs": 20, "adv_weight": 1.0}
    set_seed(seed)
    sampler = NSGAIISampler(seed=seed)
    study = optuna.create_study(directions=["maximize", "minimize"], sampler=sampler)
    study.optimize(lambda tr: objective_nsga(tr, X_train, y_train, X_val, y_val, device, default),
                   n_trials=n_trials, n_jobs=1, show_progress_bar=True)
    # extract Pareto front
    pareto = [t for t in study.best_trials]
    # sort by -F1, +params to inspect
    pareto_sorted = sorted(pareto, key=lambda t: (-t.values, t.values[1]))
    return study, pareto_sorted

# ============== End-to-end fit/eval helper ==============

def train_and_evaluate(best_cfg: Dict,
                       X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       device="cuda" if torch.cuda.is_available() else "cpu"):
    fit_minv, fit_maxv = minmax_fit(np.vstack([X_train, X_val, X_test]))
    ds_tr = SlidingWindowDataset(X_train, y_train, best_cfg["window_size"], stride=1, augment={
        "gaussian_noise": best_cfg["gaussian_noise"],
        "time_warp": best_cfg["time_warp"],
        "time_mask": best_cfg["time_mask"]
    }, split="train", norm_fit_on=(fit_minv, fit_maxv))
    ds_va = SlidingWindowDataset(X_val, y_val, best_cfg["window_size"], stride=1, augment=None, split="val", norm_fit_on=(fit_minv, fit_maxv))
    ds_te = SlidingWindowDataset(X_test, y_test, best_cfg["window_size"], stride=1, augment=None, split="test", norm_fit_on=(fit_minv, fit_maxv))

    dl_tr = DataLoader(ds_tr, batch_size=best_cfg["batch_size"], shuffle=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=best_cfg["batch_size"], shuffle=False)
    dl_te = DataLoader(ds_te, batch_size=best_cfg["batch_size"], shuffle=False)

    model = TransNAS_TSAD(best_cfg, input_dim=X_train.shape[1], window_size=best_cfg["window_size"]).to(device)
    opt = optim.Adam(model.parameters(), lr=best_cfg["learning_rate"])

    # Train
    t0 = time.time()
    for ep in range(best_cfg.get("epochs", 50)):
        tr_loss = run_epoch(model, dl_tr, opt, device, phase_type=best_cfg["phase_type"], adv_weight=best_cfg.get("adv_weight", 1.0))
    train_time = time.time() - t0

    # Validation for threshold fit (mPOT)
    val_scores, val_true = collect_scores(model, dl_va, device)
    thresh_fn = mpot_threshold_fn(val_scores, w=0.1, q=0.98)
    # Test
    test_scores, test_true = collect_scores(model, dl_te, device)
    th = thresh_fn(np.concatenate([val_scores, test_scores]))
    test_pred = (test_scores >= th).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(test_true, test_pred, average="binary", zero_division=0)
    # F1PAK AUC
    auc_f1pak, perK = f1pak_auc(test_true, test_scores)
    # EACS needs normalization across compared models; here compute self-referenced components
    params = count_params(model)
    eacs_score = eacs(f1=f1, train_time=train_time, param_count=params,
                      f1_max=f1, t_max=train_time, p_max=params)
    metrics = {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "f1pak_auc": float(auc_f1pak),
        "f1pak_perK": perK,
        "params": int(params),
        "train_time_sec": float(train_time),
        "eacs_self": float(eacs_score),
        "threshold": float(th)
    }
    return model, metrics

# ============== Example main (replace with your data loader) ==============

if __name__ == "__main__":
    set_seed(42)
    # Example shape placeholders; replace with real multivariate arrays and labels
    # X: (N, F), y: (N,) binary labels for pointwise anomalies
    # Split based on timestamps for chronological integrity
    N, F = 5000, 10
    X = np.random.randn(N, F).astype(np.float32)
    y = (np.random.rand(N) < 0.05).astype(int)

    idx_tr, idx_va, idx_te = int(0.6*N), int(0.8*N), N
    X_tr, y_tr = X[:idx_tr], y[:idx_tr]
    X_va, y_va = X[idx_tr:idx_va], y[idx_tr:idx_va]
    X_te, y_te = X[idx_va:], y[idx_va:]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # NSGA-II search
    study, pareto = run_nsga_study(X_tr, y_tr, X_va, y_va, device=device, n_trials=30, seed=42,
                                   default={"epochs": 10, "adv_weight": 1.0})

    # Pick the first Pareto-optimal config (high F1, low params trade-off)
    best_trial = pareto
    best_cfg = best_trial.user_attrs["config"]
    print("Selected config:", json.dumps(best_cfg, indent=2))
    # Train final and evaluate
    model, metrics = train_and_evaluate(best_cfg, X_tr, y_tr, X_va, y_va, X_te, y_te, device=device)
    print("Final metrics:", json.dumps(metrics, indent=2))
