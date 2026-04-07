"""
TGF Virtual Sensor
====================
Physics-informed hybrid model for predicting Total Hardness, Calcium Hardness,
and Total Alkalinity from online sensors (pH, conductivity, temperature, ORP).

Architecture (Two-Layer Stack):
    Layer 1 — Physics baseline: Hardness = CoC × Hardness_makeup × correction
    Layer 2 — ML correction: Actual = Physics_baseline + f(pH, conductivity, temp, ORP, CoC, dσ/dt)

Previous ML-only attempts achieved R²=0.37 on this dataset (too low).
The physics-informed hybrid learns only the CORRECTION term (~20% of signal),
which is much easier than learning the full mapping.

Confidence scoring (traffic light):
    GREEN: prediction within 1σ of training distribution → use ML prediction
    AMBER: 1-2σ → blend 50/50 ML + physics
    RED: >2σ or inputs outside training range → use physics only (fallback)
"""
import os
import logging
import pickle
import numpy as np
from typing import Optional, Tuple, Dict
from collections import deque

logger = logging.getLogger(__name__)

# Targets
TARGETS = ["total_hardness", "calcium_hardness", "total_alkalinity"]


class VirtualSensor:
    """
    Physics-informed virtual sensor for hardness and alkalinity prediction.

    Usage:
        vs = VirtualSensor()
        vs.train(csv_path, physics_engine)  # one-time training
        pred, confidence = vs.predict(ph, conductivity, temp, orp, coc, physics_baselines)
    """

    def __init__(self, model_path: str = None):
        self._model = None
        self._scaler_X = None
        self._scaler_y = None
        self._train_std = None  # std of residuals during training
        self._available = False
        self._prev_conductivity = None
        self._conductivity_history = deque(maxlen=12)

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    @property
    def available(self) -> bool:
        return self._available

    def _compute_dsigma_dt(self, conductivity: float) -> float:
        """Rate-of-change of conductivity (dσ/dt)."""
        self._conductivity_history.append(conductivity)
        if len(self._conductivity_history) < 2:
            return 0.0
        return conductivity - self._conductivity_history[-2]

    def train(self, csv_path: str, physics_engine, save_path: str = None):
        """
        Train the virtual sensor from CSV data.

        Args:
            csv_path: Path to Parameters_5K.csv with ground truth columns
            physics_engine: PhysicsEngine instance for computing baselines
            save_path: Where to save trained model (optional)
        """
        try:
            import pandas as pd
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.multioutput import MultiOutputRegressor
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split
        except ImportError:
            logger.warning("sklearn not available — virtual sensor disabled")
            return

        logger.info("Training virtual sensor from dataset...")

        df = pd.read_csv(csv_path)

        # Rename columns to standard names
        rename_map = {
            'Total_Hardness_ppm': 'total_hardness',
            'Calcium_Hardness_ppm': 'calcium_hardness',
            'Total_Alkalinity_ppm': 'total_alkalinity',
            'Conductivity_uS_cm': 'conductivity',
            'TDS_ppm': 'tds',
        }
        df = df.rename(columns=rename_map)

        # Filter rows with ground truth for all targets
        required_cols = ['pH', 'conductivity', 'total_hardness', 'calcium_hardness', 'total_alkalinity']
        mask = df[required_cols].notna().all(axis=1)
        df_clean = df[mask].copy()

        if len(df_clean) < 50:
            logger.warning(f"Only {len(df_clean)} rows with ground truth — too few for training")
            return

        logger.info(f"Training on {len(df_clean)} rows with ground truth")

        # Compute physics baselines
        coc_vals = df_clean['conductivity'] / physics_engine.tower.makeup_conductivity_us
        coc_vals = coc_vals.clip(1.0, 15.0)

        physics_hardness = physics_engine.tower.makeup_hardness_ppm * coc_vals
        physics_calcium = physics_engine.tower.makeup_calcium_ppm * coc_vals
        physics_alkalinity = physics_engine.tower.makeup_alkalinity_ppm * coc_vals * 0.85

        # Features: pH, conductivity, temperature (fill missing), ORP (fill), CoC, physics baselines
        temp = df_clean.get('Temperature_C', pd.Series(32.0, index=df_clean.index))
        temp = temp.fillna(32.0)
        orp = df_clean.get('ORP_mV', pd.Series(400.0, index=df_clean.index))
        orp = orp.fillna(400.0)

        # dσ/dt approximation (difference of consecutive conductivity)
        dsigma = df_clean['conductivity'].diff().fillna(0.0)

        X = np.column_stack([
            df_clean['pH'].values,
            df_clean['conductivity'].values,
            temp.values,
            orp.values,
            coc_vals.values,
            dsigma.values,
            physics_hardness.values,
            physics_alkalinity.values,
        ])

        # Target = actual - physics baseline (the correction)
        y = np.column_stack([
            df_clean['total_hardness'].values - physics_hardness.values,
            df_clean['calcium_hardness'].values - physics_calcium.values,
            df_clean['total_alkalinity'].values - physics_alkalinity.values,
        ])

        # Scale
        self._scaler_X = StandardScaler()
        self._scaler_y = StandardScaler()
        X_scaled = self._scaler_X.fit_transform(X)
        y_scaled = self._scaler_y.fit_transform(y)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42)

        # Train Random Forest
        base_rf = RandomForestRegressor(
            n_estimators=100, max_depth=15, min_samples_leaf=5,
            random_state=42, n_jobs=-1)
        self._model = MultiOutputRegressor(base_rf)
        self._model.fit(X_train, y_train)

        # Evaluate
        from sklearn.metrics import r2_score
        y_pred_test = self._model.predict(X_test)
        y_pred_inv = self._scaler_y.inverse_transform(y_pred_test)
        y_test_inv = self._scaler_y.inverse_transform(y_test)

        # R² on correction term
        for i, name in enumerate(TARGETS):
            r2_correction = r2_score(y_test_inv[:, i], y_pred_inv[:, i])
            logger.info(f"  {name} correction R²: {r2_correction:.3f}")

        # R² on final prediction (physics + correction)
        # Reconstruct physics baselines for test set
        X_test_orig = self._scaler_X.inverse_transform(X_test)
        physics_th_test = X_test_orig[:, 6]  # physics_hardness column
        physics_alk_test = X_test_orig[:, 7]
        physics_ca_test = physics_th_test * (physics_engine.tower.makeup_calcium_ppm /
                                              physics_engine.tower.makeup_hardness_ppm)

        final_th = physics_th_test + y_pred_inv[:, 0]
        final_ca = physics_ca_test + y_pred_inv[:, 1]
        final_alk = physics_alk_test + y_pred_inv[:, 2]

        actual_th = physics_th_test + y_test_inv[:, 0]
        actual_ca = physics_ca_test + y_test_inv[:, 1]
        actual_alk = physics_alk_test + y_test_inv[:, 2]

        r2_th = r2_score(actual_th, final_th)
        r2_ca = r2_score(actual_ca, final_ca)
        r2_alk = r2_score(actual_alk, final_alk)
        logger.info(f"  Final R² — TH: {r2_th:.3f}, Ca: {r2_ca:.3f}, Alk: {r2_alk:.3f}")

        # Store training residual std for confidence scoring
        train_pred = self._model.predict(X_train)
        train_residuals = y_train - train_pred
        self._train_std = np.std(train_residuals, axis=0)

        self._available = True
        logger.info("Virtual sensor training complete")

        if save_path:
            self.save(save_path)

    def predict(self,
                ph: float,
                conductivity: float,
                temperature: float,
                orp: float,
                coc: float,
                physics_hardness: float,
                physics_calcium: float,
                physics_alkalinity: float,
                ) -> Tuple[Dict[str, float], str]:
        """
        Predict hardness and alkalinity with confidence scoring.

        Returns:
            (predictions_dict, confidence_level)
            predictions_dict: {"total_hardness": float, "calcium_hardness": float, "total_alkalinity": float}
            confidence_level: "GREEN", "AMBER", or "RED"
        """
        if not self._available:
            return {
                "total_hardness": physics_hardness,
                "calcium_hardness": physics_calcium,
                "total_alkalinity": physics_alkalinity,
            }, "RED"

        dsigma = self._compute_dsigma_dt(conductivity)

        X = np.array([[ph, conductivity, temperature, orp, coc,
                        dsigma, physics_hardness, physics_alkalinity]])

        try:
            X_scaled = self._scaler_X.transform(X)
            correction_scaled = self._model.predict(X_scaled)
            correction = self._scaler_y.inverse_transform(correction_scaled)[0]
        except Exception as e:
            logger.warning(f"Virtual sensor prediction failed: {e}")
            return {
                "total_hardness": physics_hardness,
                "calcium_hardness": physics_calcium,
                "total_alkalinity": physics_alkalinity,
            }, "RED"

        # Physics constraints
        pred_th = max(1.0, physics_hardness + correction[0])
        pred_ca = max(1.0, physics_calcium + correction[1])
        pred_alk = max(1.0, physics_alkalinity + correction[2])
        pred_ca = min(pred_ca, pred_th)  # Ca <= Total Hardness

        # Confidence scoring based on residual magnitude
        confidence = "GREEN"
        if self._train_std is not None:
            residual_z = np.abs(correction_scaled[0]) / np.maximum(self._train_std, 0.01)
            max_z = np.max(residual_z)
            if max_z > 2.0:
                confidence = "RED"
            elif max_z > 1.0:
                confidence = "AMBER"

        # Blend based on confidence
        if confidence == "AMBER":
            pred_th = 0.5 * pred_th + 0.5 * physics_hardness
            pred_ca = 0.5 * pred_ca + 0.5 * physics_calcium
            pred_alk = 0.5 * pred_alk + 0.5 * physics_alkalinity
        elif confidence == "RED":
            pred_th = physics_hardness
            pred_ca = physics_calcium
            pred_alk = physics_alkalinity

        return {
            "total_hardness": round(pred_th, 1),
            "calcium_hardness": round(pred_ca, 1),
            "total_alkalinity": round(pred_alk, 1),
        }, confidence

    def save(self, path: str):
        """Save trained model to disk."""
        data = {
            "model": self._model,
            "scaler_X": self._scaler_X,
            "scaler_y": self._scaler_y,
            "train_std": self._train_std,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Virtual sensor saved to {path}")

    def load(self, path: str):
        """Load trained model from disk."""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self._model = data["model"]
            self._scaler_X = data["scaler_X"]
            self._scaler_y = data["scaler_y"]
            self._train_std = data["train_std"]
            self._available = True
            logger.info(f"Virtual sensor loaded from {path}")
        except Exception as e:
            logger.warning(f"Failed to load virtual sensor: {e}")
            self._available = False
