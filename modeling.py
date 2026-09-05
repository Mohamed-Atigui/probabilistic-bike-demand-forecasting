from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import FEATURE_COLUMNS


def make_point_model(random_state: int = 42) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.07,
        max_iter=260,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=0.8,
        random_state=random_state,
    )


def fit_model(train: pd.DataFrame) -> HistGradientBoostingRegressor:
    model = make_point_model()
    model.fit(train[FEATURE_COLUMNS], train["demand"])
    return model


def conformal_radius(y_true: np.ndarray, y_pred: np.ndarray, coverage: float = 0.90) -> float:
    """Finite-sample split-conformal radius for symmetric absolute-residual intervals."""
    if not 0 < coverage < 1:
        raise ValueError("coverage must be in (0, 1)")
    residuals = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    n = residuals.size
    quantile_level = min(1.0, np.ceil((n + 1) * coverage) / n)
    return float(np.quantile(residuals, quantile_level, method="higher"))


def predict_with_interval(model, frame: pd.DataFrame, radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred = np.maximum(0.0, model.predict(frame[FEATURE_COLUMNS]))
    lower = np.maximum(0.0, pred - radius)
    upper = pred + radius
    return pred, lower, upper

