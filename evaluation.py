from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.maximum(0.0, np.asarray(y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "rmsle": float(np.sqrt(mean_squared_log_error(y_true, y_pred))),
    }


def interval_metrics(y_true, lower, upper) -> dict[str, float]:
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    return {
        "coverage": float(np.mean((y_true >= lower) & (y_true <= upper))),
        "mean_interval_width": float(np.mean(upper - lower)),
    }

