from __future__ import annotations

import numpy as np
import pandas as pd

from .data import validate_schema


FEATURE_COLUMNS = [
    "temperature_c",
    "humidity_pct",
    "wind_kmh",
    "weather_severity",
    "is_holiday",
    "is_weekend",
    "trend_hours",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "year_sin",
    "year_cos",
    "demand_lag_24",
    "demand_lag_168",
    "demand_rolling_24",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time and lag features using only information available at prediction time."""
    validate_schema(df)
    out = df.copy()
    ts = out["timestamp"]
    hour = ts.dt.hour
    dow = ts.dt.dayofweek
    doy = ts.dt.dayofyear

    out["is_weekend"] = (dow >= 5).astype(int)
    out["trend_hours"] = (ts - ts.min()).dt.total_seconds() / 3600
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["year_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["year_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["demand_lag_24"] = out["demand"].shift(24)
    out["demand_lag_168"] = out["demand"].shift(168)
    out["demand_rolling_24"] = out["demand"].shift(1).rolling(24).mean()
    return out.dropna().reset_index(drop=True)


def chronological_split(
    featured: pd.DataFrame,
    train_fraction: float = 0.65,
    calibration_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0 < calibration_fraction < 1 - train_fraction:
        raise ValueError("calibration_fraction leaves no test set")
    n = len(featured)
    train_end = int(n * train_fraction)
    calibration_end = int(n * (train_fraction + calibration_fraction))
    return (
        featured.iloc[:train_end].copy(),
        featured.iloc[train_end:calibration_end].copy(),
        featured.iloc[calibration_end:].copy(),
    )
