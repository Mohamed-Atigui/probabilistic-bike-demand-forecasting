from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


REQUIRED_COLUMNS = {
    "timestamp",
    "temperature_c",
    "humidity_pct",
    "wind_kmh",
    "weather_severity",
    "is_holiday",
    "demand",
}


def generate_demo_data(n_hours: int = 24 * 365 * 2, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic hourly demand series for a fully reproducible demo."""
    if n_hours < 24 * 30:
        raise ValueError("n_hours must cover at least 30 days")

    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n_hours, freq="h")
    hour = ts.hour.to_numpy()
    dow = ts.dayofweek.to_numpy()
    doy = ts.dayofyear.to_numpy()

    annual = np.sin(2 * np.pi * (doy - 80) / 365.25)
    temperature = 13 + 11 * annual + rng.normal(0, 3.5, n_hours)
    humidity = np.clip(65 - 0.8 * temperature + rng.normal(0, 10, n_hours), 18, 100)
    wind = np.clip(rng.gamma(2.0, 4.0, n_hours), 0, 45)
    rainy = rng.random(n_hours) < np.clip(0.16 + 0.003 * (humidity - 60), 0.05, 0.45)
    precipitation = rainy * rng.gamma(1.4, 1.8, n_hours)

    weekend = dow >= 5
    holiday = ((ts.month == 1) & (ts.day == 1)) | ((ts.month == 12) & (ts.day == 25))
    commute = 90 * np.exp(-0.5 * ((hour - 8) / 1.6) ** 2) + 105 * np.exp(-0.5 * ((hour - 18) / 2.0) ** 2)
    leisure = 65 * np.exp(-0.5 * ((hour - 14) / 3.2) ** 2)
    hourly_effect = np.where(weekend, leisure, commute)
    weather_effect = 4.2 * np.clip(temperature, -5, 24) - 8.0 * precipitation - 0.75 * wind
    trend = np.linspace(0, 32, n_hours)
    weekly = 12 * np.sin(2 * np.pi * np.arange(n_hours) / (24 * 7))
    latent = 48 + hourly_effect + weather_effect + trend + weekly - 45 * holiday

    noise = np.zeros(n_hours)
    innovations = rng.normal(0, 13, n_hours)
    for i in range(1, n_hours):
        noise[i] = 0.55 * noise[i - 1] + innovations[i]
    shock = (rng.random(n_hours) < 0.004) * rng.normal(-55, 20, n_hours)
    demand = np.maximum(0, np.rint(latent + noise + shock)).astype(int)

    return pd.DataFrame(
        {
            "timestamp": ts,
            "temperature_c": temperature.round(2),
            "humidity_pct": humidity.round(2),
            "wind_kmh": wind.round(2),
            "weather_severity": np.select(
                [precipitation > 4, precipitation > 0, humidity > 85],
                [4, 3, 2],
                default=1,
            ),
            "is_holiday": holiday.astype(int),
            "demand": demand,
        }
    )


def load_uci_hourly(path: str | Path) -> pd.DataFrame:
    """Load the official UCI Capital Bikeshare hourly file without target leakage."""
    raw = pd.read_csv(path)
    required = {"dteday", "hr", "temp", "hum", "windspeed", "weathersit", "holiday", "cnt"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"UCI file is missing columns: {sorted(missing)}")
    timestamp = pd.to_datetime(raw["dteday"]) + pd.to_timedelta(raw["hr"], unit="h")
    frame = pd.DataFrame(
        {
            "timestamp": timestamp,
            "temperature_c": raw["temp"] * 41.0,
            "humidity_pct": raw["hum"] * 100.0,
            "wind_kmh": raw["windspeed"] * 67.0,
            "weather_severity": raw["weathersit"].astype(int),
            "is_holiday": raw["holiday"].astype(int),
            "demand": raw["cnt"].astype(int),
        }
    ).sort_values("timestamp", ignore_index=True)
    validate_schema(frame)
    return frame


def validate_schema(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        raise TypeError("timestamp must be datetime64")
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("timestamps must be sorted")
    if df["timestamp"].duplicated().any():
        raise ValueError("timestamps must be unique")
    if (df["demand"] < 0).any():
        raise ValueError("demand must be non-negative")
