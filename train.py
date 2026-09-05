from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .data import generate_demo_data, load_uci_hourly
from .evaluation import interval_metrics, regression_metrics
from .features import FEATURE_COLUMNS, build_features, chronological_split
from .modeling import conformal_radius, fit_model, predict_with_interval


def run_experiment(
    output_dir: str | Path = "artifacts",
    n_hours: int = 24 * 365 * 2,
    data_path: str | Path | None = None,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if data_path is not None:
        raw = load_uci_hourly(data_path)
        data_source = "UCI Bike Sharing - Capital Bikeshare hourly data"
    else:
        raw = generate_demo_data(n_hours=n_hours)
        data_source = "reproducible synthetic demonstration data"
    featured = build_features(raw)
    train, calibration, test = chronological_split(featured)
    model = fit_model(train)

    calibration_pred = np.maximum(0.0, model.predict(calibration[FEATURE_COLUMNS]))
    radius = conformal_radius(calibration["demand"].to_numpy(), calibration_pred, coverage=0.90)
    pred, lower, upper = predict_with_interval(model, test, radius)
    baseline = test["demand_lag_168"].to_numpy()

    metrics = {
        "split": {
            "data_source": data_source,
            "train_rows": len(train),
            "calibration_rows": len(calibration),
            "test_rows": len(test),
            "test_start": str(test["timestamp"].min()),
            "test_end": str(test["timestamp"].max()),
        },
        "model": regression_metrics(test["demand"], pred),
        "seasonal_naive": regression_metrics(test["demand"], baseline),
        "prediction_interval_90": {
            **interval_metrics(test["demand"], lower, upper),
            "conformal_radius": radius,
        },
    }

    scored = test[["timestamp", "demand"]].copy()
    scored["prediction"] = pred
    scored["lower_90"] = lower
    scored["upper_90"] = upper
    scored["seasonal_naive"] = baseline
    scored.to_csv(output / "test_predictions.csv", index=False)
    with (output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    joblib.dump({"model": model, "radius": radius, "features": FEATURE_COLUMNS}, output / "model.joblib")

    sample = test.tail(24 * 14)
    sample_pred = pred[-len(sample):]
    sample_lower = lower[-len(sample):]
    sample_upper = upper[-len(sample):]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(sample["timestamp"], sample["demand"], label="Observed", color="#1f2937", linewidth=1.5)
    ax.plot(sample["timestamp"], sample_pred, label="Forecast", color="#b54741", linewidth=1.4)
    ax.fill_between(sample["timestamp"], sample_lower, sample_upper, color="#b54741", alpha=0.18, label="90% interval")
    ax.set(title="Hourly bike demand - final 14 test days", ylabel="Bike rentals")
    ax.legend(frameon=False, ncol=3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output / "forecast.png", dpi=160)
    plt.close(fig)

    perm_sample = test.tail(min(2000, len(test)))
    importance = permutation_importance(
        model,
        perm_sample[FEATURE_COLUMNS],
        perm_sample["demand"],
        scoring="neg_mean_absolute_error",
        n_repeats=5,
        random_state=42,
    )
    importance_df = pd.DataFrame(
        {"feature": FEATURE_COLUMNS, "importance_mean": importance.importances_mean, "importance_std": importance.importances_std}
    ).sort_values("importance_mean", ascending=False)
    importance_df.to_csv(output / "feature_importance.csv", index=False)

    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the bike-demand forecasting pipeline")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--n-hours", type=int, default=24 * 365 * 2)
    parser.add_argument("--data-path", default="data/uci/hour.csv", help="UCI hour.csv; pass an empty string for demo data")
    args = parser.parse_args()
    path = Path(args.data_path) if args.data_path else None
    run_experiment(args.output_dir, args.n_hours, data_path=path)


if __name__ == "__main__":
    main()
