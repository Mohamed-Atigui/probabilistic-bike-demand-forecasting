from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Bike Demand Forecasting", layout="wide")
st.title("Probabilistic Bike Demand Forecasting")
st.caption("Leakage-safe hourly forecasting with calibrated 90% prediction intervals")

artifact_dir = Path("artifacts")
metrics_path = artifact_dir / "metrics.json"
predictions_path = artifact_dir / "test_predictions.csv"

if not metrics_path.exists() or not predictions_path.exists():
    st.warning("Train the model first: `python -m bikeshare_forecast.train --output-dir artifacts`")
    st.stop()

metrics = pd.read_json(metrics_path, typ="series")
pred = pd.read_csv(predictions_path, parse_dates=["timestamp"])

model_metrics = metrics["model"]
interval = metrics["prediction_interval_90"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Test MAE", f"{model_metrics['mae']:.1f}")
c2.metric("Test RMSE", f"{model_metrics['rmse']:.1f}")
c3.metric("90% coverage", f"{100 * interval['coverage']:.1f}%")
c4.metric("Mean interval width", f"{interval['mean_interval_width']:.1f}")

days = st.slider("Number of final test days", 3, 30, 14)
view = pred.tail(days * 24).set_index("timestamp")
st.line_chart(view[["demand", "prediction", "lower_90", "upper_90"]])

st.subheader("Recent predictions")
st.dataframe(view.tail(48), use_container_width=True)

