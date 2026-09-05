# Probabilistic Bike Demand Forecasting

An end-to-end machine-learning project for hourly bike-demand forecasting. The project is designed to demonstrate the skills expected in a strong ML internship application: temporal validation, leakage-safe features, robust baselines, nonlinear models, uncertainty quantification, interpretability, testing, and a small interactive application.

## Why this project is credible

- The split is chronological: train, calibration, then test.
- Every target-derived feature is lagged, so no future demand enters training features.
- A seasonal-naive baseline is evaluated alongside the ML model.
- The point model uses a Poisson loss suited to non-negative count data.
- Prediction intervals are calibrated on a held-out time window using split conformal prediction.
- Metrics include MAE, RMSE, RMSLE, interval coverage and interval width.
- Permutation importance is computed only on the untouched test period.

## Verified holdout results

The checked-in artifacts were produced from the real UCI hourly dataset. The final chronological test window runs from 8 August to 31 December 2012.

| Model | MAE | RMSE | RMSLE |
| --- | ---: | ---: | ---: |
| Seasonal naive (`t-168h`) | 78.66 | 138.81 | 0.808 |
| Poisson histogram gradient boosting | **73.98** | **110.34** | **0.445** |

Compared with the weekly baseline, the model reduces MAE by 6.0%, RMSE by 20.5% and RMSLE by 44.9%. The nominal 90% conformal interval reaches 85.4% coverage on the shifted test period. This under-coverage is retained and discussed as a limitation rather than hidden.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m bikeshare_forecast.train --output-dir artifacts --data-path data/uci/hour.csv
```

The repository includes the official UCI Capital Bikeshare hourly data (17,379 observations from 2011-2012). The loader explicitly excludes `casual` and `registered`, because they are components of the target `cnt` and would cause direct target leakage.

An autonomous demo mode is also available. It generates a reproducible two-year hourly series with trend, annual and weekly seasonality, commuting peaks, weather effects, holidays, autocorrelation and rare demand shocks:

```bash
python -m bikeshare_forecast.train --output-dir artifacts_demo --data-path ""
```

Launch the dashboard after training:

```bash
streamlit run app.py
```

Run the tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Real-data extension

The included dataset is loaded through the schema adapter in `data.py`. The normalized UCI weather variables are converted back to their documented physical scales. Time-series observations are never randomly shuffled.

Dataset reference: Fanaee-T, H. and Gama, J. (2014), *Bike Sharing*, UCI Machine Learning Repository.

## Repository structure

```text
.
├── app.py
├── pyproject.toml
├── requirements.txt
├── src/bikeshare_forecast/
│   ├── data.py
│   ├── evaluation.py
│   ├── features.py
│   ├── modeling.py
│   └── train.py
└── tests/
    ├── test_features.py
    └── test_pipeline.py
```

## Interview discussion points

1. Why random cross-validation leaks information in time-series problems.
2. Why a weekly seasonal baseline can be hard to beat.
3. Why a Poisson objective is natural for demand counts.
4. How conformal calibration turns residuals into empirically valid intervals under exchangeability assumptions.
5. What changes under distribution shift: strikes, new stations, pricing changes or extreme weather.
6. How to monitor MAE, interval coverage, feature drift and residual drift in production.

## Honest CV wording

Use this only after running the experiments and being able to explain the code:

> **Probabilistic forecasting of hourly bike demand - Python, scikit-learn**  
> Built a leakage-safe time-series ML pipeline with seasonal baselines and gradient boosting; calibrated 90% conformal prediction intervals and evaluated accuracy, uncertainty coverage and feature importance on a chronological holdout.

After you have run and understood the project, you can add: “Reduced holdout RMSE by 20.5% versus a weekly seasonal baseline on 17k+ real hourly observations.”

## Sources

- UCI Bike Sharing dataset: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
- scikit-learn time-series cross-validation: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- scikit-learn gradient boosting prediction intervals: https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html
- scikit-learn permutation importance: https://scikit-learn.org/stable/modules/permutation_importance.html
