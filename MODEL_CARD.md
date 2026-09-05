# Model card

## Intended use

Educational forecasting prototype for hourly bike-rental demand. It demonstrates chronological validation, count regression and uncertainty calibration. It is not intended to allocate a real transport fleet without retraining, cost-sensitive evaluation and operational review.

## Data

- Source: UCI Bike Sharing dataset, Capital Bikeshare, Washington, D.C.
- Period: 2011-2012.
- Unit: one hourly observation.
- Target: total hourly rentals (`cnt`).
- Leakage control: `casual` and `registered` are excluded because their sum equals the target.

## Validation design

Observations are ordered by time and divided into 65% training, 15% conformal calibration and 20% testing. No random shuffle is used. Demand lags are matched by exact timestamp (`t-24h` and `t-168h`) rather than by row position, because the source series contains missing hours. The rolling feature uses only observations in `[t-24h, t)`.

## Model

Histogram gradient boosting with Poisson loss. Predictions are clipped at zero. Symmetric split-conformal intervals are calibrated from absolute residuals on the calibration window.

## Results

| Metric | ML model | Weekly seasonal baseline |
| --- | ---: | ---: |
| MAE | 48.65 | 63.95 |
| RMSE | 79.98 | 109.62 |
| RMSLE | 0.364 | 0.586 |

The nominal 90% interval has 89.8% empirical test coverage and a mean width of 229.95 rentals.

## Limitations

- Weather observations are treated as known; a live system would use weather forecasts and inherit their uncertainty.
- The symmetric interval is marginal rather than conditional: coverage may vary by hour, season or demand level.
- Demand from 2011-2012 is not representative of current mobility patterns.
- Aggregate demand does not solve station-level rebalancing.
- Hyperparameters are fixed; a nested time-series validation study would be needed before deployment.

## Monitoring recommendations

Track rolling MAE, interval coverage, residual bias, weather-feature drift and demand-lag drift. Trigger retraining or recalibration when coverage or residual distributions deteriorate materially.
