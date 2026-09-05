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

Observations are ordered by time and divided into 65% training, 15% conformal calibration and 20% testing. No random shuffle is used. Lagged demand features use only observations available at least 24 hours before the forecast time.

## Model

Histogram gradient boosting with Poisson loss. Predictions are clipped at zero. Symmetric split-conformal intervals are calibrated from absolute residuals on the calibration window.

## Results

| Metric | ML model | Weekly seasonal baseline |
| --- | ---: | ---: |
| MAE | 73.98 | 78.66 |
| RMSE | 110.34 | 138.81 |
| RMSLE | 0.445 | 0.808 |

The nominal 90% interval has 85.4% empirical test coverage and a mean width of 270.14 rentals.

## Limitations

- Weather observations are treated as known; a live system would use weather forecasts and inherit their uncertainty.
- The symmetric interval has imperfect coverage under temporal distribution shift.
- Demand from 2011-2012 is not representative of current mobility patterns.
- Aggregate demand does not solve station-level rebalancing.
- Hyperparameters are fixed; a nested time-series validation study would be needed before deployment.

## Monitoring recommendations

Track rolling MAE, interval coverage, residual bias, weather-feature drift and demand-lag drift. Trigger retraining or recalibration when coverage or residual distributions deteriorate materially.
