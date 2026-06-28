
## Trust Score Design Note

Classification and regression tasks require separate trust formulations.

Classification trust components:
- AUROC / AUPRC
- MCC / F1
- Brier score
- negative log-likelihood
- expected calibration error
- uncertainty estimates
- applicability domain

Regression trust components:
- RMSE / MAE
- R2
- prediction interval coverage
- uncertainty estimates
- residual calibration
- applicability domain

NaN values from non-applicable metrics should be treated as not applicable, not failed experiments.
