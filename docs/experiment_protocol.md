# Experiment Protocol

## Splitting
- Scaffold split is the primary evaluation split.
- Random split is reported as a secondary comparison.
- Splits are fixed and saved under data/splits/.

## Seeds
- Development runs use seed 42.
- Final experiments should use at least three seeds: 42, 123, 2026.

## Metrics

### Classification
- AUROC
- AUPRC
- Accuracy
- F1
- MCC
- Brier score
- NLL
- Expected Calibration Error

### Regression
- RMSE
- MAE
- R2
- Later: prediction interval coverage and residual calibration

## Reporting
- Report mean ± standard deviation for final experiments.
- Log all experiments to ClearML.
- Save metrics, predictions, and model checkpoints.
- Generate tables and figures automatically from saved outputs.

## Trust Framework
Classification and regression tasks require separate trust formulations.
Non-applicable metrics should be treated as not applicable rather than failed.
