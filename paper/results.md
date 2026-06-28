# Results

## Dataset Quality Control

All datasets were processed through the same validation pipeline. Dataset statistics are provided in Table 1.

## Benchmark Performance

Classical baselines were highly competitive with neural, graph, and foundation models under scaffold splitting. Benchmark results are summarized in Tables 2 and 3.

## Applicability Domain

Scaffold splitting increased the proportion of molecules outside the applicability domain. Models generally performed better inside the applicability domain than outside it.

## Uncertainty

Monte Carlo dropout uncertainty separated reliable and unreliable predictions for most classification tasks. High-uncertainty predictions generally showed higher error rates.

## Explainability

Random Forest SHAP and XGBoost feature-importance analyses provide interpretable molecular fingerprint features associated with model predictions.

## TRUST Profiles

TRUST profiles combine prediction confidence, applicability-domain status, uncertainty, and correctness into a prediction-level decision-support artifact.
