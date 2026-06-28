# Methods

## Overview

TRUST-ADMET is a reproducible framework for evaluating ADMET prediction models using predictive performance, calibration, applicability-domain analysis, uncertainty estimation, and explainability.

## Datasets

We evaluated classification datasets including BBBP, ClinTox, Tox21 NR-AR, and Tox21 SR-p53, and regression datasets including Lipophilicity and AqSolDB Solubility.

## Data Processing

Molecules were standardized using RDKit, invalid SMILES were removed, and duplicate canonical SMILES were identified. Both random and scaffold splits were generated, with scaffold split used as the primary evaluation setting.

## Models

We benchmarked Random Forest, XGBoost, MLP, GCN, GIN, and ChemBERTa.

## Evaluation

Classification models were evaluated using AUROC, AUPRC, F1, MCC, Brier score, NLL, and ECE. Regression models were evaluated using RMSE, MAE, and R2.

## Trustworthiness Analysis

TRUST-ADMET evaluates prediction reliability using calibration, applicability domain, uncertainty estimation, and explainability.
