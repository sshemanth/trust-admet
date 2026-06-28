<div align="center">

# 🧬 TRUST-ADMET

### **TRUSTworthy Artificial Intelligence Framework for ADMET Prediction**

*Towards Reliable, Explainable and Deployable Molecular Property Prediction*

<p>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python\&logoColor=white)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch\&logoColor=white)]()
[![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-GNN-orange)]()
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)]()
[![RDKit](https://img.shields.io/badge/RDKit-Cheminformatics-green)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi)]()
[![Gradio](https://img.shields.io/badge/Gradio-Web%20Demo-FF6F00)]()
[![Docker](https://img.shields.io/badge/Docker-Deployment-2496ED?logo=docker\&logoColor=white)]()
[![ClearML](https://img.shields.io/badge/ClearML-Experiment%20Tracking-blue)]()
[![MIT License](https://img.shields.io/badge/License-MIT-success.svg)]()

</p>

---

### 🧪 Benchmark • 📊 Calibration • 🌐 Applicability Domain • 📉 Uncertainty • 🔬 Explainability • 🛡 Trustworthy AI

*A unified framework for trustworthy ADMET prediction using Classical Machine Learning, Graph Neural Networks, and Molecular Foundation Models.*

</div>

---

# 🚀 Highlights

TRUST-ADMET goes beyond predictive performance by answering an equally important question:

> **Can this prediction be trusted?**

Unlike conventional ADMET benchmark repositories that primarily report AUROC or RMSE, TRUST-ADMET integrates multiple complementary trustworthiness components into a single reproducible framework.

### ✨ Features

* 🧠 Multiple AI paradigms

  * Random Forest
  * XGBoost
  * Multilayer Perceptron (MLP)
  * Graph Convolutional Networks (GCN)
  * Graph Isomorphism Networks (GIN)
  * ChemBERTa

* 🧪 Multiple benchmark datasets

* 📊 Calibration-aware prediction

* 🌐 Applicability Domain analysis

* 📉 Predictive uncertainty estimation

* 🔬 Explainability

* 🧪 External validation

* ⚡ FastAPI deployment

* 🎨 Interactive Gradio interface

* 🐳 Docker support

* 📈 Automated benchmark reports

* 📄 Paper-ready figures and tables

---

# 📖 Overview

TRUST-ADMET is an end-to-end framework for molecular property prediction with an emphasis on **trustworthy artificial intelligence**.

The framework provides reproducible benchmarking across multiple model families while evaluating not only predictive accuracy but also calibration, applicability domain, uncertainty estimation, explainability, and deployment readiness.

The project combines:

* Classical Machine Learning
* Deep Learning
* Graph Neural Networks
* Molecular Foundation Models
* Trustworthy AI methodologies
* Reproducible experimentation
* Interactive deployment

into one unified research framework.

---

# 🎯 Motivation

High predictive accuracy alone is insufficient for molecular decision making.

Two molecules with identical predicted probabilities may differ dramatically in reliability due to:

* being outside the training chemical space,
* poor probability calibration,
* disagreement across different models,
* high predictive uncertainty,
* or limited interpretability.

TRUST-ADMET addresses these challenges by integrating trust-aware decision support directly into the prediction pipeline.

---

# 🛡 TRUST Framework

Every prediction is evaluated through complementary trust dimensions.

| Component                 | Purpose                                     |
| ------------------------- | ------------------------------------------- |
| 🎯 Predictive Performance | AUROC, AUPRC, RMSE, MAE                     |
| 📊 Calibration            | ECE, Brier Score, Negative Log-Likelihood   |
| 🌐 Applicability Domain   | Similarity + Physicochemical Space          |
| 📉 Uncertainty            | Monte Carlo Dropout + Ensemble Disagreement |
| 🔬 Explainability         | SHAP, Integrated Gradients, GNNExplainer    |
| 🧪 External Validation    | Independent benchmark datasets              |
| 🤖 Deployment             | FastAPI + Gradio + Docker                   |

---

# 🏗 Framework Overview

```text
                    Raw Molecular Datasets
                              │
                              ▼
                     Dataset Validation
                              │
                              ▼
                  Canonical SMILES Generation
                              │
                              ▼
                 Random / Scaffold Splitting
                              │
                              ▼
                     Molecular Representation
           ┌──────────────────┼──────────────────┐
           │                  │                  │
           ▼                  ▼                  ▼
 Morgan Fingerprints     Molecular Graphs    SMILES Tokens
           │                  │                  │
           └──────────────┬───┴──────────────────┘
                          ▼
                   Model Training Pipeline
      ┌────────────┬────────────┬────────────┬────────────┐
      │            │            │            │            │
      ▼            ▼            ▼            ▼            ▼
 RandomForest   XGBoost        MLP          GNN       ChemBERTa
                                           (GCN/GIN)
                          │
                          ▼
                Trust Evaluation Framework
      ┌────────────┬────────────┬────────────┬────────────┐
      │            │            │            │
      ▼            ▼            ▼            ▼
 Calibration   Applicability  Uncertainty  Explainability
                          │
                          ▼
               TRUST Decision & Reporting
                          │
                          ▼
        CLI • REST API • Gradio • Paper Figures
```

---

# 📊 Supported Models

| Category                       | Models                 |
| ------------------------------ | ---------------------- |
| 🌲 Classical Machine Learning  | Random Forest, XGBoost |
| 🧠 Neural Networks             | MLP                    |
| 🔗 Graph Neural Networks       | GCN, GIN               |
| 🤖 Molecular Foundation Models | ChemBERTa              |

---

# 🧪 Supported Datasets

## Classification

* BBBP
* ClinTox
* Tox21 NR-AR
* Tox21 SR-p53

## Regression

* Lipophilicity
* AqSolDB Solubility

---

# 📂 Repository Structure

```text
trust-admet/

├── api/                    # FastAPI service
├── app/                    # Gradio application
├── configs/                # Experiment configurations
├── data/                   # Datasets
├── docs/
├── outputs/
├── paper/
│   ├── figures/
│   └── tables/
├── scripts/
├── src/
│   └── trust_admet/
├── tests/

├── Dockerfile
├── docker-compose.yml
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
└── pyproject.toml
```
# 🚀 Installation

## Clone the Repository

```bash
git clone https://github.com/sshemanth/trust-admet.git
cd trust-admet
```

---

## Create Environment

Using Conda:

```bash
conda env create -f environment.yml
conda activate trust-admet
```

Or using pip:

```bash
python -m venv .venv

source .venv/bin/activate      # Linux/macOS

# Windows
# .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

---

## Verify Installation

```bash
python -c "import trust_admet; print('TRUST-ADMET installed successfully!')"
```

---

# ⚡ Quick Start

## 1. Download datasets

```bash
python scripts/download_data.py
```

---

## 2. Validate datasets

```bash
python scripts/validate_datasets.py
```

---

## 3. Train a Classical Machine Learning model

Random Forest

```bash
python scripts/train_classical_baseline.py \
    --dataset BBBP \
    --split scaffold \
    --model random_forest
```

XGBoost

```bash
python scripts/train_classical_baseline.py \
    --dataset BBBP \
    --split scaffold \
    --model xgboost
```

---

## 4. Train Neural Network

```bash
python scripts/train_mlp.py \
    --dataset BBBP \
    --split scaffold
```

---

## 5. Train Graph Neural Networks

GCN

```bash
python scripts/run_gcn_experiment.py \
    --model gcn \
    --dataset BBBP
```

GIN

```bash
python scripts/run_gcn_experiment.py \
    --model gin \
    --dataset BBBP
```

---

## 6. Fine-tune ChemBERTa

```bash
python scripts/run_chemberta_experiment.py \
    --dataset BBBP \
    --split scaffold
```

---

# 📊 Benchmark Generation

Generate benchmark tables

```bash
python scripts/generate_leaderboard.py
```

Generate calibration analysis

```bash
python scripts/analyze_calibration.py
```

Generate applicability-domain analysis

```bash
python scripts/analyze_applicability_domain.py
```

Generate uncertainty analysis

```bash
python scripts/summarize_uncertainty.py
```

Generate explainability reports

```bash
python scripts/run_shap_classical.py
python scripts/run_integrated_gradients.py
python scripts/run_gnnexplainer_gin.py
```

Generate publication assets

```bash
python scripts/generate_paper_assets.py
```

---

# 🧪 Single Molecule Prediction

Predict a single molecule from the command line.

```bash
python scripts/predict_single.py \
    --smiles "CCO" \
    --dataset BBBP \
    --split scaffold \
    --model random_forest
```

Example output

```text
Prediction : BBB Permeable

Probability : 0.97

Conformal Prediction Set:
{BBB Permeable}

Applicability Domain:
Inside

Recommendation:
Accept prediction.
```

---

# 🎨 Interactive Gradio Application

Launch

```bash
python app/gradio_app.py
```

Open

```
http://localhost:7860
```

Features

* Interactive molecular prediction
* TRUST profile
* Applicability-domain analysis
* Conformal prediction
* Ensemble agreement
* JSON export

---

# ⚡ FastAPI REST API

Start server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Interactive documentation

```
http://localhost:8000/docs
```

Health endpoint

```bash
curl http://localhost:8000/health
```

Prediction endpoint

```bash
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{
    "smiles":"CCO",
    "dataset":"BBBP",
    "split":"scaffold",
    "model":"random_forest"
}'
```

---

# 🐳 Docker Deployment

Build

```bash
docker compose build
```

Launch

```bash
docker compose up
```

API

```
http://localhost:8000
```

Gradio

```
http://localhost:7860
```

---

# 📈 Experiment Tracking

Experiments are automatically tracked using **ClearML**.

Tracked information includes

* Hyperparameters
* Training metrics
* Validation metrics
* Model checkpoints
* System information
* Figures
* Tables

This enables complete experiment reproducibility.

---

# 📄 Reproducibility

TRUST-ADMET emphasizes reproducible research.

Every experiment includes

* Configuration-driven execution
* Fixed random seeds
* Dataset validation
* Canonical preprocessing
* Version-controlled code
* Automated experiment logging
* Generated figures and tables
* Benchmark reports

All reported results can be regenerated using the provided scripts.


<!-- BENCHMARK_RESULTS_START -->
# 📊 Benchmark Results

TRUST-ADMET evaluates multiple model families across ADMET classification and regression tasks. The tables below report **scaffold split test performance** as **mean ± standard deviation** across available seeds.

## Classification Tasks

| Dataset | Model | Seeds | AUROC | AUPRC | MCC | ECE |
|---|---:|---:|---:|---:|---:|---:|
| BBBP | Random Forest | 5 | 0.874 ± 0.005 | 0.951 ± 0.002 | 0.547 ± 0.009 | 0.061 ± 0.016 |
| BBBP | MLP | 5 | 0.866 ± 0.000 | 0.945 ± 0.000 | 0.516 ± 0.000 | 0.132 ± 0.000 |
| BBBP | XGBoost | 5 | 0.866 ± 0.004 | 0.950 ± 0.002 | 0.565 ± 0.034 | 0.087 ± 0.007 |
| BBBP | ChemBERTa | 1 | 0.826 ± 0.000 | 0.920 ± 0.000 | 0.557 ± 0.000 | 0.069 ± 0.000 |
| BBBP | GIN | 5 | 0.823 ± 0.069 | 0.918 ± 0.037 | 0.473 ± 0.077 | 0.112 ± 0.037 |
| BBBP | GCN | 1 | 0.710 ± 0.000 | 0.859 ± 0.000 | 0.247 ± 0.000 | 0.046 ± 0.000 |
| ClinTox | GIN | 5 | 0.844 ± 0.019 | 0.390 ± 0.056 | 0.321 ± 0.116 | 0.057 ± 0.012 |
| ClinTox | Random Forest | 5 | 0.772 ± 0.011 | 0.495 ± 0.015 | 0.102 ± 0.140 | 0.037 ± 0.002 |
| ClinTox | XGBoost | 5 | 0.764 ± 0.010 | 0.479 ± 0.007 | 0.500 ± 0.026 | 0.070 ± 0.004 |
| ClinTox | MLP | 5 | 0.680 ± 0.000 | 0.330 ± 0.000 | 0.254 ± 0.000 | 0.101 ± 0.000 |
| Tox21 NR-AR | Random Forest | 5 | 0.769 ± 0.013 | 0.461 ± 0.004 | 0.533 ± 0.012 | 0.025 ± 0.001 |
| Tox21 NR-AR | GIN | 5 | 0.764 ± 0.013 | 0.384 ± 0.053 | 0.430 ± 0.039 | 0.019 ± 0.004 |
| Tox21 NR-AR | XGBoost | 5 | 0.756 ± 0.006 | 0.427 ± 0.007 | 0.503 ± 0.008 | 0.018 ± 0.002 |
| Tox21 NR-AR | MLP | 5 | 0.667 ± 0.000 | 0.412 ± 0.000 | 0.000 ± 0.000 | 0.023 ± 0.000 |
| Tox21 SR-p53 | XGBoost | 5 | 0.753 ± 0.005 | 0.337 ± 0.006 | 0.194 ± 0.009 | 0.067 ± 0.002 |
| Tox21 SR-p53 | Random Forest | 5 | 0.745 ± 0.005 | 0.378 ± 0.007 | 0.241 ± 0.000 | 0.032 ± 0.002 |
| Tox21 SR-p53 | GIN | 5 | 0.702 ± 0.011 | 0.296 ± 0.023 | 0.117 ± 0.107 | 0.046 ± 0.007 |
| Tox21 SR-p53 | MLP | 5 | 0.610 ± 0.000 | 0.272 ± 0.000 | 0.223 ± 0.000 | 0.108 ± 0.000 |

## Regression Tasks

| Dataset | Model | Seeds | RMSE | MAE | R² |
|---|---:|---:|---:|---:|---:|
| Lipophilicity | GIN | 5 | 0.748 ± 0.012 | 0.582 ± 0.013 | 0.604 ± 0.013 |
| Lipophilicity | XGBoost | 5 | 0.890 ± 0.007 | 0.687 ± 0.006 | 0.440 ± 0.008 |
| Lipophilicity | MLP | 5 | 0.903 ± 0.000 | 0.696 ± 0.000 | 0.423 ± 0.000 |
| Lipophilicity | ChemBERTa | 1 | 0.912 ± 0.000 | 0.723 ± 0.000 | 0.411 ± 0.000 |
| Lipophilicity | Random Forest | 5 | 0.966 ± 0.003 | 0.739 ± 0.002 | 0.341 ± 0.003 |
| Solubility | GIN | 5 | 1.843 ± 0.085 | 1.385 ± 0.117 | 0.347 ± 0.061 |
| Solubility | XGBoost | 5 | 1.893 ± 0.010 | 1.473 ± 0.005 | 0.312 ± 0.007 |
| Solubility | MLP | 5 | 1.950 ± 0.000 | 1.495 ± 0.000 | 0.270 ± 0.000 |
| Solubility | Random Forest | 5 | 2.024 ± 0.003 | 1.554 ± 0.004 | 0.213 ± 0.003 |

## Key Findings

- Classical models remain highly competitive on BBBP scaffold evaluation.
- GIN provides strong regression performance on Lipophilicity and Solubility.
- Model performance varies across tasks, supporting model-family comparison rather than assuming one architecture is universally superior.
- Calibration, applicability-domain analysis, conformal prediction, ensemble agreement, uncertainty, and explainability are essential for trust-aware ADMET prediction.
<!-- BENCHMARK_RESULTS_END -->

<!-- EXTERNAL_VALIDATION_START -->
# 🌍 External Validation

TRUST-ADMET evaluates BBBP-trained models on the independent B3DB dataset after removing molecules overlapping with the BBBP training set.

| External Dataset | Train Dataset | Model | N | Removed Overlap | AUROC | AUPRC | MCC | ECE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| B3DB | BBBP | Random Forest | 6482 | 1325 | 0.912 | 0.941 | 0.580 | 0.087 |
| B3DB | BBBP | XGBoost | 6482 | 1325 | 0.888 | 0.914 | 0.606 | 0.096 |

## External Validation Summary

- B3DB molecules before overlap removal: 7,807
- B3DB molecules after overlap removal: 6,482
- Removed train-overlap molecules: 1,325
- Random Forest achieved strong external generalization with AUROC above 0.91.
- XGBoost also retained strong external performance with AUROC above 0.88.

<!-- EXTERNAL_VALIDATION_END -->
