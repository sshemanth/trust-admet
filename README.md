<div align="center">

# 🧬 TRUST-ADMET

### **TRUSTworthy Artificial Intelligence Framework for ADMET Prediction**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)]()
[![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-GNN-orange)]()
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)]()
[![RDKit](https://img.shields.io/badge/RDKit-Cheminformatics-green)]()
[![ClearML](https://img.shields.io/badge/ClearML-Experiment%20Tracking-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)]()

---

### 🧪 Benchmark • 📊 Calibration • 🎯 Applicability Domain • 📉 Uncertainty • 🔬 Explainability

*A unified framework for **trustworthy ADMET prediction** using Classical ML, Graph Neural Networks, and Molecular Foundation Models.*

</div>

---

# 📖 Overview

TRUST-ADMET is an **end-to-end framework** for trustworthy molecular property prediction.

Unlike conventional ADMET benchmarks that only report predictive accuracy, TRUST-ADMET evaluates **whether a prediction should be trusted** by integrating:

- 🎯 Predictive Performance
- 📊 Calibration
- 🌐 Applicability Domain
- 📉 Predictive Uncertainty
- 🔬 Explainability

---

# ✨ Features

## 🧬 Supported Models

| Category | Models |
|-----------|---------|
| 🌲 Classical ML | Random Forest, XGBoost |
| 🧠 Deep Learning | MLP |
| 🔗 Graph Neural Networks | GCN, GIN |
| 🤖 Foundation Models | ChemBERTa |

---

## 📊 Supported Datasets

### Classification

- BBBP
- ClinTox
- Tox21 NR-AR
- Tox21 SR-p53

### Regression

- Lipophilicity
- AqSolDB Solubility

---

# 🛡 Trust Framework

TRUST-ADMET evaluates every prediction using five complementary dimensions.

| Component | Description |
|------------|-------------|
| 🎯 Performance | AUROC / RMSE |
| 📊 Calibration | Brier Score, ECE, NLL |
| 🌐 Applicability Domain | Tanimoto Similarity |
| 📉 Uncertainty | Monte Carlo Dropout |
| 🔬 Explainability | SHAP, Integrated Gradients, GNNExplainer |

---

# 🏗 Project Architecture

```
                Raw Datasets
                     │
                     ▼
            Data Validation
                     │
                     ▼
          Canonicalization
                     │
                     ▼
      Random / Scaffold Split
                     │
                     ▼
          Feature Generation
      ├── Morgan Fingerprints
      ├── Molecular Graphs
      └── SMILES Tokens
                     │
                     ▼
             Model Training
     ├── Random Forest
     ├── XGBoost
     ├── MLP
     ├── GCN
     ├── GIN
     └── ChemBERTa
                     │
                     ▼
         Trust Evaluation
     ├── Calibration
     ├── Applicability Domain
     ├── Uncertainty
     └── Explainability
                     │
                     ▼
      TRUST Profile + Reports
```

---

# 📂 Repository Structure

```text
trust-admet/

├── configs/
├── data/
├── docs/
├── notebooks/
├── outputs/
├── paper/
├── scripts/
├── src/
├── tests/
│
├── README.md
├── LICENSE
├── CITATION.cff
└── environment.yml
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/trust-admet.git

cd trust-admet
```

Create environment

```bash
conda env create -f environment.yml

conda activate trust-admet
```

Install

```bash
pip install -e .
```

---

# ⚡ Quick Start

Download datasets

```bash
python scripts/download_data.py
```

Validate

```bash
python scripts/validate_datasets.py
```

Train Random Forest

```bash
python scripts/train_classical_baseline.py \
    --dataset BBBP \
    --split scaffold \
    --model random_forest
```

Run GIN

```bash
python scripts/run_gcn_experiment.py \
    --model gin \
    --dataset BBBP
```

Generate Leaderboard

```bash
python scripts/generate_leaderboard.py
```

Generate Paper Assets

```bash
python scripts/generate_paper_assets.py
```

---

# 📈 Results

Example benchmark metrics

| Dataset | Model | AUROC |
|----------|--------|-------:|
| BBBP | Random Forest | **0.88** |
| BBBP | XGBoost | **0.87** |
| BBBP | GIN | **0.86** |
| BBBP | ChemBERTa | **0.83** |

(Multi-seed benchmark results are available in the `paper/tables/` directory.)

---

# 🔬 Explainability

TRUST-ADMET supports multiple explanation techniques.

| Model | Method |
|---------|----------|
| Random Forest | SHAP |
| XGBoost | Feature Importance |
| MLP | Integrated Gradients |
| GIN | GNNExplainer |
| ChemBERTa | Token Integrated Gradients |

---

# 📊 Generated Paper Assets

The framework automatically generates:

✅ Benchmark Tables

✅ Calibration Figures

✅ Applicability Domain Analysis

✅ Uncertainty Analysis

✅ Explainability Figures

✅ TRUST Profiles

---

# 📜 Reproducibility

All experiments are

- ✅ Version controlled
- ✅ Multi-seed
- ✅ Automatically logged with ClearML
- ✅ Configuration driven
- ✅ Reproducible

---

# 📚 Citation

If you use TRUST-ADMET in your research please cite

```bibtex
@software{trustadmet2026,
  title={TRUST-ADMET},
  author={Sri Sai Hemanth Bollepalli},
  year={2026},
  url={https://github.com/sshemanth/trust-admet}
}
```

---

# 🤝 Contributing

Pull requests are welcome.

For major changes please open an issue first to discuss what you would like to change.

---

# 📄 License

MIT License

---

<div align="center">

## ⭐ If you find TRUST-ADMET useful, please consider giving the repository a star ⭐

Made with ❤️ for Trustworthy AI in Drug Discovery

</div>

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
