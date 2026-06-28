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
