# TRUST-ADMET Architecture

## Pipeline

Raw datasets
      │
      ▼
Validation
      │
      ▼
Canonicalization
      │
      ▼
Random / Scaffold Split
      │
      ▼
Feature Generation
      │
      ├── Morgan Fingerprints
      ├── Molecular Graphs
      └── SMILES Tokens
      │
      ▼
Model Training
      │
      ├── Random Forest
      ├── XGBoost
      ├── MLP
      ├── GCN
      ├── GIN
      └── ChemBERTa
      │
      ▼
Evaluation
      │
      ├── Performance
      ├── Calibration
      ├── Applicability Domain
      ├── Uncertainty
      └── Explainability
