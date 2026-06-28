# Architecture

TRUST-ADMET is organized as a modular research framework.

## Main Modules

- data: download, validation, featurization, splitting
- models: model definitions
- training: reusable training loops
- evaluation: metric computation
- reporting: tables and figures
- trust: calibration, uncertainty, applicability domain, explainability
- registry: model and experiment registries

## Design Principle

A model should be evaluated through the same standardized pipeline regardless of whether it is a classical ML model, neural baseline, graph model, or molecular foundation model.
