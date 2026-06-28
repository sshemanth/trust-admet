from pathlib import Path

import joblib
import pandas as pd
from rdkit import Chem
from sklearn.metrics import pairwise_distances

from trust_admet.data.featurize import dataframe_to_fingerprints
from trust_admet.trust.trust_score import TrustScore
from trust_admet.trust.physchem_ad import check_physchem_ad
from trust_admet.trust.conformal import load_or_build_conformal_threshold, conformal_prediction_set
from trust_admet.trust.ensemble import ensemble_predict_classical


def validate_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True)


def load_model(dataset: str, split: str, model: str, seed: str):
    model_path = Path("outputs/models") / dataset / split / model / f"seed{seed}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)


def get_nearest_similarity(canonical_smiles: str, dataset: str, split: str) -> float:
    train_path = Path("data/splits") / dataset / split / "train.csv"
    train_df = pd.read_csv(train_path)

    query_df = pd.DataFrame({"canonical_smiles": [canonical_smiles]})

    x_query = dataframe_to_fingerprints(query_df)
    x_train = dataframe_to_fingerprints(train_df)

    distances = pairwise_distances(x_query, x_train, metric="jaccard")
    similarity = 1.0 - distances.min()

    return float(similarity)


def get_model_ece(dataset: str, split: str, model: str, seed: str) -> float:
    candidates = [
        Path("outputs/models") / dataset / split / model / f"seed{seed}_metrics.csv",
        Path("outputs/models") / dataset / split / model / f"seed{seed}" / "metrics.csv",
    ]

    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            for col in ["test_ece", "valid_ece"]:
                if col in df.columns and pd.notna(df[col].iloc[0]):
                    return float(df[col].iloc[0])

    return 0.10


def probability_uncertainty(probability_positive: float) -> float:
    return float(1.0 - abs(probability_positive - 0.5) * 2.0)


def predict_with_trust(
    smiles: str,
    dataset: str = "BBBP",
    split: str = "scaffold",
    model_name: str = "random_forest",
    seed: str = "42",
):
    canonical_smiles = validate_smiles(smiles)

    physchem_ad = check_physchem_ad(canonical_smiles, dataset, split)

    n_physchem_violations = len(physchem_ad.get("violations", []))

    if n_physchem_violations >= 2:
        return {
            "smiles": canonical_smiles,
            "dataset": dataset,
            "split": split,
            "model": model_name,
            "seed": str(seed),
            "prediction": None,
            "label": "Prediction refused",
            "probability_positive": None,
            "prediction_confidence": None,
            "nearest_similarity": None,
            "applicability_domain": "Outside physicochemical AD",
            "uncertainty": None,
            "ece": None,
            "trust_score": 0.0,
            "trust_level": "REFUSED",
            "recommendation": "Prediction refused because the molecule violates multiple physicochemical applicability-domain limits.",
            "physchem_ad": physchem_ad,
            "score_breakdown": {
                "prediction_confidence": 0.0,
                "calibration": 0.0,
                "applicability_domain": 0.0,
                "uncertainty": 0.0,
            },
        }

    model = load_model(dataset, split, model_name, seed)

    query_df = pd.DataFrame({"canonical_smiles": [canonical_smiles]})
    x = dataframe_to_fingerprints(query_df)

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "Automatic TRUST Score v1 currently supports classification models with predict_proba."
        )

    prob = float(model.predict_proba(x)[0, 1])
    pred = int(prob >= 0.5)

    prediction_confidence = max(prob, 1.0 - prob)
    similarity = get_nearest_similarity(canonical_smiles, dataset, split)
    uncertainty = probability_uncertainty(prob)
    ece = get_model_ece(dataset, split, model_name, seed)

    qhat = load_or_build_conformal_threshold(dataset, split, model_name, seed, alpha=0.05)
    prediction_set = conformal_prediction_set(prob, qhat)

    ensemble = ensemble_predict_classical(canonical_smiles, dataset, split, seed)

    score = TrustScore(
        probability=prediction_confidence,
        similarity=similarity,
        uncertainty=uncertainty,
        ece=ece,
    )

    return {
        "smiles": canonical_smiles,
        "dataset": dataset,
        "split": split,
        "model": model_name,
        "seed": str(seed),
        "prediction": pred,
        "label": "BBB Permeable" if pred == 1 else "BBB Non-permeable",
        "probability_positive": prob,
        "prediction_confidence": prediction_confidence,
        "nearest_similarity": similarity,
        "applicability_domain": ("Physchem warning" if n_physchem_violations == 1 else ("Inside" if similarity >= 0.5 else "Outside")),
        "uncertainty": uncertainty,
        "ece": ece,
        "conformal_alpha": 0.05,
        "conformal_qhat": qhat,
        "conformal_prediction_set": prediction_set,
        "ensemble": ensemble,
        "trust_score": score.total,
        "trust_level": score.level,
        "recommendation": ("Prediction allowed with physicochemical AD warning." if n_physchem_violations == 1 else ("Accept prediction." if len(prediction_set) == 1 else "Review manually: conformal prediction set is uncertain.")),
        "physchem_ad": physchem_ad,
        "score_breakdown": {
            "prediction_confidence": score.confidence_component(),
            "calibration": score.calibration_component(),
            "applicability_domain": score.applicability_component(),
            "uncertainty": score.uncertainty_component(),
        },
    }
