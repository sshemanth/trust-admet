from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import pairwise_distances

from trust_admet.data.featurize import dataframe_to_fingerprints
from trust_admet.trust.trust_score import TrustScore


def load_model(dataset, split, model, seed):
    model_path = Path("outputs/models") / dataset / split / model / f"seed{seed}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)


def get_nearest_similarity(smiles, dataset, split):
    train_path = Path("data/splits") / dataset / split / "train.csv"
    train_df = pd.read_csv(train_path)

    query_df = pd.DataFrame({"canonical_smiles": [smiles]})

    x_query = dataframe_to_fingerprints(query_df)
    x_train = dataframe_to_fingerprints(train_df)

    distances = pairwise_distances(x_query, x_train, metric="jaccard")
    similarity = 1.0 - distances.min()

    return float(similarity)


def get_model_ece(dataset, split, model, seed):
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


def probability_uncertainty(prob):
    return float(1.0 - abs(prob - 0.5) * 2.0)


def predict_with_trust(smiles, dataset="BBBP", split="scaffold", model_name="random_forest", seed="42"):
    model = load_model(dataset, split, model_name, seed)

    query_df = pd.DataFrame({"canonical_smiles": [smiles]})
    x = dataframe_to_fingerprints(query_df)

    if not hasattr(model, "predict_proba"):
        raise ValueError("Automatic TRUST Score v1 currently supports classification models with predict_proba.")

    prob = float(model.predict_proba(x)[0, 1])
    pred = int(prob >= 0.5)

    similarity = get_nearest_similarity(smiles, dataset, split)
    uncertainty = probability_uncertainty(prob)
    ece = get_model_ece(dataset, split, model_name, seed)

    score = TrustScore(
        probability=max(prob, 1.0 - prob),
        similarity=similarity,
        uncertainty=uncertainty,
        ece=ece,
    )

    return {
        "smiles": smiles,
        "dataset": dataset,
        "split": split,
        "model": model_name,
        "seed": seed,
        "prediction": pred,
        "label": "BBB Permeable" if pred == 1 else "BBB Non-permeable",
        "probability_positive": prob,
        "prediction_confidence": max(prob, 1.0 - prob),
        "nearest_similarity": similarity,
        "applicability_domain": "Inside" if similarity >= 0.5 else "Outside",
        "uncertainty": uncertainty,
        "ece": ece,
        "trust_score": score.total,
        "trust_level": score.level,
        "recommendation": score.recommendation,
        "score_breakdown": {
            "prediction_confidence": score.confidence_component(),
            "calibration": score.calibration_component(),
            "applicability_domain": score.applicability_component(),
            "uncertainty": score.uncertainty_component(),
        },
    }
