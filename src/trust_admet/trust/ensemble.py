from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from trust_admet.data.featurize import dataframe_to_fingerprints


def available_classical_models(dataset, split, seed="42"):
    models = []

    for model_name in ["random_forest", "xgboost"]:
        path = Path("outputs/models") / dataset / split / model_name / f"seed{seed}.joblib"
        if path.exists():
            models.append(model_name)

    return models


def ensemble_predict_classical(smiles, dataset="BBBP", split="scaffold", seed="42"):
    model_names = available_classical_models(dataset, split, seed)

    if not model_names:
        raise FileNotFoundError("No classical ensemble models found.")

    query_df = pd.DataFrame({"canonical_smiles": [smiles]})
    x = dataframe_to_fingerprints(query_df)

    rows = []

    for model_name in model_names:
        path = Path("outputs/models") / dataset / split / model_name / f"seed{seed}.joblib"
        model = joblib.load(path)

        prob = float(model.predict_proba(x)[0, 1])
        rows.append({
            "model": model_name,
            "probability_positive": prob,
            "prediction": int(prob >= 0.5),
        })

    df = pd.DataFrame(rows)

    mean_prob = float(df["probability_positive"].mean())
    std_prob = float(df["probability_positive"].std(ddof=0))
    agreement = float((df["prediction"] == int(mean_prob >= 0.5)).mean())

    return {
        "models": rows,
        "ensemble_probability_positive": mean_prob,
        "ensemble_prediction": int(mean_prob >= 0.5),
        "ensemble_std": std_prob,
        "ensemble_agreement": agreement,
        "n_models": len(rows),
    }
