from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from trust_admet.data.featurize import dataframe_to_fingerprints


def build_conformal_threshold(dataset, split, model_name, seed="42", alpha=0.05):
    model_path = Path("outputs/models") / dataset / split / model_name / f"seed{seed}.joblib"
    valid_path = Path("data/splits") / dataset / split / "valid.csv"

    model = joblib.load(model_path)
    valid_df = pd.read_csv(valid_path)

    x_valid = dataframe_to_fingerprints(valid_df)
    y_valid = valid_df["Y"].astype(int).values

    probs = model.predict_proba(x_valid)

    true_class_probs = probs[np.arange(len(y_valid)), y_valid]
    nonconformity = 1.0 - true_class_probs

    qhat = np.quantile(
        nonconformity,
        np.ceil((len(nonconformity) + 1) * (1 - alpha)) / len(nonconformity),
        method="higher",
    )

    out_dir = Path("outputs/reports/conformal")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{dataset}_{split}_{model_name}_seed{seed}_alpha{alpha}_threshold.csv"
    pd.DataFrame([{
        "dataset": dataset,
        "split": split,
        "model": model_name,
        "seed": seed,
        "alpha": alpha,
        "qhat": qhat,
        "n_calibration": len(valid_df),
    }]).to_csv(out_path, index=False)

    return float(qhat)


def load_or_build_conformal_threshold(dataset, split, model_name, seed="42", alpha=0.05):
    path = Path("outputs/reports/conformal") / f"{dataset}_{split}_{model_name}_seed{seed}_alpha{alpha}_threshold.csv"

    if path.exists():
        return float(pd.read_csv(path)["qhat"].iloc[0])

    return build_conformal_threshold(dataset, split, model_name, seed, alpha)


def conformal_prediction_set(probability_positive, qhat):
    probs = {
        "BBB Non-permeable": 1.0 - probability_positive,
        "BBB Permeable": probability_positive,
    }

    prediction_set = [
        label for label, prob in probs.items()
        if 1.0 - prob <= qhat
    ]

    return prediction_set
