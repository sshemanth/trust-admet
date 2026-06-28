from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


CLASSIFICATION = {"BBBP", "ClinTox", "Tox21_NR-AR", "Tox21_SR-p53"}
REGRESSION = {"Solubility", "Lipophilicity"}


def parse_prediction_file(pred_path: Path):
    parts = pred_path.parts

    if pred_path.name == "test_predictions.csv":
        # outputs/models/DATASET/SPLIT/MODEL/seed42/test_predictions.csv
        dataset = parts[-5]
        split = parts[-4]
        model = parts[-3]
        seed = parts[-2].replace("seed", "")
        pred = pd.read_csv(pred_path)
        return dataset, split, model, seed, pred

    if pred_path.name.endswith("_predictions.csv"):
        # outputs/models/DATASET/SPLIT/MODEL/seed42_predictions.csv
        dataset = parts[-4]
        split = parts[-3]
        model = parts[-2]
        seed = pred_path.stem.replace("_predictions", "").replace("seed", "")
        pred = pd.read_csv(pred_path)
        pred = pred[pred["split"] == "test"].copy()
        return dataset, split, model, seed, pred

    return None


def main():
    ad_dir = Path("outputs/reports/applicability_domain")

    prediction_files = []
    prediction_files += list(Path("outputs/models").glob("*/*/*/seed*/test_predictions.csv"))
    prediction_files += list(Path("outputs/models").glob("*/*/*/seed*_predictions.csv"))

    rows = []

    for pred_path in sorted(prediction_files):
        parsed = parse_prediction_file(pred_path)
        if parsed is None:
            continue

        dataset, split, model, seed, pred = parsed

        ad_path = ad_dir / f"{dataset}_{split}_test_ad.csv"
        if not ad_path.exists():
            continue

        ad = pd.read_csv(ad_path)[["canonical_smiles", "ad_region", "ad_max_tanimoto"]]
        merged = pred.merge(ad, on="canonical_smiles", how="left")

        for region in ["inside", "outside"]:
            sub = merged[merged["ad_region"] == region].copy()
            if len(sub) < 5:
                continue

            row = {
                "dataset": dataset,
                "split": split,
                "model": model,
                "seed": seed,
                "ad_region": region,
                "n": len(sub),
                "mean_ad_max_tanimoto": sub["ad_max_tanimoto"].mean(),
            }

            if dataset in CLASSIFICATION and "y_prob" in sub.columns:
                if sub["Y"].nunique() > 1:
                    row["auroc"] = roc_auc_score(sub["Y"], sub["y_prob"])
                    row["auprc"] = average_precision_score(sub["Y"], sub["y_prob"])
                else:
                    row["auroc"] = None
                    row["auprc"] = None

            if dataset in REGRESSION:
                pred_col = "y_pred" if "y_pred" in sub.columns else None
                if pred_col is not None:
                    mse = mean_squared_error(sub["Y"], sub[pred_col])
                    row["rmse"] = float(np.sqrt(mse))
                    row["mae"] = mean_absolute_error(sub["Y"], sub[pred_col])

            rows.append(row)

    out_dir = Path("outputs/reports/applicability_domain")
    out_dir.mkdir(parents=True, exist_ok=True)

    out = pd.DataFrame(rows)
    out_path = out_dir / "ad_vs_performance.csv"
    out.to_csv(out_path, index=False)

    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
