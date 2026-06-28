from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error, mean_absolute_error
import numpy as np


CLASSIFICATION = {"BBBP", "ClinTox", "Tox21_NR-AR", "Tox21_SR-p53"}
REGRESSION = {"Solubility", "Lipophilicity"}


def main():
    ad_dir = Path("outputs/reports/applicability_domain")
    pred_files = list(Path("outputs/models").glob("*/*/*/seed*/test_predictions.csv"))
    pred_files += list(Path("outputs/models").glob("*/*/*/seed*_predictions.csv"))

    rows = []

    for pred_path in pred_files:
        parts = pred_path.parts

        if pred_path.name == "test_predictions.csv":
            dataset = parts[-5]
            split = parts[-4]
            model = parts[-3]
            seed = parts[-2].replace("seed", "")
        else:
            continue

        ad_path = ad_dir / f"{dataset}_{split}_test_ad.csv"
        if not ad_path.exists():
            continue

        pred = pd.read_csv(pred_path)
        ad = pd.read_csv(ad_path)[["canonical_smiles", "ad_region", "ad_max_tanimoto"]]

        merged = pred.merge(ad, on="canonical_smiles", how="left")

        for region in ["inside", "outside"]:
            sub = merged[merged["ad_region"] == region]
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

            if dataset in REGRESSION and "y_pred" in sub.columns:
                mse = mean_squared_error(sub["Y"], sub["y_pred"])
                row["rmse"] = float(np.sqrt(mse))
                row["mae"] = mean_absolute_error(sub["Y"], sub["y_pred"])

            rows.append(row)

    out_dir = Path("outputs/reports/applicability_domain")
    out = pd.DataFrame(rows)
    out_path = out_dir / "ad_vs_performance.csv"
    out.to_csv(out_path, index=False)

    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
