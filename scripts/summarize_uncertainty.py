from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np

CLASSIFICATION = {"BBBP", "ClinTox", "Tox21_NR-AR", "Tox21_SR-p53"}
REGRESSION = {"Lipophilicity", "Solubility"}

rows = []

for path in Path("outputs/reports/uncertainty").glob("*_uncertainty.csv"):
    name = path.stem
    parts = name.split("_")

    # Expected: mlp_<dataset>_<split>_seed42_uncertainty
    model = parts[0]
    split = parts[-3]
    dataset = "_".join(parts[1:-3])

    df = pd.read_csv(path)

    q75 = df["mc_std"].quantile(0.75)
    high = df[df["mc_std"] >= q75]
    low = df[df["mc_std"] < q75]

    row = {
        "dataset": dataset,
        "split": split,
        "model": model,
        "mean_uncertainty": df["mc_std"].mean(),
        "high_fraction": len(high) / len(df),
    }

    if dataset in CLASSIFICATION:
        high_pred = (high["mc_mean"] >= 0.5).astype(int)
        low_pred = (low["mc_mean"] >= 0.5).astype(int)

        row["high_accuracy"] = accuracy_score(high["Y"], high_pred)
        row["low_accuracy"] = accuracy_score(low["Y"], low_pred)
        row["accuracy_gap"] = row["low_accuracy"] - row["high_accuracy"]

    elif dataset in REGRESSION:
        high_rmse = np.sqrt(mean_squared_error(high["Y"], high["mc_mean"]))
        low_rmse = np.sqrt(mean_squared_error(low["Y"], low["mc_mean"]))

        row["high_rmse"] = high_rmse
        row["low_rmse"] = low_rmse
        row["rmse_gap"] = high_rmse - low_rmse

    rows.append(row)

out = pd.DataFrame(rows)
out_path = Path("outputs/reports/uncertainty/uncertainty_summary.csv")
out.to_csv(out_path, index=False)

print(out.sort_values(["dataset", "split"]).to_string(index=False))
