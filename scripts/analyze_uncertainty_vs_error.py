from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

inp = Path("outputs/reports/uncertainty/mlp_BBBP_scaffold_seed42_uncertainty.csv")
df = pd.read_csv(inp)

df["y_pred"] = (df["mc_mean"] >= 0.5).astype(int)
df["error"] = (df["y_pred"] != df["Y"]).astype(int)

df["uncertainty_bin"] = pd.qcut(
    df["mc_std"],
    q=4,
    labels=["low", "medium_low", "medium_high", "high"],
    duplicates="drop",
)

rows = []
for name, sub in df.groupby("uncertainty_bin", observed=False):
    rows.append({
        "uncertainty_bin": name,
        "n": len(sub),
        "mean_uncertainty": sub["mc_std"].mean(),
        "error_rate": sub["error"].mean(),
        "accuracy": accuracy_score(sub["Y"], sub["y_pred"]),
        "mean_prediction": sub["mc_mean"].mean(),
    })

out = pd.DataFrame(rows)
out_path = Path("outputs/reports/uncertainty/mlp_BBBP_scaffold_uncertainty_vs_error.csv")
out.to_csv(out_path, index=False)

print(out.to_string(index=False))

print("Overall AUROC:", roc_auc_score(df["Y"], df["mc_mean"]))
print("Overall AUPRC:", average_precision_score(df["Y"], df["mc_mean"]))
