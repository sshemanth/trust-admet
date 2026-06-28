from pathlib import Path
import pandas as pd

inp = Path("outputs/reports/applicability_domain/ad_vs_performance.csv")
df = pd.read_csv(inp)

rows = []

for keys, group in df.groupby(["dataset", "split", "model", "seed"]):
    dataset, split, model, seed = keys

    inside = group[group["ad_region"] == "inside"]
    outside = group[group["ad_region"] == "outside"]

    if inside.empty or outside.empty:
        continue

    row = {
        "dataset": dataset,
        "split": split,
        "model": model,
        "seed": seed,
        "inside_n": int(inside["n"].iloc[0]),
        "outside_n": int(outside["n"].iloc[0]),
    }

    if "auroc" in inside.columns and pd.notna(inside["auroc"].iloc[0]):
        row["inside_auroc"] = inside["auroc"].iloc[0]
        row["outside_auroc"] = outside["auroc"].iloc[0]
        row["auroc_gap"] = row["inside_auroc"] - row["outside_auroc"]

    if "rmse" in inside.columns and pd.notna(inside["rmse"].iloc[0]):
        row["inside_rmse"] = inside["rmse"].iloc[0]
        row["outside_rmse"] = outside["rmse"].iloc[0]
        row["rmse_gap"] = row["outside_rmse"] - row["inside_rmse"]

    rows.append(row)

out = pd.DataFrame(rows)
out_path = Path("outputs/reports/applicability_domain/ad_performance_gap.csv")
out.to_csv(out_path, index=False)

print(out.to_string(index=False))
