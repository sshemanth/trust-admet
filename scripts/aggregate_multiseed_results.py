from pathlib import Path
import pandas as pd
import numpy as np

root = Path("outputs/reports/tables")

leaderboard = pd.read_csv(root / "leaderboard_all_metrics.csv")

group_cols = ["dataset", "split", "model"]

numeric_cols = leaderboard.select_dtypes(include=[np.number]).columns.tolist()

# Don't aggregate the seed itself
numeric_cols = [c for c in numeric_cols if c != "seed"]

rows = []

for keys, group in leaderboard.groupby(group_cols):

    row = {
        "dataset": keys[0],
        "split": keys[1],
        "model": keys[2],
        "n_seeds": len(group),
    }

    for metric in numeric_cols:
        vals = group[metric].dropna()

        if len(vals) == 0:
            continue

        row[f"{metric}_mean"] = vals.mean()
        row[f"{metric}_std"] = vals.std(ddof=1) if len(vals) > 1 else 0.0
        row[f"{metric}_min"] = vals.min()
        row[f"{metric}_max"] = vals.max()

        row[f"{metric}_mean_std"] = (
            f"{vals.mean():.3f} ± "
            f"{(vals.std(ddof=1) if len(vals)>1 else 0):.3f}"
        )

    rows.append(row)

summary = pd.DataFrame(rows)

summary = summary.sort_values(
    ["dataset", "split", "model"]
)

out = root / "leaderboard_mean_std.csv"

summary.to_csv(out, index=False)

print(summary.head(20).to_string(index=False))

print("\nSaved:", out)
