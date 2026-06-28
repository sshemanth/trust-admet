from pathlib import Path
import pandas as pd


def collect_metrics():
    rows = []
    model_root = Path("outputs/models")

    metric_files = list(model_root.glob("*/*/*/seed*_metrics.csv"))
    metric_files += list(model_root.glob("*/*/*/seed*/metrics.csv"))

    for path in metric_files:
        parts = path.parts

        if path.name == "metrics.csv":
            dataset = parts[-5]
            split = parts[-4]
            model = parts[-3]
            seed = parts[-2].replace("seed", "")
        else:
            dataset = parts[-4]
            split = parts[-3]
            model = parts[-2]
            seed = path.stem.replace("_metrics", "").replace("seed", "")

        df = pd.read_csv(path)
        row = df.iloc[0].to_dict()
        row.update({
            "dataset": dataset,
            "split": split,
            "model": model,
            "seed": int(seed),
        })
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    out_dir = Path("outputs/reports/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = collect_metrics()

    front_cols = ["dataset", "split", "model", "seed"]
    other_cols = [c for c in df.columns if c not in front_cols]
    df = df[front_cols + other_cols]

    df.to_csv(out_dir / "leaderboard_all_metrics.csv", index=False)

    classification = df[df["test_auroc"].notna()].copy()
    if not classification.empty:
        classification = classification.sort_values(
            ["dataset", "split", "test_auroc"],
            ascending=[True, True, False],
        )
        classification.to_csv(
            out_dir / "leaderboard_classification.csv",
            index=False,
        )

    regression = df[df["test_rmse"].notna()].copy()
    if not regression.empty:
        regression = regression.sort_values(
            ["dataset", "split", "test_rmse"],
            ascending=[True, True, True],
        )
        regression.to_csv(
            out_dir / "leaderboard_regression.csv",
            index=False,
        )

    print("Saved:")
    print(out_dir / "leaderboard_all_metrics.csv")
    print(out_dir / "leaderboard_classification.csv")
    print(out_dir / "leaderboard_regression.csv")

    print("\nTop classification rows:")
    if not classification.empty:
        print(classification[["dataset", "split", "model", "test_auroc", "test_auprc", "test_mcc", "test_ece"]].head(20).to_string(index=False))

    print("\nTop regression rows:")
    if not regression.empty:
        print(regression[["dataset", "split", "model", "test_rmse", "test_mae", "test_r2"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
