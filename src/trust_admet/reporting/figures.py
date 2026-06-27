from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def figure_dataset_sizes(dataset_table: pd.DataFrame, output_path: Path):
    df = dataset_table.sort_values("rows", ascending=False)

    plt.figure(figsize=(8, 5))
    plt.bar(df["dataset"], df["rows"])
    plt.ylabel("Number of molecules")
    plt.xlabel("Dataset")
    plt.title("Dataset sizes")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def figure_classification_auroc(baseline_table: pd.DataFrame, output_path: Path):
    if baseline_table.empty or "test_auroc" not in baseline_table.columns:
        return

    df = baseline_table.dropna(subset=["test_auroc"]).copy()
    df = df[df["dataset"].isin(["BBBP", "ClinTox", "Tox21_NR-AR", "Tox21_SR-p53"])]

    if df.empty:
        return

    labels = (
        df["dataset"]
        + " | "
        + df["split"]
        + " | "
        + df["model"]
    )

    plt.figure(figsize=(12, 6))
    plt.bar(labels, df["test_auroc"])
    plt.ylabel("Test AUROC")
    plt.xlabel("Experiment")
    plt.title("Classical baseline AUROC")
    plt.xticks(rotation=75, ha="right")
    plt.ylim(0, 1)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def figure_regression_rmse(baseline_table: pd.DataFrame, output_path: Path):
    if baseline_table.empty or "test_rmse" not in baseline_table.columns:
        return

    df = baseline_table.dropna(subset=["test_rmse"]).copy()
    df = df[df["dataset"].isin(["Solubility", "Lipophilicity"])]

    if df.empty:
        return

    labels = (
        df["dataset"]
        + " | "
        + df["split"]
        + " | "
        + df["model"]
    )

    plt.figure(figsize=(10, 5))
    plt.bar(labels, df["test_rmse"])
    plt.ylabel("Test RMSE")
    plt.xlabel("Experiment")
    plt.title("Classical baseline regression RMSE")
    plt.xticks(rotation=75, ha="right")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
