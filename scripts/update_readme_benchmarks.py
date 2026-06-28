from pathlib import Path
import pandas as pd


README = Path("README.md")
CSV = Path("outputs/reports/tables/leaderboard_mean_std.csv")


START = "<!-- BENCHMARK_RESULTS_START -->"
END = "<!-- BENCHMARK_RESULTS_END -->"


CLASSIFICATION_DATASETS = ["BBBP", "ClinTox", "Tox21_NR-AR", "Tox21_SR-p53"]
REGRESSION_DATASETS = ["Lipophilicity", "Solubility"]


DISPLAY_DATASET = {
    "BBBP": "BBBP",
    "ClinTox": "ClinTox",
    "Tox21_NR-AR": "Tox21 NR-AR",
    "Tox21_SR-p53": "Tox21 SR-p53",
    "Lipophilicity": "Lipophilicity",
    "Solubility": "Solubility",
}


DISPLAY_MODEL = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "mlp": "MLP",
    "gcn": "GCN",
    "gin": "GIN",
    "chemberta": "ChemBERTa",
}


def fmt(row, col):
    mean_std_col = f"{col}_mean_std"
    if mean_std_col in row and pd.notna(row[mean_std_col]):
        return str(row[mean_std_col])

    mean_col = f"{col}_mean"
    std_col = f"{col}_std"
    if mean_col in row and pd.notna(row[mean_col]):
        std = row[std_col] if std_col in row and pd.notna(row[std_col]) else 0.0
        return f"{row[mean_col]:.3f} ± {std:.3f}"

    return "—"


def make_classification_table(df):
    rows = []
    cls = df[
        (df["split"] == "scaffold")
        & (df["dataset"].isin(CLASSIFICATION_DATASETS))
        & (df["test_auroc_mean"].notna())
    ].copy()

    cls = cls.sort_values(["dataset", "test_auroc_mean"], ascending=[True, False])

    rows.append("| Dataset | Model | Seeds | AUROC | AUPRC | MCC | ECE |")
    rows.append("|---|---:|---:|---:|---:|---:|---:|")

    for _, row in cls.iterrows():
        rows.append(
            f"| {DISPLAY_DATASET.get(row['dataset'], row['dataset'])} "
            f"| {DISPLAY_MODEL.get(row['model'], row['model'])} "
            f"| {int(row['n_seeds'])} "
            f"| {fmt(row, 'test_auroc')} "
            f"| {fmt(row, 'test_auprc')} "
            f"| {fmt(row, 'test_mcc')} "
            f"| {fmt(row, 'test_ece')} |"
        )

    return "\n".join(rows)


def make_regression_table(df):
    rows = []
    reg = df[
        (df["split"] == "scaffold")
        & (df["dataset"].isin(REGRESSION_DATASETS))
        & (df["test_rmse_mean"].notna())
    ].copy()

    reg = reg.sort_values(["dataset", "test_rmse_mean"], ascending=[True, True])

    rows.append("| Dataset | Model | Seeds | RMSE | MAE | R² |")
    rows.append("|---|---:|---:|---:|---:|---:|")

    for _, row in reg.iterrows():
        rows.append(
            f"| {DISPLAY_DATASET.get(row['dataset'], row['dataset'])} "
            f"| {DISPLAY_MODEL.get(row['model'], row['model'])} "
            f"| {int(row['n_seeds'])} "
            f"| {fmt(row, 'test_rmse')} "
            f"| {fmt(row, 'test_mae')} "
            f"| {fmt(row, 'test_r2')} |"
        )

    return "\n".join(rows)


def build_section(df):
    return f"""{START}
# 📊 Benchmark Results

TRUST-ADMET evaluates multiple model families across ADMET classification and regression tasks. The tables below report **scaffold split test performance** as **mean ± standard deviation** across available seeds.

## Classification Tasks

{make_classification_table(df)}

## Regression Tasks

{make_regression_table(df)}

## Key Findings

- Classical models remain highly competitive on BBBP scaffold evaluation.
- GIN provides strong regression performance on Lipophilicity and Solubility.
- Model performance varies across tasks, supporting model-family comparison rather than assuming one architecture is universally superior.
- Calibration, applicability-domain analysis, conformal prediction, ensemble agreement, uncertainty, and explainability are essential for trust-aware ADMET prediction.
{END}"""


def main():
    if not CSV.exists():
        raise FileNotFoundError(f"Missing CSV: {CSV}")

    if not README.exists():
        raise FileNotFoundError(f"Missing README: {README}")

    df = pd.read_csv(CSV)
    new_section = build_section(df)

    text = README.read_text()

    if START in text and END in text:
        before = text.split(START)[0].rstrip()
        after = text.split(END)[1].lstrip()
        updated = before + "\n\n" + new_section + "\n\n" + after
    else:
        updated = text.rstrip() + "\n\n" + new_section + "\n"

    README.write_text(updated)
    print("Updated README benchmark section from:", CSV)


if __name__ == "__main__":
    main()
