from pathlib import Path
import pandas as pd


CLASSIFICATION_DATASETS = {
    "BBBP",
    "ClinTox",
    "Tox21_NR-AR",
    "Tox21_SR-p53",
}

REGRESSION_DATASETS = {
    "Solubility",
    "Lipophilicity",
}


def build_dataset_table(qc_summary_path: Path, output_path: Path):
    df = pd.read_csv(qc_summary_path)

    rows = []
    for _, row in df.iterrows():
        dataset = row["dataset"]
        task_type = "classification" if dataset in CLASSIFICATION_DATASETS else "regression"

        out = {
            "dataset": dataset,
            "task_type": task_type,
            "rows": row["rows"],
            "valid_smiles": row["valid_smiles"],
            "invalid_smiles": row["invalid_smiles"],
            "unique_molecules": row["unique_canonical_smiles"],
            "duplicate_molecules": row["duplicate_canonical_smiles"],
            "mol_weight_mean": row["mol_weight_mean"],
            "logp_mean": row["logp_mean"],
        }

        if task_type == "classification":
            out["class_distribution"] = row.get("target_value_counts", "")
        else:
            out["target_mean"] = row.get("target_mean", "")
            out["target_std"] = row.get("target_std", "")

        rows.append(out)

    table = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    return table


def collect_baseline_metrics(models_dir: Path, output_path: Path):
    rows = []

    for metrics_path in models_dir.glob("*/*/*/seed*_metrics.csv"):
        parts = metrics_path.parts

        # outputs/models/DATASET/SPLIT/MODEL/seed42_metrics.csv
        dataset = parts[-4]
        split = parts[-3]
        model = parts[-2]
        seed = metrics_path.stem.replace("_metrics", "").replace("seed", "")

        df = pd.read_csv(metrics_path)
        row = df.iloc[0].to_dict()
        row.update(
            {
                "dataset": dataset,
                "split": split,
                "model": model,
                "seed": int(seed),
            }
        )
        rows.append(row)

    table = pd.DataFrame(rows)

    if not table.empty:
        front_cols = ["dataset", "split", "model", "seed"]
        other_cols = [c for c in table.columns if c not in front_cols]
        table = table[front_cols + other_cols]
        table = table.sort_values(["dataset", "split", "model", "seed"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    return table
