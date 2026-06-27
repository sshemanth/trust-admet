from pathlib import Path
import pandas as pd
from clearml import Task

from trust_admet.data.split import (
    remove_duplicate_molecules,
    random_split,
    scaffold_split,
    summarize_split,
)


def save_split(dataset_name, split_method, train_df, valid_df, test_df):
    out_dir = Path("data/splits") / dataset_name / split_method
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(out_dir / "train.csv", index=False)
    valid_df.to_csv(out_dir / "valid.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    return out_dir


def main():
    task = Task.init(
        project_name="TRUST-ADMET/Data",
        task_name="create_random_and_scaffold_splits",
    )

    processed_dir = Path("data/processed")
    reports_dir = Path("outputs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    all_summary_rows = []

    qc_files = sorted(processed_dir.glob("*_qc.csv"))

    for qc_path in qc_files:
        dataset_name = qc_path.stem.replace("_qc", "")
        print(f"Creating splits for {dataset_name}")

        df = pd.read_csv(qc_path)
        df = df[df["valid_smiles"] == True].copy()
        df = remove_duplicate_molecules(df)

        train_df, valid_df, test_df = random_split(df, seed=42)
        out_dir = save_split(dataset_name, "random", train_df, valid_df, test_df)

        all_summary_rows.extend(
            summarize_split(dataset_name, "random", train_df, valid_df, test_df)
        )

        task.upload_artifact(
            name=f"{dataset_name}_random_splits",
            artifact_object=str(out_dir),
        )

        train_df, valid_df, test_df = scaffold_split(df, seed=42)
        out_dir = save_split(dataset_name, "scaffold", train_df, valid_df, test_df)

        all_summary_rows.extend(
            summarize_split(dataset_name, "scaffold", train_df, valid_df, test_df)
        )

        task.upload_artifact(
            name=f"{dataset_name}_scaffold_splits",
            artifact_object=str(out_dir),
        )

    summary_df = pd.DataFrame(all_summary_rows)
    summary_path = reports_dir / "split_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    task.upload_artifact(
        name="split_summary",
        artifact_object=str(summary_path),
    )

    print(summary_df.to_string(index=False))
    task.close()


if __name__ == "__main__":
    main()
