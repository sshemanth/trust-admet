from pathlib import Path
import pandas as pd
from clearml import Task

from trust_admet.data.validate import validate_dataset


def main():
    task = Task.init(
        project_name="TRUST-ADMET/Data",
        task_name="validate_public_admet_datasets",
    )

    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    reports_dir = Path("outputs/reports")

    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    summaries = []

    csv_files = sorted(
        p for p in raw_dir.glob("*.csv")
        if p.name != "dataset_summary.csv"
    )

    for csv_path in csv_files:
        print(f"Validating {csv_path.name}")

        df_qc, summary = validate_dataset(csv_path)

        qc_path = processed_dir / f"{csv_path.stem}_qc.csv"
        df_qc.to_csv(qc_path, index=False)

        summaries.append(summary)

        task.upload_artifact(
            name=f"{csv_path.stem}_qc",
            artifact_object=str(qc_path),
        )

    summary_df = pd.DataFrame(summaries)
    summary_path = reports_dir / "dataset_qc_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    task.upload_artifact(
        name="dataset_qc_summary",
        artifact_object=str(summary_path),
    )

    print(summary_df.to_string(index=False))
    task.close()


if __name__ == "__main__":
    main()
