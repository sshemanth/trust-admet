from pathlib import Path
import pandas as pd
from clearml import Task
from tdc.single_pred import ADME, Tox


DATASETS = {
    "BBBP": ("ADME", "BBB_Martins", None),
    "Lipophilicity": ("ADME", "Lipophilicity_AstraZeneca", None),
    "Solubility": ("ADME", "Solubility_AqSolDB", None),
    "ClinTox": ("Tox", "ClinTox", None),

    # Tox21 requires a specific label_name in TDC
    "Tox21_NR-AR": ("Tox", "Tox21", "NR-AR"),
    "Tox21_SR-p53": ("Tox", "Tox21", "SR-p53"),
}


def load_dataset(group: str, name: str, label_name=None):
    if group == "ADME":
        return ADME(name=name)
    if group == "Tox":
        if label_name is not None:
            return Tox(name=name, label_name=label_name)
        return Tox(name=name)
    raise ValueError(f"Unknown dataset group: {group}")


def main():
    task = Task.init(
        project_name="TRUST-ADMET/Data",
        task_name="download_public_admet_datasets",
    )

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    summary = []

    for dataset_name, (group, tdc_name, label_name) in DATASETS.items():
        try:
            print(f"Downloading {dataset_name} from TDC: {tdc_name}")
            data = load_dataset(group, tdc_name, label_name)
            df = data.get_data()

            output_path = raw_dir / f"{dataset_name}.csv"
            df.to_csv(output_path, index=False)

            summary.append(
                {
                    "dataset": dataset_name,
                    "tdc_name": tdc_name,
                    "label_name": label_name,
                    "rows": len(df),
                    "columns": "|".join(df.columns),
                    "status": "success",
                    "output": str(output_path),
                }
            )

            task.upload_artifact(
                name=f"{dataset_name}_raw_csv",
                artifact_object=str(output_path),
            )

        except Exception as e:
            print(f"FAILED: {dataset_name}: {e}")
            summary.append(
                {
                    "dataset": dataset_name,
                    "tdc_name": tdc_name,
                    "label_name": label_name,
                    "rows": 0,
                    "columns": "",
                    "status": f"failed: {e}",
                    "output": "",
                }
            )

    summary_df = pd.DataFrame(summary)
    summary_path = raw_dir / "dataset_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    task.upload_artifact(
        name="dataset_summary",
        artifact_object=str(summary_path),
    )

    print(summary_df)
    task.close()


if __name__ == "__main__":
    main()
