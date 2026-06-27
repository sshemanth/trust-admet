from pathlib import Path
from clearml import Task

from trust_admet.reporting.calibration_figures import reliability_diagram


def main():
    task = Task.init(
        project_name="TRUST-ADMET/Reporting",
        task_name="generate_reliability_diagrams",
    )

    models_dir = Path("outputs/models")
    out_dir = Path("outputs/reports/figures/reliability")
    out_dir.mkdir(parents=True, exist_ok=True)

    prediction_files = sorted(models_dir.glob("*/*/*/seed*_predictions.csv"))

    for pred_path in prediction_files:
        parts = pred_path.parts
        dataset = parts[-4]
        split_method = parts[-3]
        model = parts[-2]

        fig_path = out_dir / f"{dataset}_{split_method}_{model}_reliability.png"

        print(f"Generating {fig_path}")
        reliability_diagram(pred_path, fig_path)

        task.upload_artifact(
            name=fig_path.stem,
            artifact_object=str(fig_path),
        )

    task.close()


if __name__ == "__main__":
    main()
