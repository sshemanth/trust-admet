from pathlib import Path
from clearml import Task

from trust_admet.reporting.tables import (
    build_dataset_table,
    collect_baseline_metrics,
)
from trust_admet.reporting.figures import (
    figure_dataset_sizes,
    figure_classification_auroc,
    figure_regression_rmse,
)


def main():
    task = Task.init(
        project_name="TRUST-ADMET/Reporting",
        task_name="generate_baseline_tables_and_figures",
    )

    reports_dir = Path("outputs/reports")
    tables_dir = reports_dir / "tables"
    figures_dir = reports_dir / "figures"

    dataset_table = build_dataset_table(
        qc_summary_path=reports_dir / "dataset_qc_summary.csv",
        output_path=tables_dir / "table1_dataset_summary.csv",
    )

    baseline_table = collect_baseline_metrics(
        models_dir=Path("outputs/models"),
        output_path=tables_dir / "table2_classical_baselines.csv",
    )

    figure_dataset_sizes(
        dataset_table,
        figures_dir / "figure1_dataset_sizes.png",
    )

    figure_classification_auroc(
        baseline_table,
        figures_dir / "figure2_classification_auroc.png",
    )

    figure_regression_rmse(
        baseline_table,
        figures_dir / "figure3_regression_rmse.png",
    )

    for path in tables_dir.glob("*.csv"):
        task.upload_artifact(path.stem, str(path))

    for path in figures_dir.glob("*.png"):
        task.upload_artifact(path.stem, str(path))

    print("Generated tables:")
    for path in tables_dir.glob("*.csv"):
        print(path)

    print("Generated figures:")
    for path in figures_dir.glob("*.png"):
        print(path)

    task.close()


if __name__ == "__main__":
    main()
