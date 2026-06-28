from pathlib import Path
import pandas as pd
from clearml import Task

from trust_admet.trust.applicability_domain import add_applicability_scores


def main():
    task = Task.init(
        project_name="TRUST-ADMET/Trust",
        task_name="applicability_domain_v1",
    )

    datasets = ["BBBP", "ClinTox", "Tox21_NR-AR", "Tox21_SR-p53", "Solubility", "Lipophilicity"]
    splits = ["random", "scaffold"]

    out_dir = Path("outputs/reports/applicability_domain")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for dataset in datasets:
        for split in splits:
            base = Path("data/splits") / dataset / split

            train_df = pd.read_csv(base / "train.csv")
            test_df = pd.read_csv(base / "test.csv")

            scored = add_applicability_scores(train_df, test_df)

            out_path = out_dir / f"{dataset}_{split}_test_ad.csv"
            scored.to_csv(out_path, index=False)

            summary_rows.append({
                "dataset": dataset,
                "split": split,
                "test_rows": len(scored),
                "mean_max_tanimoto": scored["ad_max_tanimoto"].mean(),
                "median_max_tanimoto": scored["ad_max_tanimoto"].median(),
                "inside_fraction": (scored["ad_region"] == "inside").mean(),
                "outside_fraction": (scored["ad_region"] == "outside").mean(),
            })

            task.upload_artifact(out_path.stem, str(out_path))

    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "applicability_domain_summary.csv"
    summary.to_csv(summary_path, index=False)

    task.upload_artifact("applicability_domain_summary", str(summary_path))

    print(summary.to_string(index=False))
    task.close()


if __name__ == "__main__":
    main()
