from pathlib import Path
import shutil
import pandas as pd
import matplotlib.pyplot as plt


paper_fig = Path("paper/figures")
paper_tab = Path("paper/tables")
paper_fig.mkdir(parents=True, exist_ok=True)
paper_tab.mkdir(parents=True, exist_ok=True)


def copy_table(src, name):
    src = Path(src)
    if src.exists():
        dst = paper_tab / name
        shutil.copy(src, dst)
        print(f"Table: {dst}")


def save_classification_bar():
    path = Path("outputs/reports/tables/leaderboard_classification.csv")
    if not path.exists():
        return

    df = pd.read_csv(path)
    df = df[df["split"] == "scaffold"].copy()

    plt.figure(figsize=(10, 5))
    labels = df["dataset"] + " | " + df["model"]
    plt.bar(labels, df["test_auroc"])
    plt.ylabel("Test AUROC")
    plt.title("Scaffold split classification performance")
    plt.xticks(rotation=75, ha="right")
    plt.ylim(0, 1)
    plt.tight_layout()
    out = paper_fig / "fig2_classification_scaffold_auroc.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Figure: {out}")


def save_regression_bar():
    path = Path("outputs/reports/tables/leaderboard_regression.csv")
    if not path.exists():
        return

    df = pd.read_csv(path)
    df = df[df["split"] == "scaffold"].copy()

    plt.figure(figsize=(9, 5))
    labels = df["dataset"] + " | " + df["model"]
    plt.bar(labels, df["test_rmse"])
    plt.ylabel("Test RMSE")
    plt.title("Scaffold split regression performance")
    plt.xticks(rotation=75, ha="right")
    plt.tight_layout()
    out = paper_fig / "fig2_regression_scaffold_rmse.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Figure: {out}")


def save_ad_gap():
    path = Path("outputs/reports/applicability_domain/ad_performance_gap.csv")
    if not path.exists():
        return

    df = pd.read_csv(path)

    if "auroc_gap" in df.columns:
        sub = df.dropna(subset=["auroc_gap"]).copy()
        if not sub.empty:
            plt.figure(figsize=(10, 5))
            labels = sub["dataset"] + " | " + sub["split"] + " | " + sub["model"]
            plt.bar(labels, sub["auroc_gap"])
            plt.ylabel("Inside AD AUROC - Outside AD AUROC")
            plt.title("Applicability-domain performance gap")
            plt.xticks(rotation=75, ha="right")
            plt.tight_layout()
            out = paper_fig / "fig4_ad_auroc_gap.png"
            plt.savefig(out, dpi=300)
            plt.close()
            print(f"Figure: {out}")

    if "rmse_gap" in df.columns:
        sub = df.dropna(subset=["rmse_gap"]).copy()
        if not sub.empty:
            plt.figure(figsize=(8, 5))
            labels = sub["dataset"] + " | " + sub["split"] + " | " + sub["model"]
            plt.bar(labels, sub["rmse_gap"])
            plt.ylabel("Outside AD RMSE - Inside AD RMSE")
            plt.title("Applicability-domain regression gap")
            plt.xticks(rotation=75, ha="right")
            plt.tight_layout()
            out = paper_fig / "fig4_ad_rmse_gap.png"
            plt.savefig(out, dpi=300)
            plt.close()
            print(f"Figure: {out}")


def save_uncertainty():
    path = Path("outputs/reports/uncertainty/uncertainty_summary.csv")
    if not path.exists():
        return

    df = pd.read_csv(path)

    if "accuracy_gap" in df.columns:
        sub = df.dropna(subset=["accuracy_gap"]).copy()
        if not sub.empty:
            plt.figure(figsize=(10, 5))
            labels = sub["dataset"] + " | " + sub["split"]
            plt.bar(labels, sub["accuracy_gap"])
            plt.ylabel("Low uncertainty accuracy - High uncertainty accuracy")
            plt.title("Uncertainty separates reliable and unreliable predictions")
            plt.xticks(rotation=75, ha="right")
            plt.tight_layout()
            out = paper_fig / "fig5_uncertainty_accuracy_gap.png"
            plt.savefig(out, dpi=300)
            plt.close()
            print(f"Figure: {out}")

    if "rmse_gap" in df.columns:
        sub = df.dropna(subset=["rmse_gap"]).copy()
        if not sub.empty:
            plt.figure(figsize=(8, 5))
            labels = sub["dataset"] + " | " + sub["split"]
            plt.bar(labels, sub["rmse_gap"])
            plt.ylabel("High uncertainty RMSE - Low uncertainty RMSE")
            plt.title("Regression uncertainty gap")
            plt.xticks(rotation=75, ha="right")
            plt.tight_layout()
            out = paper_fig / "fig5_uncertainty_rmse_gap.png"
            plt.savefig(out, dpi=300)
            plt.close()
            print(f"Figure: {out}")


def copy_explainability_figures():
    srcs = [
        Path("outputs/reports/explainability/BBBP/scaffold/random_forest/seed42_shap_summary.png"),
        Path("outputs/reports/explainability/BBBP/scaffold/xgboost/seed42_xgboost_importance.png"),
    ]

    for src in srcs:
        if src.exists():
            dst = paper_fig / f"fig6_{src.name}"
            shutil.copy(src, dst)
            print(f"Figure: {dst}")


def main():
    copy_table("outputs/reports/tables/table1_dataset_summary.csv", "table1_dataset_summary.csv")
    copy_table("outputs/reports/tables/leaderboard_classification.csv", "table2_classification_leaderboard.csv")
    copy_table("outputs/reports/tables/leaderboard_regression.csv", "table3_regression_leaderboard.csv")
    copy_table("outputs/reports/applicability_domain/ad_performance_gap.csv", "table4_ad_performance_gap.csv")
    copy_table("outputs/reports/uncertainty/uncertainty_summary.csv", "table5_uncertainty_summary.csv")

    save_classification_bar()
    save_regression_bar()
    save_ad_gap()
    save_uncertainty()
    copy_explainability_figures()


if __name__ == "__main__":
    main()
