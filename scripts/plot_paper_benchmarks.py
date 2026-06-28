from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 10,
    "figure.dpi": 300,
})

paper_fig = Path("paper/figures")
paper_fig.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("outputs/reports/tables/leaderboard_mean_std.csv")

# ---------------- Classification ----------------

cls = df[df["test_auroc_mean"].notna()].copy()
cls = cls[cls["split"] == "scaffold"]

datasets = sorted(cls["dataset"].unique())

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for ax, dataset in zip(axes.flat, datasets):

    sub = cls[cls.dataset == dataset].sort_values(
        "test_auroc_mean",
        ascending=False,
    )

    ax.bar(
        sub["model"],
        sub["test_auroc_mean"],
        yerr=sub["test_auroc_std"],
        capsize=4,
    )

    ax.set_ylim(0.5, 1.0)
    ax.set_title(dataset)
    ax.set_ylabel("AUROC")
    ax.tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig(
    paper_fig / "Figure2_Benchmark_Classification.png"
)
plt.close()

# ---------------- Regression ----------------

reg = df[df["test_rmse_mean"].notna()].copy()
reg = reg[reg["split"] == "scaffold"]

datasets = sorted(reg["dataset"].unique())

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, dataset in zip(axes.flat, datasets):

    sub = reg[reg.dataset == dataset].sort_values(
        "test_rmse_mean"
    )

    ax.bar(
        sub["model"],
        sub["test_rmse_mean"],
        yerr=sub["test_rmse_std"],
        capsize=4,
    )

    ax.set_title(dataset)
    ax.set_ylabel("RMSE")
    ax.tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig(
    paper_fig / "Figure3_Benchmark_Regression.png"
)
plt.close()

print("Saved:")
print(paper_fig / "Figure2_Benchmark_Classification.png")
print(paper_fig / "Figure3_Benchmark_Regression.png")
