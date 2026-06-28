from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

root = Path("outputs/reports/tables")
df = pd.read_csv(root / "leaderboard_mean_std.csv")

paper_fig = Path("paper/figures")
paper_fig.mkdir(parents=True, exist_ok=True)

# ---------- Classification ----------
cls = df[df["test_auroc_mean"].notna()].copy()

for dataset in sorted(cls["dataset"].unique()):
    sub = cls[(cls["dataset"] == dataset) & (cls["split"] == "scaffold")]

    if sub.empty:
        continue

    plt.figure(figsize=(7, 4))

    plt.bar(
        sub["model"],
        sub["test_auroc_mean"],
        yerr=sub["test_auroc_std"],
        capsize=5,
    )

    plt.ylim(0.5, 1.0)
    plt.ylabel("Test AUROC")
    plt.title(f"{dataset} (Scaffold)")
    plt.tight_layout()

    out = paper_fig / f"benchmark_{dataset}_classification.png"
    plt.savefig(out, dpi=300)
    plt.close()

    print(out)

# ---------- Regression ----------
reg = df[df["test_rmse_mean"].notna()].copy()

for dataset in sorted(reg["dataset"].unique()):
    sub = reg[(reg["dataset"] == dataset) & (reg["split"] == "scaffold")]

    if sub.empty:
        continue

    plt.figure(figsize=(7, 4))

    plt.bar(
        sub["model"],
        sub["test_rmse_mean"],
        yerr=sub["test_rmse_std"],
        capsize=5,
    )

    plt.ylabel("Test RMSE")
    plt.title(f"{dataset} (Scaffold)")
    plt.tight_layout()

    out = paper_fig / f"benchmark_{dataset}_regression.png"
    plt.savefig(out, dpi=300)
    plt.close()

    print(out)

print("Done.")
