from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    "font.size": 11,
    "figure.dpi": 300,
})

df = pd.read_csv(
    "outputs/reports/external_validation/B3DB_BBBP_external_metrics.csv"
)

paper_fig = Path("paper/figures")
paper_fig.mkdir(parents=True, exist_ok=True)

metrics = ["auroc", "auprc", "mcc"]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for ax, metric in zip(axes, metrics):

    ax.bar(
        df["model"],
        df[metric],
    )

    ax.set_ylim(0, 1)
    ax.set_title(metric.upper())
    ax.set_ylabel(metric.upper())

plt.tight_layout()

out = paper_fig / "Figure7_External_Validation.png"
plt.savefig(out)

print("Saved:", out)
