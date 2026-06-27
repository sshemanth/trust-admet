from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


def reliability_diagram(prediction_csv: Path, output_path: Path, n_bins=10):
    df = pd.read_csv(prediction_csv)
    df = df[df["split"] == "test"].copy()

    y_true = df["Y"].values
    y_prob = df["y_prob"].values

    prob_true, prob_pred = calibration_curve(
        y_true,
        y_prob,
        n_bins=n_bins,
        strategy="uniform",
    )

    plt.figure(figsize=(5, 5))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed fraction positive")
    plt.title("Reliability diagram")
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
