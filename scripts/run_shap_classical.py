from pathlib import Path
import argparse
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from trust_admet.data.featurize import dataframe_to_fingerprints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="scaffold")
    parser.add_argument("--model", required=True, choices=["random_forest", "xgboost"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=200)
    args = parser.parse_args()

    base = Path("data/splits") / args.dataset / args.split
    test_df = pd.read_csv(base / "test.csv")

    if len(test_df) > args.max_samples:
        test_df = test_df.sample(args.max_samples, random_state=args.seed)

    x_test = dataframe_to_fingerprints(test_df)

    model_path = Path("outputs/models") / args.dataset / args.split / args.model / f"seed{args.seed}.joblib"
    model = joblib.load(model_path)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_test)

    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[1]
    else:
        shap_values_to_plot = shap_values

    # Some SHAP versions return shape: (n_samples, n_features, n_outputs)
    # For binary classification, use the positive class.
    if len(shap_values_to_plot.shape) == 3:
        shap_values_to_plot = shap_values_to_plot[:, :, 1]

    out_dir = Path("outputs/reports/explainability") / args.dataset / args.split / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / f"seed{args.seed}_shap_values.npy", shap_values_to_plot)

    shap.summary_plot(
        shap_values_to_plot,
        x_test,
        show=False,
        max_display=25,
    )
    plt.tight_layout()
    plt.savefig(out_dir / f"seed{args.seed}_shap_summary.png", dpi=300)
    plt.close()

    importance = np.abs(shap_values_to_plot).mean(axis=0)
    top_idx = np.argsort(importance)[::-1][:50]

    pd.DataFrame({
        "fingerprint_bit": top_idx,
        "mean_abs_shap": importance[top_idx],
    }).to_csv(out_dir / f"seed{args.seed}_top_shap_bits.csv", index=False)

    print(f"Saved SHAP outputs to {out_dir}")


if __name__ == "__main__":
    main()
