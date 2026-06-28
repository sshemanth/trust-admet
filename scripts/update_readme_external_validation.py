from pathlib import Path
import pandas as pd


README = Path("README.md")
CSV = Path("outputs/reports/external_validation/B3DB_BBBP_external_metrics.csv")

START = "<!-- EXTERNAL_VALIDATION_START -->"
END = "<!-- EXTERNAL_VALIDATION_END -->"


DISPLAY_MODEL = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


def fmt(x):
    return f"{x:.3f}"


def build_section(df):
    rows = []
    rows.append("| External Dataset | Train Dataset | Model | N | Removed Overlap | AUROC | AUPRC | MCC | ECE |")
    rows.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")

    for _, r in df.iterrows():
        rows.append(
            f"| {r['external_dataset']} "
            f"| {r['train_dataset']} "
            f"| {DISPLAY_MODEL.get(r['model'], r['model'])} "
            f"| {int(r['n_external'])} "
            f"| {int(r['n_removed_overlap'])} "
            f"| {fmt(r['auroc'])} "
            f"| {fmt(r['auprc'])} "
            f"| {fmt(r['mcc'])} "
            f"| {fmt(r['ece'])} |"
        )

    table = "\n".join(rows)

    return f"""{START}
# 🌍 External Validation

TRUST-ADMET evaluates BBBP-trained models on the independent B3DB dataset after removing molecules overlapping with the BBBP training set.

{table}

## External Validation Summary

- B3DB molecules before overlap removal: 7,807
- B3DB molecules after overlap removal: 6,482
- Removed train-overlap molecules: 1,325
- Random Forest achieved strong external generalization with AUROC above 0.91.
- XGBoost also retained strong external performance with AUROC above 0.88.

{END}"""


def main():
    if not CSV.exists():
        raise FileNotFoundError(f"Missing CSV: {CSV}")

    df = pd.read_csv(CSV)
    new_section = build_section(df)

    text = README.read_text()

    if START in text and END in text:
        before = text.split(START)[0].rstrip()
        after = text.split(END)[1].lstrip()
        updated = before + "\n\n" + new_section + "\n\n" + after
    else:
        updated = text.rstrip() + "\n\n" + new_section + "\n"

    README.write_text(updated)
    print("Updated README external validation section from:", CSV)


if __name__ == "__main__":
    main()
