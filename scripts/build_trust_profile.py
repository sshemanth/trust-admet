from pathlib import Path
import pandas as pd


def main():
    dataset = "BBBP"
    split = "scaffold"
    model = "mlp"
    seed = 42

    pred_path = Path(f"outputs/reports/uncertainty/{model}_{dataset}_{split}_seed{seed}_uncertainty.csv")
    ad_path = Path(f"outputs/reports/applicability_domain/{dataset}_{split}_test_ad.csv")

    pred = pd.read_csv(pred_path)
    ad = pd.read_csv(ad_path)[["canonical_smiles", "ad_region", "ad_max_tanimoto"]]

    df = pred.merge(ad, on="canonical_smiles", how="left")

    df["predicted_label"] = (df["mc_mean"] >= 0.5).astype(int)
    df["correct"] = df["predicted_label"] == df["Y"]

    q75 = df["mc_std"].quantile(0.75)
    df["uncertainty_level"] = df["mc_std"].apply(
        lambda x: "high" if x >= q75 else "low_or_medium"
    )

    df["trust_profile"] = df.apply(
        lambda r: (
            f"AD={r['ad_region']}; "
            f"uncertainty={r['uncertainty_level']}; "
            f"confidence={r['mc_mean']:.3f}; "
            f"correct={r['correct']}"
        ),
        axis=1,
    )

    out = df[
        [
            "canonical_smiles",
            "Y",
            "predicted_label",
            "mc_mean",
            "mc_std",
            "ad_region",
            "ad_max_tanimoto",
            "uncertainty_level",
            "correct",
            "trust_profile",
        ]
    ].copy()

    out_dir = Path("outputs/reports/trust_profiles")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{model}_{dataset}_{split}_seed{seed}_trust_profile.csv"
    out.to_csv(out_path, index=False)

    paper_dir = Path("paper/tables")
    paper_dir.mkdir(parents=True, exist_ok=True)
    out.head(25).to_csv(paper_dir / "table6_example_trust_profiles.csv", index=False)

    print(out.head(25).to_string(index=False))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
