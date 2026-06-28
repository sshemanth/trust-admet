from pathlib import Path
import argparse
import joblib
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="scaffold")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    model_path = Path("outputs/models") / args.dataset / args.split / "xgboost" / f"seed{args.seed}.joblib"
    model = joblib.load(model_path)

    booster = model.get_booster()
    scores = booster.get_score(importance_type="gain")

    rows = []
    for feature, score in scores.items():
        bit = int(feature.replace("f", ""))
        rows.append({"fingerprint_bit": bit, "importance_gain": score})

    df = pd.DataFrame(rows).sort_values("importance_gain", ascending=False)
    df_top = df.head(args.top_k)

    out_dir = Path("outputs/reports/explainability") / args.dataset / args.split / "xgboost"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"seed{args.seed}_xgboost_importance.csv"
    fig_path = out_dir / f"seed{args.seed}_xgboost_importance.png"

    df_top.to_csv(csv_path, index=False)

    plt.figure(figsize=(8, 6))
    plt.barh(df_top["fingerprint_bit"].astype(str), df_top["importance_gain"])
    plt.gca().invert_yaxis()
    plt.xlabel("Gain importance")
    plt.ylabel("Morgan fingerprint bit")
    plt.title(f"XGBoost feature importance: {args.dataset} {args.split}")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"Saved: {csv_path}")
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
