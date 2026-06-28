import argparse
from pathlib import Path

import joblib
import pandas as pd

from trust_admet.data.featurize import dataframe_to_fingerprints
from trust_admet.trust.trust_profile import TrustProfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--dataset", default="BBBP")
    parser.add_argument("--split", default="scaffold")
    parser.add_argument("--model", default="random_forest")
    parser.add_argument("--seed", default="42")
    args = parser.parse_args()

    model_path = (
        Path("outputs/models")
        / args.dataset
        / args.split
        / args.model
        / f"seed{args.seed}.joblib"
    )

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)

    df = pd.DataFrame({"canonical_smiles": [args.smiles]})
    x = dataframe_to_fingerprints(df)

    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(x)[0, 1])
        pred = int(prob >= 0.5)

        # Temporary values; later we will compute these from AD + uncertainty modules.
        similarity = 0.81
        uncertainty = 0.02

        trust = TrustProfile(
            probability=prob,
            similarity=similarity,
            uncertainty=uncertainty,
        )

        print("=" * 60)
        print("                 TRUST-ADMET REPORT")
        print("=" * 60)
        print(f"SMILES               : {args.smiles}")
        print(f"Dataset              : {args.dataset}")
        print(f"Model                : {args.model}")
        print(f"Prediction           : {pred}")
        print(f"Probability          : {prob:.3f}")
        print()
        print(f"Applicability Domain : {'Inside' if similarity >= 0.5 else 'Outside'}")
        print(f"Nearest Similarity   : {similarity:.3f}")
        print(f"Uncertainty          : {uncertainty:.3f}")
        print()
        print(f"Overall Confidence   : {trust.confidence}")
        print(f"Recommendation       : {trust.recommendation}")
        print("=" * 60)

    else:
        pred = model.predict(x)[0]
        print("=" * 60)
        print("                 TRUST-ADMET REPORT")
        print("=" * 60)
        print(f"SMILES               : {args.smiles}")
        print(f"Dataset              : {args.dataset}")
        print(f"Model                : {args.model}")
        print(f"Prediction           : {pred:.3f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
