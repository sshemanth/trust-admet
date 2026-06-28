import argparse
import json

from trust_admet.trust.trust_engine import predict_with_trust


def print_report(result):
    b = result["score_breakdown"]

    print("=" * 70)
    print("                       TRUST-ADMET REPORT")
    print("=" * 70)

    if result["prediction"] is None:
        print()
        print("Input Molecule")
        print("--------------")
        print(f"SMILES                   : {result['smiles']}")
        print()
        print("Status")
        print("------")
        print("Prediction               : REFUSED")
        print(f"Reason                   : {result['recommendation']}")
        print()

        if "physchem_ad" in result:
            violations = result["physchem_ad"].get("violations", [])
            if violations:
                print("Applicability Domain Violations")
                print("-------------------------------")
                for v in violations:
                    print(
                        f"- {v['descriptor']}: "
                        f"{v['value']:.2f} "
                        f"(allowed {v['min']:.2f} to {v['max']:.2f})"
                    )

        print()
        print("=" * 70)
        return
    if result["prediction"] is None:
        print()
        print("Input Molecule")
        print("--------------")
        print(f"SMILES                   : {result['smiles']}")
        print()
        print("Status")
        print("------")
        print("Prediction              : REFUSED")
        print(f"Reason                  : {result['recommendation']}")
        print()

        if "physchem_ad" in result:
            violations = result["physchem_ad"].get("violations", [])
            if violations:
                print("Applicability Domain Violations")
                print("-------------------------------")
                for v in violations:
                    print(
                        f"- {v['descriptor']}: "
                        f"{v['value']:.2f} "
                        f"(allowed {v['min']:.2f}-{v['max']:.2f})"
                    )

        print()
        print("=" * 70)
        return

    print()
    print("Input Molecule")
    print("--------------")
    print(f"SMILES                   : {result['smiles']}")
    print()
    print("Model")
    print("-----")
    print(f"Dataset                  : {result['dataset']}")
    print(f"Split                    : {result['split']}")
    print(f"Model                    : {result['model']}")
    print(f"Seed                     : {result['seed']}")
    print()
    print("Prediction")
    print("----------")
    print(f"Class                    : {result['label']}")
    print(f"Probability BBB+         : {result['probability_positive']:.3f}")
    print(f"Prediction Confidence    : {result['prediction_confidence']:.3f}")
    if "conformal_prediction_set" in result:
        print(f"Conformal Set            : {result['conformal_prediction_set']}")
    print()
    print("TRUST Score")
    print("-----------")
    print(f"Total                    : {result['trust_score']:.1f} / 100")
    print(f"Level                    : {result['trust_level']}")
    print()
    print("Breakdown")
    print("---------")
    print(f"Prediction Confidence    : {b['prediction_confidence']:5.1f} / 25")
    print(f"Calibration              : {b['calibration']:5.1f} / 25")
    print(f"Applicability Domain     : {b['applicability_domain']:5.1f} / 25")
    print(f"Uncertainty              : {b['uncertainty']:5.1f} / 25")
    print()
    print("Trust Signals")
    print("-------------")
    print(f"Applicability Domain     : {result['applicability_domain']}")
    print(f"Nearest Similarity       : {result['nearest_similarity']:.3f}")
    print(f"Uncertainty              : {result['uncertainty']:.3f}")
    print(f"ECE                      : {result['ece']:.3f}")
    if "conformal_qhat" in result:
        print(f"Conformal qhat           : {result['conformal_qhat']:.3f}")
    print()
    print("Recommendation")
    print("--------------")
    print(result["recommendation"])
    print()
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--dataset", default="BBBP")
    parser.add_argument("--split", default="scaffold")
    parser.add_argument("--model", default="random_forest")
    parser.add_argument("--seed", default="42")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        result = predict_with_trust(
            smiles=args.smiles,
            dataset=args.dataset,
            split=args.split,
            model_name=args.model,
            seed=args.seed,
        )

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print_report(result)

    except ValueError as e:
        print("=" * 70)
        print("TRUST-ADMET INPUT ERROR")
        print("=" * 70)
        print(str(e))
        print("Please enter a valid molecular SMILES string.")
        print("=" * 70)


if __name__ == "__main__":
    main()
