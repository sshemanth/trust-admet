from trust_admet.trust.trust_engine import predict_with_trust

test_smiles = [
    "CCO",
    "c1ccccc1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CN1CCC[C@H]1c2cccnc2",
]

for s in test_smiles:
    result = predict_with_trust(
        smiles=s,
        dataset="BBBP",
        split="scaffold",
        model_name="random_forest",
        seed="42",
    )

    print("=" * 60)
    print("SMILES:", s)
    print("Prediction:", result["label"])
    print("Probability:", result["probability_positive"])
    print("TRUST Score:", result["trust_score"])
    print("Level:", result["trust_level"])
    print("AD:", result["applicability_domain"])
