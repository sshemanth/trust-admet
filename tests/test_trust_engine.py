import pytest

from trust_admet.trust.trust_engine import validate_smiles, predict_with_trust


def test_validate_smiles_valid():
    assert validate_smiles("CCO") == "CCO"


def test_validate_smiles_invalid():
    with pytest.raises(ValueError):
        validate_smiles("hello")


def test_predict_with_trust_valid():
    result = predict_with_trust(
        smiles="CCO",
        dataset="BBBP",
        split="scaffold",
        model_name="random_forest",
        seed="42",
    )

    assert "prediction" in result
    assert "trust_level" in result
    assert "recommendation" in result
