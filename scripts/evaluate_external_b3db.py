from pathlib import Path
import joblib
import pandas as pd
from rdkit import Chem

from trust_admet.data.featurize import dataframe_to_fingerprints
from trust_admet.utils.metrics import classification_metrics
from trust_admet.utils.calibration import calibration_metrics


def canonicalize(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find columns among {candidates}. Available: {list(df.columns)}")


def main():
    external_path = Path("data/external/B3DB_classification.tsv")
    df = pd.read_csv(external_path, sep="\t")

    smiles_col = find_column(df, ["SMILES", "smiles", "Smiles", "NO."])
    label_col = find_column(df, ["BBB+/BBB-", "BBB", "label", "Y"])

    ext = pd.DataFrame()
    ext["canonical_smiles"] = df[smiles_col].apply(canonicalize)

    label_raw = df[label_col].astype(str)
    ext["Y"] = label_raw.map({
        "BBB+": 1,
        "BBB-": 0,
        "1": 1,
        "0": 0,
        "True": 1,
        "False": 0,
    })

    ext = ext.dropna(subset=["canonical_smiles", "Y"]).copy()
    ext["Y"] = ext["Y"].astype(int)

    # Remove overlap with BBBP training molecules
    train = pd.read_csv("data/splits/BBBP/scaffold/train.csv")
    train_smiles = set(train["canonical_smiles"].astype(str))

    before = len(ext)
    ext = ext[~ext["canonical_smiles"].isin(train_smiles)].copy()
    after = len(ext)

    print(f"External molecules before train-overlap removal: {before}")
    print(f"External molecules after train-overlap removal : {after}")

    models = ["random_forest", "xgboost"]
    rows = []

    for model_name in models:
        model_path = Path("outputs/models/BBBP/scaffold") / model_name / "seed42.joblib"
        model = joblib.load(model_path)

        x = dataframe_to_fingerprints(ext)
        y_prob = model.predict_proba(x)[:, 1]

        metrics = classification_metrics(ext["Y"].values, y_prob)
        metrics.update(calibration_metrics(ext["Y"].values, y_prob))

        row = {
            "external_dataset": "B3DB",
            "train_dataset": "BBBP",
            "split": "external",
            "model": model_name,
            "seed": 42,
            "n_external": len(ext),
            "n_removed_overlap": before - after,
        }
        row.update(metrics)
        rows.append(row)

        pred = ext.copy()
        pred["model"] = model_name
        pred["y_prob"] = y_prob
        pred["y_pred"] = (y_prob >= 0.5).astype(int)

        pred.to_csv(
            f"outputs/reports/external_validation/B3DB_BBBP_{model_name}_predictions.csv",
            index=False,
        )

    out = pd.DataFrame(rows)
    out_path = Path("outputs/reports/external_validation/B3DB_BBBP_external_metrics.csv")
    out.to_csv(out_path, index=False)

    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
