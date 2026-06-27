from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors


def detect_columns(df: pd.DataFrame):
    smiles_candidates = ["Drug", "SMILES", "smiles", "Smiles"]
    target_candidates = ["Y", "Label", "label", "target", "Target"]

    smiles_col = next((c for c in smiles_candidates if c in df.columns), None)
    target_col = next((c for c in target_candidates if c in df.columns), None)

    if smiles_col is None:
        raise ValueError(f"No SMILES column found. Columns: {list(df.columns)}")

    if target_col is None:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            raise ValueError(f"No target column found. Columns: {list(df.columns)}")
        target_col = numeric_cols[-1]

    return smiles_col, target_col


def mol_from_smiles(smiles):
    if pd.isna(smiles):
        return None
    try:
        return Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None


def validate_dataset(csv_path: Path):
    df = pd.read_csv(csv_path)
    smiles_col, target_col = detect_columns(df)

    records = []
    valid_smiles = []
    canonical_smiles = []
    mol_weights = []
    logp_values = []
    heavy_atoms = []
    ring_counts = []

    for smiles in df[smiles_col]:
        mol = mol_from_smiles(smiles)

        if mol is None:
            valid_smiles.append(False)
            canonical_smiles.append(None)
            mol_weights.append(None)
            logp_values.append(None)
            heavy_atoms.append(None)
            ring_counts.append(None)
        else:
            valid_smiles.append(True)
            canonical_smiles.append(Chem.MolToSmiles(mol, canonical=True))
            mol_weights.append(Descriptors.MolWt(mol))
            logp_values.append(Descriptors.MolLogP(mol))
            heavy_atoms.append(mol.GetNumHeavyAtoms())
            ring_counts.append(Descriptors.RingCount(mol))

    df_qc = df.copy()
    df_qc["valid_smiles"] = valid_smiles
    df_qc["canonical_smiles"] = canonical_smiles
    df_qc["mol_weight"] = mol_weights
    df_qc["logp"] = logp_values
    df_qc["heavy_atoms"] = heavy_atoms
    df_qc["ring_count"] = ring_counts

    valid_df = df_qc[df_qc["valid_smiles"]].copy()

    target = df_qc[target_col]

    summary = {
        "dataset": csv_path.stem,
        "rows": len(df_qc),
        "smiles_column": smiles_col,
        "target_column": target_col,
        "missing_smiles": int(df_qc[smiles_col].isna().sum()),
        "invalid_smiles": int((~df_qc["valid_smiles"]).sum()),
        "valid_smiles": int(df_qc["valid_smiles"].sum()),
        "missing_targets": int(target.isna().sum()),
        "duplicate_original_smiles": int(df_qc[smiles_col].duplicated().sum()),
        "duplicate_canonical_smiles": int(valid_df["canonical_smiles"].duplicated().sum()),
        "unique_canonical_smiles": int(valid_df["canonical_smiles"].nunique()),
        "target_unique_values": int(target.nunique(dropna=True)),
        "target_min": float(target.min()) if pd.api.types.is_numeric_dtype(target) else None,
        "target_max": float(target.max()) if pd.api.types.is_numeric_dtype(target) else None,
        "mol_weight_mean": float(valid_df["mol_weight"].mean()) if len(valid_df) else None,
        "mol_weight_median": float(valid_df["mol_weight"].median()) if len(valid_df) else None,
        "logp_mean": float(valid_df["logp"].mean()) if len(valid_df) else None,
        "heavy_atoms_mean": float(valid_df["heavy_atoms"].mean()) if len(valid_df) else None,
        "ring_count_mean": float(valid_df["ring_count"].mean()) if len(valid_df) else None,
    }

    if target.nunique(dropna=True) <= 10:
        value_counts = target.value_counts(dropna=False).to_dict()
        summary["target_value_counts"] = str(value_counts)
    else:
        summary["target_mean"] = float(target.mean())
        summary["target_std"] = float(target.std())

    return df_qc, summary
