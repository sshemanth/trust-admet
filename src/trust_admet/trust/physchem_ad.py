from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors


def compute_physchem_descriptors(smiles: str):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    return {
        "mol_weight": Descriptors.MolWt(mol),
        "logp": Crippen.MolLogP(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "hbd": rdMolDescriptors.CalcNumHBD(mol),
        "hba": rdMolDescriptors.CalcNumHBA(mol),
    }


def build_physchem_ad_reference(dataset: str, split: str):
    train_path = Path("data/splits") / dataset / split / "train.csv"
    train_df = pd.read_csv(train_path)

    rows = []
    for smiles in train_df["canonical_smiles"]:
        rows.append(compute_physchem_descriptors(smiles))

    desc = pd.DataFrame(rows)

    limits = []
    for col in desc.columns:
        limits.append({
            "descriptor": col,
            "min": desc[col].quantile(0.01),
            "max": desc[col].quantile(0.99),
            "mean": desc[col].mean(),
            "std": desc[col].std(),
        })

    out = pd.DataFrame(limits)

    out_dir = Path("outputs/reports/applicability_domain/physchem")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{dataset}_{split}_physchem_limits.csv"
    out.to_csv(out_path, index=False)

    return out


def load_or_build_physchem_limits(dataset: str, split: str):
    path = Path("outputs/reports/applicability_domain/physchem") / f"{dataset}_{split}_physchem_limits.csv"

    if path.exists():
        return pd.read_csv(path)

    return build_physchem_ad_reference(dataset, split)


def check_physchem_ad(smiles: str, dataset: str, split: str):
    descriptors = compute_physchem_descriptors(smiles)
    limits = load_or_build_physchem_limits(dataset, split)

    violations = []

    for _, row in limits.iterrows():
        name = row["descriptor"]
        value = descriptors[name]

        if value < row["min"] or value > row["max"]:
            violations.append({
                "descriptor": name,
                "value": value,
                "min": row["min"],
                "max": row["max"],
            })

    return {
        "descriptors": descriptors,
        "inside_physchem_ad": len(violations) == 0,
        "violations": violations,
    }
