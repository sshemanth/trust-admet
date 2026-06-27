from pathlib import Path
import random
from collections import defaultdict

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def get_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def remove_duplicate_molecules(df: pd.DataFrame) -> pd.DataFrame:
    if "canonical_smiles" not in df.columns:
        raise ValueError("canonical_smiles column missing. Run validation first.")
    return df.drop_duplicates(subset=["canonical_smiles"]).reset_index(drop=True)


def random_split(df, seed=42, train_frac=0.8, valid_frac=0.1):
    rng = random.Random(seed)
    indices = list(df.index)
    rng.shuffle(indices)

    n = len(indices)
    n_train = int(n * train_frac)
    n_valid = int(n * valid_frac)

    train_idx = indices[:n_train]
    valid_idx = indices[n_train:n_train + n_valid]
    test_idx = indices[n_train + n_valid:]

    return (
        df.loc[train_idx].reset_index(drop=True),
        df.loc[valid_idx].reset_index(drop=True),
        df.loc[test_idx].reset_index(drop=True),
    )


def scaffold_split(df, seed=42, train_frac=0.8, valid_frac=0.1):
    df = df.copy()
    df["scaffold"] = df["canonical_smiles"].apply(get_scaffold)

    scaffold_to_indices = defaultdict(list)
    for idx, scaffold in zip(df.index, df["scaffold"]):
        scaffold_to_indices[scaffold].append(idx)

    scaffold_sets = list(scaffold_to_indices.values())

    rng = random.Random(seed)
    rng.shuffle(scaffold_sets)

    scaffold_sets = sorted(scaffold_sets, key=len, reverse=True)

    n_total = len(df)
    train_cutoff = train_frac * n_total
    valid_cutoff = (train_frac + valid_frac) * n_total

    train_idx, valid_idx, test_idx = [], [], []

    for scaffold_set in scaffold_sets:
        if len(train_idx) + len(scaffold_set) <= train_cutoff:
            train_idx.extend(scaffold_set)
        elif len(train_idx) + len(valid_idx) + len(scaffold_set) <= valid_cutoff:
            valid_idx.extend(scaffold_set)
        else:
            test_idx.extend(scaffold_set)

    return (
        df.loc[train_idx].reset_index(drop=True),
        df.loc[valid_idx].reset_index(drop=True),
        df.loc[test_idx].reset_index(drop=True),
    )


def summarize_split(dataset, split_name, train_df, valid_df, test_df, target_col="Y"):
    rows = []

    for name, split_df in [
        ("train", train_df),
        ("valid", valid_df),
        ("test", test_df),
    ]:
        row = {
            "dataset": dataset,
            "split_method": split_name,
            "split": name,
            "rows": len(split_df),
            "unique_molecules": split_df["canonical_smiles"].nunique(),
        }

        if "scaffold" in split_df.columns:
            row["unique_scaffolds"] = split_df["scaffold"].nunique()
        else:
            row["unique_scaffolds"] = None

        if target_col in split_df.columns:
            target = split_df[target_col]
            if target.nunique(dropna=True) <= 10:
                row["target_counts"] = str(target.value_counts(dropna=False).to_dict())
                row["target_mean"] = None
                row["target_std"] = None
            else:
                row["target_counts"] = None
                row["target_mean"] = float(target.mean())
                row["target_std"] = float(target.std())

        rows.append(row)

    return rows
