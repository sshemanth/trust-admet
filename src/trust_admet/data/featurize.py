import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def smiles_to_morgan_fp(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius,
        nBits=n_bits,
    )

    arr = np.zeros((n_bits,), dtype=np.float32)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def dataframe_to_fingerprints(df, smiles_col="canonical_smiles", radius=2, n_bits=2048):
    features = [
        smiles_to_morgan_fp(smiles, radius=radius, n_bits=n_bits)
        for smiles in df[smiles_col]
    ]
    return np.stack(features).astype(np.float32)
