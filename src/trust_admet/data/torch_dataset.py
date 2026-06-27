import torch
from torch.utils.data import Dataset

from trust_admet.data.featurize import dataframe_to_fingerprints


class FingerprintDataset(Dataset):
    def __init__(self, df, task_type, n_bits=2048, radius=2):
        self.x = dataframe_to_fingerprints(
            df,
            smiles_col="canonical_smiles",
            n_bits=n_bits,
            radius=radius,
        )
        self.y = df["Y"].values
        self.task_type = task_type

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx], dtype=torch.float32)

        if self.task_type == "classification":
            y = torch.tensor(self.y[idx], dtype=torch.float32)
        else:
            y = torch.tensor(self.y[idx], dtype=torch.float32)

        return x, y
