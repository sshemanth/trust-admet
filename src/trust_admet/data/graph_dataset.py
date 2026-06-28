import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from rdkit import Chem


ATOM_LIST = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "H"]


def atom_features(atom):
    symbol = atom.GetSymbol()
    features = [
        int(symbol == atom_symbol)
        for atom_symbol in ATOM_LIST
    ]
    features += [
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetIsAromatic()),
        atom.GetTotalNumHs(),
    ]
    return features


def smiles_to_graph(smiles, y):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None

    x = torch.tensor(
        [atom_features(atom) for atom in mol.GetAtoms()],
        dtype=torch.float,
    )

    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i])

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    y = torch.tensor([y], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, y=y)


class MoleculeGraphDataset:
    def __init__(self, df, smiles_col="canonical_smiles", target_col="Y"):
        self.graphs = []
        for _, row in df.iterrows():
            graph = smiles_to_graph(row[smiles_col], row[target_col])
            if graph is not None:
                self.graphs.append(graph)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


def make_graph_loader(dataset, batch_size=64, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
