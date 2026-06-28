from pathlib import Path
import argparse

import pandas as pd
import torch
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import Draw

from trust_admet.data.graph_dataset import MoleculeGraphDataset
from trust_admet.models.gnn import GINModel


CLASSIFICATION = {"BBBP", "ClinTox", "Tox21_NR-AR", "Tox21_SR-p53"}
REGRESSION = {"Solubility", "Lipophilicity"}


def infer_task_type(dataset):
    if dataset in CLASSIFICATION:
        return "classification"
    if dataset in REGRESSION:
        return "regression"
    raise ValueError(dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="BBBP")
    parser.add_argument("--split", default="scaffold")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    task_type = infer_task_type(args.dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_df = pd.read_csv(Path("data/splits") / args.dataset / args.split / "test.csv")
    graph_ds = MoleculeGraphDataset(test_df)

    data = graph_ds[args.index].to(device)

    input_dim = data.x.shape[1]

    model = GINModel(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=3,
        dropout=0.2,
        task_type=task_type,
    ).to(device)

    model_path = Path("outputs/models") / args.dataset / args.split / "gin" / f"seed{args.seed}" / "best_model.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    if task_type == "classification":
        mode = "binary_classification"
        return_type = "probs"
    else:
        mode = "regression"
        return_type = "raw"

    class ExplainerWrapper(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

        def forward(self, x, edge_index, batch=None):
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            data_obj = Data(x=x, edge_index=edge_index, batch=batch)
            out = self.base_model(data_obj)
            if task_type == "classification":
                return self.base_model.predict_from_logits(out)
            return out

    wrapped_model = ExplainerWrapper(model).to(device)
    wrapped_model.eval()

    explainer = Explainer(
        model=wrapped_model,
        algorithm=GNNExplainer(epochs=args.epochs),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode=mode,
            task_level="graph",
            return_type=return_type,
        ),
    )

    explanation = explainer(data.x, data.edge_index, batch=torch.zeros(data.x.size(0), dtype=torch.long, device=device))

    node_scores = explanation.node_mask.detach().cpu().abs().sum(dim=1)
    node_scores = node_scores / (node_scores.max() + 1e-8)

    smiles = test_df.iloc[args.index]["canonical_smiles"]
    mol = Chem.MolFromSmiles(smiles)

    atom_weights = {
        i: float(node_scores[i])
        for i in range(min(len(node_scores), mol.GetNumAtoms()))
    }

    out_dir = Path("outputs/reports/explainability") / args.dataset / args.split / "gin"
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "atom_index": list(atom_weights.keys()),
        "importance": list(atom_weights.values()),
    }).to_csv(out_dir / f"seed{args.seed}_idx{args.index}_gnnexplainer_atoms.csv", index=False)

    drawer = Draw.MolDraw2DCairo(700, 500)
    opts = drawer.drawOptions()

    highlight_atoms = list(atom_weights.keys())
    highlight_atom_colors = {
        i: (1.0, 1.0 - atom_weights[i], 1.0 - atom_weights[i])
        for i in highlight_atoms
    }

    Draw.rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_atom_colors,
    )
    drawer.FinishDrawing()

    fig_path = out_dir / f"seed{args.seed}_idx{args.index}_gnnexplainer.png"
    with open(fig_path, "wb") as f:
        f.write(drawer.GetDrawingText())

    paper_dir = Path("paper/figures")
    paper_dir.mkdir(parents=True, exist_ok=True)
    paper_path = paper_dir / "Figure6_GIN_GNNExplainer.png"
    with open(paper_path, "wb") as f:
        f.write(drawer.GetDrawingText())

    print(f"Saved atom importance and molecule figure to {out_dir}")
    print(f"Paper figure: {paper_path}")


if __name__ == "__main__":
    main()
