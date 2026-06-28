from pathlib import Path
import argparse

import pandas as pd
import torch
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt

from trust_admet.data.torch_dataset import FingerprintDataset
from trust_admet.models.mlp import FingerprintMLP


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
    parser.add_argument("--n_examples", type=int, default=50)
    args = parser.parse_args()

    task_type = infer_task_type(args.dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_df = pd.read_csv(Path("data/splits") / args.dataset / args.split / "test.csv")
    test_df = test_df.head(args.n_examples).copy()

    ds = FingerprintDataset(test_df, task_type)
    x = torch.stack([ds[i][0] for i in range(len(ds))]).to(device)

    model = FingerprintMLP(task_type=task_type).to(device)
    model_path = Path("outputs/models") / args.dataset / args.split / "mlp" / f"seed{args.seed}" / "best_model.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    def forward_func(inputs):
        out = model(inputs)
        if task_type == "classification":
            return torch.sigmoid(out)
        return out

    ig = IntegratedGradients(forward_func)
    baseline = torch.zeros_like(x)

    attributions = ig.attribute(x, baselines=baseline, n_steps=32)
    attr = attributions.detach().cpu().numpy()

    importance = abs(attr).mean(axis=0)
    top_idx = importance.argsort()[::-1][:50]

    out_dir = Path("outputs/reports/explainability") / args.dataset / args.split / "mlp"
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "fingerprint_bit": top_idx,
        "mean_abs_integrated_gradient": importance[top_idx],
    }).to_csv(out_dir / f"seed{args.seed}_mlp_integrated_gradients.csv", index=False)

    plt.figure(figsize=(8, 6))
    plt.barh([str(i) for i in top_idx[:25]][::-1], importance[top_idx[:25]][::-1])
    plt.xlabel("Mean absolute integrated gradient")
    plt.ylabel("Morgan fingerprint bit")
    plt.title(f"MLP Integrated Gradients: {args.dataset} {args.split}")
    plt.tight_layout()

    fig_path = out_dir / f"seed{args.seed}_mlp_integrated_gradients.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()

    paper_dir = Path("paper/figures")
    paper_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.barh([str(i) for i in top_idx[:25]][::-1], importance[top_idx[:25]][::-1])
    plt.xlabel("Mean absolute integrated gradient")
    plt.ylabel("Morgan fingerprint bit")
    plt.title(f"MLP Integrated Gradients: {args.dataset} {args.split}")
    plt.tight_layout()
    plt.savefig(paper_dir / "Figure6_MLP_IntegratedGradients.png", dpi=300)
    plt.close()

    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
