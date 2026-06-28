from pathlib import Path
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from clearml import Task

from trust_admet.data.torch_dataset import FingerprintDataset
from trust_admet.models.mlp import FingerprintMLP
from trust_admet.trust.uncertainty import mc_dropout_predict_mlp


CLASSIFICATION_DATASETS = {"BBBP", "ClinTox", "Tox21_NR-AR", "Tox21_SR-p53"}
REGRESSION_DATASETS = {"Solubility", "Lipophilicity"}


def load_split(dataset, split_method):
    base = Path("data/splits") / dataset / split_method
    return pd.read_csv(base / "test.csv")


def infer_task_type(dataset):
    if dataset in CLASSIFICATION_DATASETS:
        return "classification"
    if dataset in REGRESSION_DATASETS:
        return "regression"
    raise ValueError(f"Unknown dataset: {dataset}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="scaffold")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    args = parser.parse_args()

    task_type = infer_task_type(args.dataset)

    task = Task.init(
        project_name="TRUST-ADMET/Trust",
        task_name=f"mlp_uncertainty_{args.dataset}_{args.split}_seed{args.seed}",
    )
    task.connect(vars(args))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_df = load_split(args.dataset, args.split)
    test_ds = FingerprintDataset(test_df, task_type)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = FingerprintMLP(
        input_dim=2048,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        task_type=task_type,
    ).to(device)

    model_path = Path("outputs/models") / args.dataset / args.split / "mlp" / f"seed{args.seed}" / "best_model.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))

    unc = mc_dropout_predict_mlp(
        model=model,
        loader=test_loader,
        device=device,
        n_samples=args.n_samples,
    )

    out = test_df[["canonical_smiles", "Y"]].copy()
    out["mc_mean"] = unc["mean"]
    out["mc_variance"] = unc["variance"]
    out["mc_std"] = unc["std"]
    out["dataset"] = args.dataset
    out["split"] = args.split
    out["model"] = "mlp"
    out["seed"] = args.seed

    out_dir = Path("outputs/reports/uncertainty")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"mlp_{args.dataset}_{args.split}_seed{args.seed}_uncertainty.csv"
    out.to_csv(out_path, index=False)

    task.upload_artifact("uncertainty_predictions", str(out_path))

    print(out[["canonical_smiles", "Y", "mc_mean", "mc_std"]].head())
    task.close()


if __name__ == "__main__":
    main()
