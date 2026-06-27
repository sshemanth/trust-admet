import argparse
from pathlib import Path

import pandas as pd
import torch
from clearml import Task

from trust_admet.data.torch_dataset import FingerprintDataset
from trust_admet.models.mlp import FingerprintMLP
from trust_admet.training.trainer import Trainer


CLASSIFICATION_DATASETS = {
    "BBBP",
    "ClinTox",
    "Tox21_NR-AR",
    "Tox21_SR-p53",
}

REGRESSION_DATASETS = {
    "Solubility",
    "Lipophilicity",
}


def load_split(dataset, split_method):
    base = Path("data/splits") / dataset / split_method
    return (
        pd.read_csv(base / "train.csv"),
        pd.read_csv(base / "valid.csv"),
        pd.read_csv(base / "test.csv"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="scaffold", choices=["random", "scaffold"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--n_bits", type=int, default=2048)
    parser.add_argument("--radius", type=int, default=2)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.dataset in CLASSIFICATION_DATASETS:
        task_type = "classification"
    elif args.dataset in REGRESSION_DATASETS:
        task_type = "regression"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    task = Task.init(
        project_name="TRUST-ADMET/Neural",
        task_name=f"mlp_{args.dataset}_{args.split}_seed{args.seed}",
    )
    task.connect(vars(args))

    train_df, valid_df, test_df = load_split(args.dataset, args.split)

    train_ds = FingerprintDataset(train_df, task_type, args.n_bits, args.radius)
    valid_ds = FingerprintDataset(valid_df, task_type, args.n_bits, args.radius)
    test_ds = FingerprintDataset(test_df, task_type, args.n_bits, args.radius)

    model = FingerprintMLP(
        input_dim=args.n_bits,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        task_type=task_type,
    )

    output_dir = Path("outputs/models") / args.dataset / args.split / "mlp" / f"seed{args.seed}"

    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        valid_dataset=valid_ds,
        test_dataset=test_ds,
        task_type=task_type,
        output_dir=output_dir,
        clearml_task=task,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        patience=10,
    )

    metrics = trainer.fit()

    metrics_path = output_dir / "metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    task.upload_artifact("metrics", str(metrics_path))

    print(pd.DataFrame([metrics]).to_string(index=False))
    task.close()


if __name__ == "__main__":
    main()
