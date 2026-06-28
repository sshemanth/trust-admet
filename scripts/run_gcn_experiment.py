import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from clearml import Task

from trust_admet.data.graph_dataset import MoleculeGraphDataset, make_graph_loader
from trust_admet.models.gnn import GCNModel, GINModel
from trust_admet.utils.metrics import classification_metrics, regression_metrics
from trust_admet.utils.calibration import calibration_metrics


CLASSIFICATION_DATASETS = {"BBBP", "ClinTox", "Tox21_NR-AR", "Tox21_SR-p53"}
REGRESSION_DATASETS = {"Solubility", "Lipophilicity"}


def load_split(dataset, split_method):
    base = Path("data/splits") / dataset / split_method
    return (
        pd.read_csv(base / "train.csv"),
        pd.read_csv(base / "valid.csv"),
        pd.read_csv(base / "test.csv"),
    )


def evaluate(model, loader, task_type, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            pred = model.predict_from_logits(logits)

            y_true.extend(batch.y.view(-1).cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if task_type == "classification":
        metrics = classification_metrics(y_true, y_pred)
        metrics.update(calibration_metrics(y_true, y_pred))
    else:
        metrics = regression_metrics(y_true, y_pred)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="scaffold", choices=["random", "scaffold"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--model", default="gcn", choices=["gcn", "gin"])
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.dataset in CLASSIFICATION_DATASETS:
        task_type = "classification"
    elif args.dataset in REGRESSION_DATASETS:
        task_type = "regression"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    task = Task.init(
        project_name="TRUST-ADMET/GNN",
        task_name=f"{args.model}_{args.dataset}_{args.split}_seed{args.seed}",
    )
    task.connect(vars(args))

    train_df, valid_df, test_df = load_split(args.dataset, args.split)

    train_ds = MoleculeGraphDataset(train_df)
    valid_ds = MoleculeGraphDataset(valid_df)
    test_ds = MoleculeGraphDataset(test_df)

    train_loader = make_graph_loader(train_ds, args.batch_size, shuffle=True)
    valid_loader = make_graph_loader(valid_ds, args.batch_size, shuffle=False)
    test_loader = make_graph_loader(test_ds, args.batch_size, shuffle=False)

    input_dim = train_ds[0].x.shape[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_cls = GCNModel if args.model == "gcn" else GINModel

    model = model_cls(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        task_type=task_type,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    output_dir = Path("outputs/models") / args.dataset / args.split / args.model / f"seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_model.pt"

    best_valid = None
    bad_epochs = 0
    patience = 15
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = model.loss_fn(logits, batch.y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        valid_metrics = evaluate(model, valid_loader, task_type, device)

        if task_type == "classification":
            monitor = valid_metrics["auroc"]
            higher_is_better = True
        else:
            monitor = valid_metrics["rmse"]
            higher_is_better = False

        task.get_logger().report_scalar("loss", "train_loss", float(np.mean(losses)), epoch)

        for k, v in valid_metrics.items():
            if v is not None:
                task.get_logger().report_scalar("valid_metrics", k, float(v), epoch)

        improved = (
            best_valid is None
            or (higher_is_better and monitor > best_valid)
            or ((not higher_is_better) and monitor < best_valid)
        )

        if improved:
            best_valid = monitor
            best_epoch = epoch
            bad_epochs = 0
            torch.save(model.state_dict(), best_path)
        else:
            bad_epochs += 1

        print(f"Epoch {epoch}: loss={np.mean(losses):.4f}, monitor={monitor:.4f}, best={best_valid:.4f}")

        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(torch.load(best_path, map_location=device))

    metrics = {}
    for split_name, loader in [("train", train_loader), ("valid", valid_loader), ("test", test_loader)]:
        split_metrics = evaluate(model, loader, task_type, device)
        for k, v in split_metrics.items():
            metrics[f"{split_name}_{k}"] = v

    metrics["best_epoch"] = best_epoch

    metrics_path = output_dir / "metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    task.upload_artifact("best_model", str(best_path))
    task.upload_artifact("metrics", str(metrics_path))

    print(pd.DataFrame([metrics]).to_string(index=False))
    task.close()


if __name__ == "__main__":
    main()
