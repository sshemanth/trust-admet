from pathlib import Path

import numpy as np
import torch
from clearml import Task
from torch.utils.data import DataLoader

from trust_admet.utils.metrics import classification_metrics, regression_metrics
from trust_admet.utils.calibration import calibration_metrics


class Trainer:
    def __init__(
        self,
        model,
        train_dataset,
        valid_dataset,
        test_dataset,
        task_type,
        output_dir,
        clearml_task: Task,
        batch_size=64,
        lr=1e-3,
        epochs=50,
        patience=10,
        device=None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.task_type = task_type
        self.output_dir = Path(output_dir)
        self.task = clearml_task
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    def _loader(self, dataset, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=True,
        )

    def train_one_epoch(self):
        self.model.train()
        losses = []

        for x, y in self._loader(self.train_dataset, shuffle=True):
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.model.loss_fn(logits, y)
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

        return float(np.mean(losses))

    @torch.no_grad()
    def predict_dataset(self, dataset):
        self.model.eval()

        y_true = []
        y_pred = []

        for x, y in self._loader(dataset, shuffle=False):
            x = x.to(self.device)

            pred = self.model.predict(x).detach().cpu().numpy()

            y_pred.extend(pred.tolist())
            y_true.extend(y.numpy().tolist())

        return np.asarray(y_true), np.asarray(y_pred)

    def evaluate(self, dataset, split_name):
        y_true, y_pred = self.predict_dataset(dataset)

        if self.task_type == "classification":
            metrics = classification_metrics(y_true, y_pred)
            metrics.update(calibration_metrics(y_true, y_pred))
        else:
            metrics = regression_metrics(y_true, y_pred)

        return {f"{split_name}_{k}": v for k, v in metrics.items()}

    def fit(self):
        best_valid = None
        best_epoch = 0
        bad_epochs = 0
        best_path = self.output_dir / "best_model.pt"

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_one_epoch()
            valid_metrics = self.evaluate(self.valid_dataset, "valid")

            if self.task_type == "classification":
                monitor = valid_metrics.get("valid_auroc")
                higher_is_better = True
            else:
                monitor = valid_metrics.get("valid_rmse")
                higher_is_better = False

            self.task.get_logger().report_scalar(
                title="loss",
                series="train_loss",
                value=train_loss,
                iteration=epoch,
            )

            for k, v in valid_metrics.items():
                if v is not None:
                    self.task.get_logger().report_scalar(
                        title="valid_metrics",
                        series=k,
                        value=float(v),
                        iteration=epoch,
                    )

            improved = False
            if best_valid is None:
                improved = True
            elif higher_is_better and monitor > best_valid:
                improved = True
            elif not higher_is_better and monitor < best_valid:
                improved = True

            if improved:
                best_valid = monitor
                best_epoch = epoch
                bad_epochs = 0
                torch.save(self.model.state_dict(), best_path)
            else:
                bad_epochs += 1

            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                f"monitor={monitor:.4f}, best={best_valid:.4f}"
            )

            if bad_epochs >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                break

        self.model.load_state_dict(torch.load(best_path, map_location=self.device))

        metrics = {}
        metrics.update(self.evaluate(self.train_dataset, "train"))
        metrics.update(self.evaluate(self.valid_dataset, "valid"))
        metrics.update(self.evaluate(self.test_dataset, "test"))
        metrics["best_epoch"] = best_epoch

        self.task.upload_artifact("best_model", str(best_path))

        return metrics
