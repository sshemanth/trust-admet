import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from clearml import Task
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)

from trust_admet.utils.metrics import classification_metrics, regression_metrics
from trust_admet.utils.calibration import calibration_metrics


CLASSIFICATION_DATASETS = {"BBBP", "ClinTox", "Tox21_NR-AR", "Tox21_SR-p53"}
REGRESSION_DATASETS = {"Solubility", "Lipophilicity"}


class SmilesDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, task_type, max_length=256):
        self.smiles = df["canonical_smiles"].astype(str).tolist()
        self.labels = df["Y"].astype(float).tolist()
        self.tokenizer = tokenizer
        self.task_type = task_type
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.smiles[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}

        if self.task_type == "classification":
            item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        else:
            item["labels"] = torch.tensor(float(self.labels[idx]), dtype=torch.float)

        return item


def load_split(dataset, split_method):
    base = Path("data/splits") / dataset / split_method
    return (
        pd.read_csv(base / "train.csv"),
        pd.read_csv(base / "valid.csv"),
        pd.read_csv(base / "test.csv"),
    )


def compute_metrics_builder(task_type):
    def compute_metrics(eval_pred):
        preds, labels = eval_pred

        if task_type == "classification":
            probs = torch.softmax(torch.tensor(preds), dim=1).numpy()[:, 1]
            metrics = classification_metrics(labels, probs)
            metrics.update(calibration_metrics(labels, probs))
            return {k: float(v) for k, v in metrics.items() if v is not None}

        preds = np.asarray(preds).reshape(-1)
        metrics = regression_metrics(labels, preds)
        return {k: float(v) for k, v in metrics.items() if v is not None}

    return compute_metrics


def predict_and_save(hf_trainer, dataset, df, split_name, task_type, output_dir, model_name, dataset_name, split_method, seed):
    pred = hf_trainer.predict(dataset)
    raw = pred.predictions

    out = df[["canonical_smiles", "Y"]].copy()
    out["split"] = split_name
    out["model"] = model_name
    out["dataset"] = dataset_name
    out["split_method"] = split_method
    out["seed"] = seed

    if task_type == "classification":
        probs = torch.softmax(torch.tensor(raw), dim=1).numpy()[:, 1]
        out["y_prob"] = probs
    else:
        out["y_pred"] = np.asarray(raw).reshape(-1)

    path = output_dir / f"{split_name}_predictions.csv"
    out.to_csv(path, index=False)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="scaffold", choices=["random", "scaffold"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", default="DeepChem/ChemBERTa-77M-MTR")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()

    set_seed(args.seed)

    if args.dataset in CLASSIFICATION_DATASETS:
        task_type = "classification"
        num_labels = 2
        problem_type = "single_label_classification"
        metric_for_best_model = "eval_auroc"
        greater_is_better = True
    elif args.dataset in REGRESSION_DATASETS:
        task_type = "regression"
        num_labels = 1
        problem_type = "regression"
        metric_for_best_model = "eval_rmse"
        greater_is_better = False
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    task = Task.init(
        project_name="TRUST-ADMET/Foundation",
        task_name=f"chemberta_{args.dataset}_{args.split}_seed{args.seed}",
    )
    task.connect(vars(args))

    train_df, valid_df, test_df = load_split(args.dataset, args.split)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        problem_type=problem_type,
        ignore_mismatched_sizes=True,
    )

    train_ds = SmilesDataset(train_df, tokenizer, task_type, args.max_length)
    valid_ds = SmilesDataset(valid_df, tokenizer, task_type, args.max_length)
    test_ds = SmilesDataset(test_df, tokenizer, task_type, args.max_length)

    output_dir = Path("outputs/models") / args.dataset / args.split / "chemberta" / f"seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "hf_checkpoints"),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        save_total_limit=2,
        report_to=[],
        fp16=torch.cuda.is_available(),
    )

    hf_trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics_builder(task_type),
    )

    hf_trainer.train()

    metrics = {}

    for split_name, ds in [
        ("train", train_ds),
        ("valid", valid_ds),
        ("test", test_ds),
    ]:
        result = hf_trainer.evaluate(ds)
        for k, v in result.items():
            clean_k = k.replace("eval_", "")
            metrics[f"{split_name}_{clean_k}"] = v

    metrics_path = output_dir / "metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    pred_paths = [
        predict_and_save(hf_trainer, train_ds, train_df, "train", task_type, output_dir, "chemberta", args.dataset, args.split, args.seed),
        predict_and_save(hf_trainer, valid_ds, valid_df, "valid", task_type, output_dir, "chemberta", args.dataset, args.split, args.seed),
        predict_and_save(hf_trainer, test_ds, test_df, "test", task_type, output_dir, "chemberta", args.dataset, args.split, args.seed),
    ]

    model.save_pretrained(output_dir / "best_model")
    tokenizer.save_pretrained(output_dir / "best_model")

    task.upload_artifact("metrics", str(metrics_path))
    for p in pred_paths:
        task.upload_artifact(p.stem, str(p))

    print(pd.DataFrame([metrics]).to_string(index=False))
    task.close()


if __name__ == "__main__":
    main()
