import argparse
from pathlib import Path

import joblib
import pandas as pd
from clearml import Task
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

from trust_admet.data.featurize import dataframe_to_fingerprints
from trust_admet.utils.metrics import classification_metrics, regression_metrics
from trust_admet.utils.calibration import calibration_metrics


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


def build_model(model_name, task_type, seed):
    if task_type == "classification":
        if model_name == "random_forest":
            return RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                class_weight="balanced",
                n_jobs=-1,
                random_state=seed,
            )

        if model_name == "xgboost":
            return XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                tree_method="hist",
                random_state=seed,
            )

    if task_type == "regression":
        if model_name == "random_forest":
            return RandomForestRegressor(
                n_estimators=500,
                max_depth=None,
                n_jobs=-1,
                random_state=seed,
            )

        if model_name == "xgboost":
            return XGBRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                tree_method="hist",
                random_state=seed,
            )

    raise ValueError(f"Unsupported model/task combination: {model_name}, {task_type}")


def evaluate_model(model, task_type, x, y, split_name):
    if task_type == "classification":
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(x)[:, 1]
        else:
            y_prob = model.predict(x)
        metrics = classification_metrics(y, y_prob)
    else:
        y_pred = model.predict(x)
        metrics = regression_metrics(y, y_pred)

    return {f"{split_name}_{k}": v for k, v in metrics.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="scaffold", choices=["random", "scaffold"])
    parser.add_argument("--model", default="random_forest", choices=["random_forest", "xgboost"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_bits", type=int, default=2048)
    parser.add_argument("--radius", type=int, default=2)
    args = parser.parse_args()

    if args.dataset in CLASSIFICATION_DATASETS:
        task_type = "classification"
    elif args.dataset in REGRESSION_DATASETS:
        task_type = "regression"
    else:
        raise ValueError(f"Unknown dataset type for {args.dataset}")

    task = Task.init(
        project_name="TRUST-ADMET/Baselines",
        task_name=f"{args.model}_{args.dataset}_{args.split}_seed{args.seed}",
    )
    task.connect(vars(args))

    train_df, valid_df, test_df = load_split(args.dataset, args.split)

    x_train = dataframe_to_fingerprints(train_df, n_bits=args.n_bits, radius=args.radius)
    x_valid = dataframe_to_fingerprints(valid_df, n_bits=args.n_bits, radius=args.radius)
    x_test = dataframe_to_fingerprints(test_df, n_bits=args.n_bits, radius=args.radius)

    y_train = train_df["Y"].values
    y_valid = valid_df["Y"].values
    y_test = test_df["Y"].values

    model = build_model(args.model, task_type, args.seed)
    model.fit(x_train, y_train)

    metrics = {}
    metrics.update(evaluate_model(model, task_type, x_train, y_train, "train"))
    metrics.update(evaluate_model(model, task_type, x_valid, y_valid, "valid"))
    metrics.update(evaluate_model(model, task_type, x_test, y_test, "test"))

    for name, value in metrics.items():
        if value is not None:
            task.get_logger().report_scalar(
                title="metrics",
                series=name,
                value=float(value),
                iteration=0,
            )

    output_dir = Path("outputs/models") / args.dataset / args.split / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"seed{args.seed}.joblib"
    joblib.dump(model, model_path)

    if task_type == "classification":
        pred_rows = []
        for split_name, split_df, x, y in [
            ("train", train_df, x_train, y_train),
            ("valid", valid_df, x_valid, y_valid),
            ("test", test_df, x_test, y_test),
        ]:
            y_prob = model.predict_proba(x)[:, 1]
            cal = calibration_metrics(y, y_prob)

            for k, v in cal.items():
                metrics[f"{split_name}_{k}"] = v

            tmp = split_df[["canonical_smiles", "Y"]].copy()
            tmp["split"] = split_name
            tmp["y_prob"] = y_prob
            tmp["model"] = args.model
            tmp["dataset"] = args.dataset
            tmp["split_method"] = args.split
            tmp["seed"] = args.seed
            pred_rows.append(tmp)

        pred_df = pd.concat(pred_rows, ignore_index=True)
        pred_path = output_dir / f"seed{args.seed}_predictions.csv"
        pred_df.to_csv(pred_path, index=False)
        task.upload_artifact("predictions", str(pred_path))

    metrics_path = output_dir / f"seed{args.seed}_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    task.upload_artifact("model", str(model_path))
    task.upload_artifact("metrics", str(metrics_path))

    print(pd.DataFrame([metrics]).to_string(index=False))
    task.close()


if __name__ == "__main__":
    main()
