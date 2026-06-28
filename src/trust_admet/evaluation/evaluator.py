from trust_admet.utils.metrics import classification_metrics, regression_metrics
from trust_admet.utils.calibration import calibration_metrics


def evaluate_predictions(y_true, y_pred, task_type):
    if task_type == "classification":
        metrics = classification_metrics(y_true, y_pred)
        metrics.update(calibration_metrics(y_true, y_pred))
        return metrics

    if task_type == "regression":
        return regression_metrics(y_true, y_pred)

    raise ValueError(f"Unknown task_type: {task_type}")
