import numpy as np
from sklearn.metrics import brier_score_loss, log_loss


def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lower = bins[i]
        upper = bins[i + 1]

        mask = (y_prob > lower) & (y_prob <= upper)
        if not np.any(mask):
            continue

        bin_confidence = np.mean(y_prob[mask])
        bin_accuracy = np.mean(y_true[mask])
        bin_weight = np.mean(mask)

        ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return float(ece)


def calibration_metrics(y_true, y_prob, n_bins=10):
    y_prob = np.clip(np.asarray(y_prob), 1e-7, 1 - 1e-7)

    return {
        "brier": brier_score_loss(y_true, y_prob),
        "nll": log_loss(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob, n_bins=n_bins),
    }
