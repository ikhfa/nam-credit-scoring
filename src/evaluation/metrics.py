"""Evaluation metrics for credit risk models."""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    f1_score,
    roc_curve,
)

from src.conformal.calibration import expected_calibration_error


def optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find optimal classification threshold using Youden's J statistic."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    return float(thresholds[np.argmax(j_scores)])


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = None,
    ece_bins: int = 10,
) -> dict:
    """Compute all evaluation metrics.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities for positive class.
        threshold: Classification threshold. If None, uses optimal threshold.
        ece_bins: Number of bins for ECE.

    Returns:
        Dict with all metric values.
    """
    if threshold is None:
        threshold = optimal_threshold(y_true, y_prob)

    y_pred = (y_prob >= threshold).astype(int)
    ece, _ = expected_calibration_error(y_true, y_prob, ece_bins)

    return {
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "auc_pr": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "ece": float(ece),
        "f1": float(f1_score(y_true, y_pred)),
        "threshold": float(threshold),
    }


def bootstrap_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict:
    """Compute metrics with bootstrap confidence intervals.

    Returns:
        Dict mapping metric name to {mean, lower, upper, std}.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    all_metrics = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        # Ensure both classes are present
        if len(np.unique(y_true[idx])) < 2:
            continue
        m = compute_all_metrics(y_true[idx], y_prob[idx])
        all_metrics.append(m)

    alpha = (1 - confidence) / 2
    results = {}
    for key in all_metrics[0]:
        values = [m[key] for m in all_metrics]
        results[key] = {
            "mean": float(np.mean(values)),
            "lower": float(np.percentile(values, 100 * alpha)),
            "upper": float(np.percentile(values, 100 * (1 - alpha))),
            "std": float(np.std(values)),
        }

    return results
