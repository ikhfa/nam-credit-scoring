"""Calibration metrics: ECE, MCE, Brier score, conformal coverage analysis."""

import numpy as np
from sklearn.metrics import brier_score_loss


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> tuple[float, dict]:
    """Compute Expected Calibration Error (ECE).

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities for positive class.
        n_bins: Number of equal-width bins.

    Returns:
        ECE value and dict with per-bin details for reliability diagrams.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_details = []

    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:  # include right edge for last bin
            mask = mask | (y_prob == bin_edges[i + 1])

        n_in_bin = mask.sum()
        if n_in_bin == 0:
            bin_details.append({
                "bin_lower": bin_edges[i],
                "bin_upper": bin_edges[i + 1],
                "avg_confidence": 0,
                "avg_accuracy": 0,
                "count": 0,
            })
            continue

        avg_confidence = y_prob[mask].mean()
        avg_accuracy = y_true[mask].mean()
        bin_weight = n_in_bin / len(y_true)

        ece += bin_weight * abs(avg_accuracy - avg_confidence)

        bin_details.append({
            "bin_lower": bin_edges[i],
            "bin_upper": bin_edges[i + 1],
            "avg_confidence": float(avg_confidence),
            "avg_accuracy": float(avg_accuracy),
            "count": int(n_in_bin),
        })

    return float(ece), {"bins": bin_details, "n_bins": n_bins}


def maximum_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Compute Maximum Calibration Error (MCE)."""
    _, details = expected_calibration_error(y_true, y_prob, n_bins)
    mce = 0.0
    for b in details["bins"]:
        if b["count"] > 0:
            mce = max(mce, abs(b["avg_accuracy"] - b["avg_confidence"]))
    return mce


def conformal_coverage_analysis(
    y_true: np.ndarray,
    y_sets: np.ndarray,
    alpha_levels: list[float],
) -> list[dict]:
    """Analyze conformal prediction set coverage and efficiency.

    Args:
        y_true: True labels (n_samples,).
        y_sets: Prediction sets from MAPIE (n_samples, n_classes, n_alpha).
        alpha_levels: Significance levels.

    Returns:
        List of dicts with coverage, avg_set_size, and other stats per alpha.
    """
    results = []
    for i, alpha in enumerate(alpha_levels):
        sets_at_alpha = y_sets[:, :, i]  # (n_samples, n_classes)

        # Coverage: true label is in the prediction set
        coverage = np.mean([
            sets_at_alpha[j, y_true[j]] for j in range(len(y_true))
        ])

        # Average set size
        set_sizes = sets_at_alpha.sum(axis=1)
        avg_set_size = set_sizes.mean()

        # Fraction of empty sets, singletons, and full sets
        empty_frac = (set_sizes == 0).mean()
        singleton_frac = (set_sizes == 1).mean()
        full_frac = (set_sizes == 2).mean()

        results.append({
            "alpha": alpha,
            "nominal_coverage": 1 - alpha,
            "empirical_coverage": float(coverage),
            "avg_set_size": float(avg_set_size),
            "empty_fraction": float(empty_frac),
            "singleton_fraction": float(singleton_frac),
            "full_set_fraction": float(full_frac),
        })

    return results


def compute_all_calibration_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> dict:
    """Compute all calibration-related metrics.

    Returns dict with ece, mce, brier_score, and reliability diagram data.
    """
    ece, reliability_data = expected_calibration_error(y_true, y_prob, n_bins)
    mce = maximum_calibration_error(y_true, y_prob, n_bins)
    brier = brier_score_loss(y_true, y_prob)

    return {
        "ece": ece,
        "mce": mce,
        "brier_score": brier,
        "reliability_data": reliability_data,
    }
