"""Statistical tests for model comparison."""

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def delong_test(
    y_true: np.ndarray, y_prob_a: np.ndarray, y_prob_b: np.ndarray
) -> dict:
    """DeLong test for comparing two AUC-ROC values.

    Tests H0: AUC_A = AUC_B.

    Args:
        y_true: True binary labels.
        y_prob_a: Predicted probabilities from model A.
        y_prob_b: Predicted probabilities from model B.

    Returns:
        Dict with auc_a, auc_b, z_statistic, p_value.
    """
    auc_a = roc_auc_score(y_true, y_prob_a)
    auc_b = roc_auc_score(y_true, y_prob_b)

    # DeLong's method for variance estimation
    n1 = y_true.sum()  # positives
    n0 = len(y_true) - n1  # negatives

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    # Placement values for model A
    v_a_pos = np.array([
        np.mean(y_prob_a[pos_idx[i]] > y_prob_a[neg_idx])
        + 0.5 * np.mean(y_prob_a[pos_idx[i]] == y_prob_a[neg_idx])
        for i in range(len(pos_idx))
    ])
    v_a_neg = np.array([
        np.mean(y_prob_a[pos_idx] > y_prob_a[neg_idx[j]])
        + 0.5 * np.mean(y_prob_a[pos_idx] == y_prob_a[neg_idx[j]])
        for j in range(len(neg_idx))
    ])

    # Placement values for model B
    v_b_pos = np.array([
        np.mean(y_prob_b[pos_idx[i]] > y_prob_b[neg_idx])
        + 0.5 * np.mean(y_prob_b[pos_idx[i]] == y_prob_b[neg_idx])
        for i in range(len(pos_idx))
    ])
    v_b_neg = np.array([
        np.mean(y_prob_b[pos_idx] > y_prob_b[neg_idx[j]])
        + 0.5 * np.mean(y_prob_b[pos_idx] == y_prob_b[neg_idx[j]])
        for j in range(len(neg_idx))
    ])

    # Covariance matrix of (AUC_A, AUC_B)
    s10 = np.cov(np.stack([v_a_pos, v_b_pos]))[0, 1] if len(pos_idx) > 1 else 0
    s01 = np.cov(np.stack([v_a_neg, v_b_neg]))[0, 1] if len(neg_idx) > 1 else 0

    var_a = np.var(v_a_pos, ddof=1) / n1 + np.var(v_a_neg, ddof=1) / n0
    var_b = np.var(v_b_pos, ddof=1) / n1 + np.var(v_b_neg, ddof=1) / n0
    cov_ab = s10 / n1 + s01 / n0

    var_diff = var_a + var_b - 2 * cov_ab
    if var_diff <= 0:
        z = 0.0
    else:
        z = (auc_a - auc_b) / np.sqrt(var_diff)

    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return {
        "auc_a": float(auc_a),
        "auc_b": float(auc_b),
        "auc_diff": float(auc_a - auc_b),
        "z_statistic": float(z),
        "p_value": float(p_value),
    }


def mcnemar_test(
    y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray
) -> dict:
    """McNemar's test for comparing two classifiers.

    Tests whether the models make different types of errors.

    Returns:
        Dict with contingency table values, chi2 statistic, and p_value.
    """
    # Contingency table
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # b: A correct, B wrong; c: A wrong, B correct
    b = np.sum(correct_a & ~correct_b)
    c = np.sum(~correct_a & correct_b)

    # McNemar with continuity correction
    if b + c == 0:
        chi2 = 0.0
        p_value = 1.0
    else:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - stats.chi2.cdf(chi2, df=1)

    return {
        "b_a_correct_b_wrong": int(b),
        "c_a_wrong_b_correct": int(c),
        "chi2": float(chi2),
        "p_value": float(p_value),
    }
