"""Conformal prediction wrapper using MAPIE for NAM and XGBoost."""

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from mapie.classification import SplitConformalClassifier

from src.models.nam import NAM


class NAMSklearnWrapper(BaseEstimator, ClassifierMixin):
    """Wraps a trained PyTorch NAM as a scikit-learn compatible classifier.

    Required for MAPIE integration. The model must already be trained.
    """

    def __init__(self, trained_model: NAM, device: str = "cpu"):
        self.trained_model = trained_model
        self.device = torch.device(device)
        self.trained_model.to(self.device)

    def fit(self, X, y=None):
        """No-op: model is already trained."""
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        """Predict class probabilities."""
        self.trained_model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(np.asarray(X)).to(self.device)
            logits, _ = self.trained_model(X_t)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
        return np.column_stack([1 - probs, probs])

    def predict(self, X):
        """Predict class labels."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def create_conformal_classifier(
    estimator,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    method: str = "lac",
    random_state: int = 42,
    alpha_levels: list[float] = None,
) -> SplitConformalClassifier:
    """Create and calibrate a MAPIE conformal classifier.

    Args:
        estimator: A fitted sklearn-compatible classifier (NAMSklearnWrapper or XGBClassifier).
        X_cal: Calibration features.
        y_cal: Calibration labels.
        method: Conformity score method (default: "lac").
        random_state: Random seed.
        alpha_levels: Significance levels (e.g., [0.05, 0.10, 0.20]).

    Returns:
        Calibrated SplitConformalClassifier.
    """
    if alpha_levels is None:
        alpha_levels = [0.05, 0.10, 0.20]

    confidence_levels = [1 - a for a in alpha_levels]

    mapie_clf = SplitConformalClassifier(
        estimator=estimator,
        conformity_score=method,
        confidence_level=confidence_levels,
        prefit=True,
        random_state=random_state,
    )
    mapie_clf.conformalize(X_cal, y_cal)
    return mapie_clf


def predict_with_confidence(
    mapie_clf: SplitConformalClassifier,
    X: np.ndarray,
    alpha_levels: list[float] = None,
) -> dict:
    """Make predictions with conformal prediction sets.

    Args:
        mapie_clf: Calibrated SplitConformalClassifier.
        X: Features to predict.
        alpha_levels: Significance levels (e.g., [0.05, 0.10, 0.20]).
            Note: alpha_levels are now set at classifier creation time.
            This parameter is kept for backward compatibility but ignored.

    Returns:
        Dict with predictions, prediction_sets, and alpha_levels.
    """
    if alpha_levels is None:
        alpha_levels = [0.05, 0.10, 0.20]

    y_pred, y_sets = mapie_clf.predict_set(X)

    return {
        "y_pred": y_pred,
        "y_sets": y_sets,  # shape: (n_samples, n_classes, n_confidence_levels)
        "alpha_levels": alpha_levels,
    }
