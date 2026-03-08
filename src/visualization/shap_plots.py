"""SHAP visualization utilities for XGBoost baseline."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import shap


def plot_shap_summary(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    save_path: str = None,
    max_display: int = 15,
) -> plt.Figure:
    """Generate SHAP summary (beeswarm) plot.

    Args:
        shap_values: SHAP values array (n_samples, n_features).
        X: Feature matrix.
        feature_names: Feature names.
        save_path: Optional save path.
        max_display: Max features to show.

    Returns:
        Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(
        shap_values, X,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return plt.gcf()


def plot_shap_dependence(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_idx: int,
    feature_name: str = None,
    ax: plt.Axes = None,
    color: str = "#FF5722",
) -> plt.Axes:
    """Plot SHAP dependence for a single feature (scatter plot).

    This creates a 'pseudo shape function' comparable to NAM shape functions.
    Key difference: SHAP values show scatter due to interaction effects.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.5))

    ax.scatter(
        X[:, feature_idx],
        shap_values[:, feature_idx],
        alpha=0.3, s=5, color=color, rasterized=True,
    )
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.set_xlabel(feature_name or f"Feature {feature_idx}")
    ax.set_ylabel("SHAP value")
    ax.set_title(feature_name or f"Feature {feature_idx}")

    return ax


def plot_nam_vs_shap_comparison(
    nam_model,
    shap_data: dict,
    feature_indices: list[int],
    X_train: np.ndarray,
    feature_names: list[str],
    save_path: str = None,
) -> plt.Figure:
    """Side-by-side comparison of NAM shape functions vs SHAP dependence plots.

    Args:
        nam_model: Trained NAM model.
        shap_data: Dict from extract_shap_values.
        feature_indices: Features to compare.
        X_train: Training data for NAM rug plots.
        feature_names: Feature names.
        save_path: Optional save path.

    Returns:
        Figure.
    """
    import torch
    from src.visualization.shape_functions import plot_shape_function

    n = len(feature_indices)
    fig, axes = plt.subplots(n, 2, figsize=(10, 3.5 * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    for i, feat_idx in enumerate(feature_indices):
        # NAM shape function (left)
        plot_shape_function(
            nam_model, feat_idx, X_train[:, feat_idx],
            feature_name=f"NAM: {feature_names[feat_idx]}",
            ax=axes[i, 0],
        )

        # SHAP dependence (right)
        plot_shap_dependence(
            shap_data["shap_values"], shap_data["X"],
            feat_idx,
            feature_name=f"SHAP: {feature_names[feat_idx]}",
            ax=axes[i, 1],
        )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
