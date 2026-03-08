"""Visualization of NAM shape functions."""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def plot_shape_function(
    model,
    feature_idx: int,
    feature_values: np.ndarray,
    feature_name: str = None,
    ax: plt.Axes = None,
    show_rug: bool = True,
    color: str = "#2196F3",
) -> plt.Axes:
    """Plot a single NAM shape function.

    Args:
        model: Trained NAM model.
        feature_idx: Index of the feature.
        feature_values: Training data values for rug plot and range.
        feature_name: Display name for the feature.
        ax: Matplotlib axes. Created if None.
        show_rug: Whether to show rug plot of training data.
        color: Line color.

    Returns:
        Matplotlib axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.5))

    # Generate smooth curve
    x_min, x_max = feature_values.min(), feature_values.max()
    x_range = torch.linspace(x_min, x_max, 200)
    y_range = model.get_shape_function(feature_idx, x_range).numpy()

    ax.plot(x_range.numpy(), y_range, color=color, linewidth=2)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

    if show_rug:
        # Subsample for rug plot if too many points
        rug_vals = feature_values
        if len(rug_vals) > 500:
            rng = np.random.RandomState(42)
            rug_vals = rng.choice(rug_vals, 500, replace=False)
        ax.plot(
            rug_vals, np.full_like(rug_vals, ax.get_ylim()[0]),
            "|", color="gray", alpha=0.3, markersize=5,
        )

    ax.set_xlabel(feature_name or f"Feature {feature_idx}")
    ax.set_ylabel("f(x)")
    ax.set_title(feature_name or f"Feature {feature_idx}")

    return ax


def plot_shape_functions_grid(
    model,
    feature_indices: list[int],
    X_train: np.ndarray,
    feature_names: list[str],
    ncols: int = 3,
    figsize_per_plot: tuple = (4, 3),
    save_path: str = None,
) -> plt.Figure:
    """Plot a grid of NAM shape functions.

    Args:
        model: Trained NAM model.
        feature_indices: Which features to plot.
        X_train: Training data for rug plots.
        feature_names: Feature names.
        ncols: Columns in the grid.
        figsize_per_plot: Size per subplot.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib Figure.
    """
    n = len(feature_indices)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
    )
    axes = np.atleast_2d(axes)

    for i, feat_idx in enumerate(feature_indices):
        row, col = divmod(i, ncols)
        ax = axes[row, col]
        plot_shape_function(
            model,
            feat_idx,
            X_train[:, feat_idx],
            feature_name=feature_names[feat_idx],
            ax=ax,
        )

    # Hide unused subplots
    for i in range(n, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row, col].set_visible(False)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def get_feature_importance_from_nam(
    model, X_train: np.ndarray
) -> np.ndarray:
    """Compute feature importance as variance of shape function over training data.

    Args:
        model: Trained NAM model.
        X_train: Training features.

    Returns:
        Array of importance scores (one per feature).
    """
    model.eval()
    importances = []

    with torch.no_grad():
        X_t = torch.FloatTensor(X_train)
        for i in range(model.num_features):
            fi = model.feature_nns[i](X_t[:, i:i+1]).numpy().ravel()
            importances.append(np.var(fi))

    return np.array(importances)
