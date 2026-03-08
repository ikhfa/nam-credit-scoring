"""Calibration and reliability diagram visualizations."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def plot_reliability_diagram(
    reliability_data: dict,
    model_name: str = "Model",
    ax: plt.Axes = None,
    color: str = "#2196F3",
) -> plt.Axes:
    """Plot a reliability diagram from ECE bin data.

    Args:
        reliability_data: Output from expected_calibration_error.
        model_name: Label for the model.
        ax: Matplotlib axes.
        color: Bar/line color.

    Returns:
        Axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    bins = reliability_data["bins"]
    confidences = [b["avg_confidence"] for b in bins if b["count"] > 0]
    accuracies = [b["avg_accuracy"] for b in bins if b["count"] > 0]

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.plot(confidences, accuracies, "o-", color=color, label=model_name, markersize=6)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    ax.set_title("Reliability Diagram")

    return ax


def plot_reliability_comparison(
    reliability_dict: dict[str, dict],
    colors: dict[str, str] = None,
    save_path: str = None,
) -> plt.Figure:
    """Plot reliability diagrams for multiple models on the same axes.

    Args:
        reliability_dict: Map model_name -> reliability_data.
        colors: Map model_name -> color.
        save_path: Optional save path.

    Returns:
        Figure.
    """
    default_colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
    if colors is None:
        colors = {
            name: default_colors[i % len(default_colors)]
            for i, name in enumerate(reliability_dict)
        }

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")

    for name, data in reliability_dict.items():
        bins = data["bins"]
        confidences = [b["avg_confidence"] for b in bins if b["count"] > 0]
        accuracies = [b["avg_accuracy"] for b in bins if b["count"] > 0]
        ax.plot(
            confidences, accuracies, "o-",
            color=colors.get(name, "#333"),
            label=name, markersize=5,
        )

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    ax.set_title("Reliability Diagram Comparison")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_conformal_set_sizes(
    set_sizes: np.ndarray,
    alpha: float,
    model_name: str = "Model",
    ax: plt.Axes = None,
    save_path: str = None,
) -> plt.Axes:
    """Plot histogram of conformal prediction set sizes.

    Args:
        set_sizes: Array of prediction set sizes (0, 1, or 2 for binary).
        alpha: Significance level.
        model_name: Label.
        ax: Axes.
        save_path: Optional save path.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    counts = [
        (set_sizes == 0).sum(),
        (set_sizes == 1).sum(),
        (set_sizes == 2).sum(),
    ]
    labels = ["Empty ({})", "Singleton ({})", "Both classes ({})"]
    labels = [l.format(c) for l, c in zip(labels, counts)]

    ax.bar([0, 1, 2], counts, color=["#f44336", "#4CAF50", "#FF9800"], alpha=0.8)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Empty", "Singleton", "Both"])
    ax.set_ylabel("Count")
    ax.set_title(f"{model_name} — Prediction Set Sizes (α={alpha})")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return ax
