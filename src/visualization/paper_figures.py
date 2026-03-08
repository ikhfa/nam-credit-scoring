"""Generate all publication-ready figures and tables."""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import pandas as pd
from sklearn.metrics import roc_curve

from src.visualization.shape_functions import (
    plot_shape_functions_grid,
    get_feature_importance_from_nam,
)
from src.visualization.shap_plots import plot_nam_vs_shap_comparison
from src.visualization.calibration_plots import (
    plot_reliability_comparison,
    plot_conformal_set_sizes,
)

# Publication style
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob_nam: np.ndarray,
    y_prob_xgb: np.ndarray,
    auc_nam: float,
    auc_xgb: float,
    save_path: str = None,
) -> plt.Figure:
    """Plot ROC curves for NAM and XGBoost."""
    fig, ax = plt.subplots(figsize=(6, 5))

    fpr_nam, tpr_nam, _ = roc_curve(y_true, y_prob_nam)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_true, y_prob_xgb)

    ax.plot(fpr_nam, tpr_nam, color="#2196F3", linewidth=2,
            label=f"NAM (AUC = {auc_nam:.4f})")
    ax.plot(fpr_xgb, tpr_xgb, color="#FF5722", linewidth=2,
            label=f"XGBoost (AUC = {auc_xgb:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    if save_path:
        fig.savefig(save_path)
    return fig


def plot_feature_importance_comparison(
    nam_importance: np.ndarray,
    shap_importance: np.ndarray,
    feature_names: list[str],
    top_k: int = 15,
    save_path: str = None,
) -> plt.Figure:
    """Compare feature importance between NAM and XGBoost (SHAP)."""
    # Normalize to [0, 1]
    nam_norm = nam_importance / nam_importance.max()
    shap_norm = shap_importance / shap_importance.max()

    # Sort by NAM importance
    order = np.argsort(nam_norm)[::-1][:top_k]

    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(top_k)
    width = 0.35

    ax.barh(y_pos - width/2, nam_norm[order], width, label="NAM (variance)",
            color="#2196F3", alpha=0.8)
    ax.barh(y_pos + width/2, shap_norm[order], width, label="XGBoost (SHAP)",
            color="#FF5722", alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in order])
    ax.invert_yaxis()
    ax.set_xlabel("Normalized Importance")
    ax.set_title("Feature Importance Comparison")
    ax.legend()

    if save_path:
        fig.savefig(save_path)
    return fig


def generate_all_figures(
    nam_model,
    xgb_model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_prob_nam: np.ndarray,
    y_prob_xgb: np.ndarray,
    shap_data: dict,
    nam_reliability: dict,
    xgb_reliability: dict,
    nam_conformal_reliability: dict,
    xgb_conformal_reliability: dict,
    feature_names: list[str],
    output_dir: str = "results/figures",
) -> None:
    """Generate all paper figures and save to output directory."""
    os.makedirs(output_dir, exist_ok=True)

    # Feature importance for selecting top features
    nam_importance = get_feature_importance_from_nam(nam_model, X_train)
    top_features = np.argsort(nam_importance)[::-1][:6]

    # Fig 2: Shape functions grid
    plot_shape_functions_grid(
        nam_model, top_features.tolist(), X_train, feature_names,
        ncols=3, save_path=os.path.join(output_dir, "shape_functions_grid.pdf"),
    )

    # Fig 3: NAM vs SHAP comparison
    top_3 = top_features[:3].tolist()
    plot_nam_vs_shap_comparison(
        nam_model, shap_data, top_3, X_train, feature_names,
        save_path=os.path.join(output_dir, "nam_vs_shap_comparison.pdf"),
    )

    # Fig 4: ROC curves
    from sklearn.metrics import roc_auc_score
    auc_nam = roc_auc_score(y_test, y_prob_nam)
    auc_xgb = roc_auc_score(y_test, y_prob_xgb)
    plot_roc_curves(
        y_test, y_prob_nam, y_prob_xgb, auc_nam, auc_xgb,
        save_path=os.path.join(output_dir, "roc_curves.pdf"),
    )

    # Fig 5: Reliability diagrams
    plot_reliability_comparison(
        {
            "NAM": nam_reliability,
            "NAM + Conformal": nam_conformal_reliability,
            "XGBoost": xgb_reliability,
            "XGBoost + Conformal": xgb_conformal_reliability,
        },
        save_path=os.path.join(output_dir, "calibration_plots.pdf"),
    )

    # Fig 8: Feature importance comparison
    shap_importance = np.abs(shap_data["shap_values"]).mean(axis=0)
    plot_feature_importance_comparison(
        nam_importance, shap_importance, feature_names,
        save_path=os.path.join(output_dir, "feature_importance_comparison.pdf"),
    )

    plt.close("all")
    print(f"All figures saved to {output_dir}/")
