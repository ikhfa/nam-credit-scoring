"""Generate comparison tables (LaTeX) for NAM vs XGBoost."""

import numpy as np
import pandas as pd


def format_metric_with_ci(bootstrap_result: dict) -> str:
    """Format a metric as 'mean (lower, upper)'."""
    m = bootstrap_result
    return f"{m['mean']:.4f} ({m['lower']:.4f}, {m['upper']:.4f})"


def create_comparison_table(
    nam_metrics: dict,
    xgb_metrics: dict,
    nam_bootstrap: dict = None,
    xgb_bootstrap: dict = None,
    nam_conformal: list[dict] = None,
    xgb_conformal: list[dict] = None,
) -> pd.DataFrame:
    """Create a comparison table of NAM vs XGBoost results.

    Args:
        nam_metrics: Dict from compute_all_metrics for NAM.
        xgb_metrics: Dict from compute_all_metrics for XGBoost.
        nam_bootstrap: Optional bootstrap CIs for NAM.
        xgb_bootstrap: Optional bootstrap CIs for XGBoost.
        nam_conformal: Optional conformal analysis for NAM.
        xgb_conformal: Optional conformal analysis for XGBoost.

    Returns:
        DataFrame with comparison.
    """
    rows = []

    # Basic metrics
    metric_names = {
        "auc_roc": "AUC-ROC",
        "auc_pr": "AUC-PR",
        "brier_score": "Brier Score",
        "log_loss": "Log Loss",
        "ece": "ECE",
        "f1": "F1 Score",
    }

    for key, display_name in metric_names.items():
        nam_val = nam_metrics[key]
        xgb_val = xgb_metrics[key]

        if nam_bootstrap and xgb_bootstrap:
            nam_str = format_metric_with_ci(nam_bootstrap[key])
            xgb_str = format_metric_with_ci(xgb_bootstrap[key])
        else:
            nam_str = f"{nam_val:.4f}"
            xgb_str = f"{xgb_val:.4f}"

        rows.append({
            "Metric": display_name,
            "NAM": nam_str,
            "XGBoost": xgb_str,
        })

    # Conformal metrics
    if nam_conformal and xgb_conformal:
        for nc, xc in zip(nam_conformal, xgb_conformal):
            alpha = nc["alpha"]
            rows.append({
                "Metric": f"Coverage @{100*(1-alpha):.0f}%",
                "NAM": f"{nc['empirical_coverage']:.4f}",
                "XGBoost": f"{xc['empirical_coverage']:.4f}",
            })
            rows.append({
                "Metric": f"Avg Set Size @{100*(1-alpha):.0f}%",
                "NAM": f"{nc['avg_set_size']:.4f}",
                "XGBoost": f"{xc['avg_set_size']:.4f}",
            })

    return pd.DataFrame(rows)


def to_latex(df: pd.DataFrame, caption: str = "", label: str = "") -> str:
    """Convert comparison DataFrame to LaTeX table."""
    latex = df.to_latex(index=False, escape=False, column_format="lcc")

    if caption:
        latex = latex.replace(
            "\\begin{tabular}",
            f"\\begin{{table}}[htbp]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\begin{{tabular}}"
        )
        latex = latex.replace("\\end{tabular}", "\\end{tabular}\n\\end{table}")

    return latex
