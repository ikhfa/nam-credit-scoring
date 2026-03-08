"""End-to-end pipeline: download -> preprocess -> train -> evaluate -> figures."""

import os
import sys
import json
import random
import numpy as np
import torch
import yaml

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.download import download_dataset
from src.data.split import create_splits, save_splits
from src.models.nam_trainer import NAMTrainer
from src.models.xgboost_baseline import (
    train_xgboost,
    hyperparameter_search_xgboost,
    extract_shap_values,
)
from src.conformal.wrapper import (
    NAMSklearnWrapper,
    create_conformal_classifier,
    predict_with_confidence,
)
from src.conformal.calibration import (
    compute_all_calibration_metrics,
    conformal_coverage_analysis,
)
from src.evaluation.metrics import compute_all_metrics, bootstrap_metrics
from src.evaluation.statistical_tests import delong_test, mcnemar_test
from src.evaluation.comparison import create_comparison_table, to_latex
from src.visualization.paper_figures import generate_all_figures
from src.visualization.shape_functions import get_feature_importance_from_nam


def set_global_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(config_path: str = "configs/default.yaml", skip_search: bool = False):
    """Run the full pipeline.

    Args:
        config_path: Path to YAML config file.
        skip_search: If True, skip hyperparameter search and use config defaults.
    """
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    seed = config["seed"]
    set_global_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}, seed: {seed}")

    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    # ========== Step 1: Data ==========
    print("\n" + "=" * 60)
    print("STEP 1: Data Pipeline")
    print("=" * 60)

    df = download_dataset(config["data"]["raw_path"])
    splits = create_splits(
        df,
        train_ratio=config["data"]["train_ratio"],
        cal_ratio=config["data"]["cal_ratio"],
        test_ratio=config["data"]["test_ratio"],
        seed=seed,
    )
    save_splits(splits, config["data"]["processed_dir"])

    X_train = splits["X_train"].values
    X_cal = splits["X_cal"].values
    X_test = splits["X_test"].values
    y_train = splits["y_train"]
    y_cal = splits["y_cal"]
    y_test = splits["y_test"]
    feature_names = splits["feature_names"]

    # ========== Step 2: Train NAM ==========
    print("\n" + "=" * 60)
    print("STEP 2: Train NAM")
    print("=" * 60)

    trainer = NAMTrainer(device=device)

    if skip_search:
        nam_config = config["nam"]
    else:
        print("Running NAM hyperparameter search...")
        nam_config, search_results = trainer.hyperparameter_search(
            X_train, y_train, n_trials=50, n_folds=5, seed=seed
        )
        # Save search results
        with open("results/logs/nam_search_results.json", "w") as f:
            json.dump(
                [
                    {"config": r["config"], "mean_auc": r["mean_auc"]}
                    for r in search_results
                ],
                f,
                indent=2,
                default=str,
            )

    print(f"\nTraining final NAM with config: {nam_config}")
    nam_model, nam_history = trainer.train_model(
        X_train, y_train, X_cal, y_cal, nam_config, verbose=True
    )

    # NAM predictions
    nam_model.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test).to(device)
        nam_logits, _ = nam_model(X_test_t)
        y_prob_nam = torch.sigmoid(nam_logits).cpu().numpy().ravel()

    # ========== Step 3: Train XGBoost ==========
    print("\n" + "=" * 60)
    print("STEP 3: Train XGBoost Baseline")
    print("=" * 60)

    if skip_search:
        xgb_params = config["xgboost"]
    else:
        print("Running XGBoost hyperparameter search...")
        xgb_params, best_xgb_auc = hyperparameter_search_xgboost(
            X_train, y_train, n_trials=50, n_folds=5, seed=seed
        )

    xgb_model = train_xgboost(X_train, y_train, X_cal, y_cal, xgb_params, seed)
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

    # SHAP values
    print("Extracting SHAP values...")
    shap_data = extract_shap_values(xgb_model, X_test, feature_names)

    # ========== Step 4: Conformal Prediction ==========
    print("\n" + "=" * 60)
    print("STEP 4: Conformal Prediction")
    print("=" * 60)

    alpha_levels = config["conformal"]["alpha_levels"]

    # NAM conformal
    nam_wrapper = NAMSklearnWrapper(nam_model, device=device)
    nam_wrapper.fit(X_cal, y_cal)
    nam_conformal = create_conformal_classifier(
        nam_wrapper, X_cal, y_cal, random_state=seed, alpha_levels=alpha_levels
    )
    nam_conf_results = predict_with_confidence(nam_conformal, X_test, alpha_levels)

    # XGBoost conformal
    xgb_conformal = create_conformal_classifier(
        xgb_model, X_cal, y_cal, random_state=seed, alpha_levels=alpha_levels
    )
    xgb_conf_results = predict_with_confidence(xgb_conformal, X_test, alpha_levels)

    # ========== Step 5: Evaluation ==========
    print("\n" + "=" * 60)
    print("STEP 5: Evaluation")
    print("=" * 60)

    # Point metrics
    nam_metrics = compute_all_metrics(y_test, y_prob_nam)
    xgb_metrics = compute_all_metrics(y_test, y_prob_xgb)

    print(f"\nNAM Metrics: {json.dumps(nam_metrics, indent=2)}")
    print(f"\nXGBoost Metrics: {json.dumps(xgb_metrics, indent=2)}")

    # Bootstrap CIs
    print("\nComputing bootstrap confidence intervals...")
    nam_bootstrap = bootstrap_metrics(
        y_test, y_prob_nam, n_bootstrap=config["evaluation"]["n_bootstrap"], seed=seed
    )
    xgb_bootstrap = bootstrap_metrics(
        y_test, y_prob_xgb, n_bootstrap=config["evaluation"]["n_bootstrap"], seed=seed
    )

    # Statistical tests
    delong = delong_test(y_test, y_prob_nam, y_prob_xgb)
    print(
        f"\nDeLong Test: AUC_NAM={delong['auc_a']:.4f}, AUC_XGB={delong['auc_b']:.4f}, "
        f"diff={delong['auc_diff']:.4f}, p={delong['p_value']:.4f}"
    )

    threshold_nam = nam_metrics["threshold"]
    threshold_xgb = xgb_metrics["threshold"]
    mcnemar = mcnemar_test(
        y_test,
        (y_prob_nam >= threshold_nam).astype(int),
        (y_prob_xgb >= threshold_xgb).astype(int),
    )
    print(f"McNemar Test: chi2={mcnemar['chi2']:.4f}, p={mcnemar['p_value']:.4f}")

    # Calibration
    nam_cal = compute_all_calibration_metrics(y_test, y_prob_nam)
    xgb_cal = compute_all_calibration_metrics(y_test, y_prob_xgb)

    # Conformal calibrated probabilities (from MAPIE)
    nam_conf_probs = nam_wrapper.predict_proba(X_test)[:, 1]
    xgb_conf_probs = xgb_model.predict_proba(X_test)[:, 1]
    nam_conf_cal = compute_all_calibration_metrics(y_test, nam_conf_probs)
    xgb_conf_cal = compute_all_calibration_metrics(y_test, xgb_conf_probs)

    # Conformal coverage
    nam_coverage = conformal_coverage_analysis(
        y_test, nam_conf_results["y_sets"], alpha_levels
    )
    xgb_coverage = conformal_coverage_analysis(
        y_test, xgb_conf_results["y_sets"], alpha_levels
    )

    print("\nNAM Conformal Coverage:")
    for c in nam_coverage:
        print(
            f"  α={c['alpha']}: coverage={c['empirical_coverage']:.4f} "
            f"(nominal={c['nominal_coverage']:.2f}), avg_set_size={c['avg_set_size']:.4f}"
        )

    print("\nXGBoost Conformal Coverage:")
    for c in xgb_coverage:
        print(
            f"  α={c['alpha']}: coverage={c['empirical_coverage']:.4f} "
            f"(nominal={c['nominal_coverage']:.2f}), avg_set_size={c['avg_set_size']:.4f}"
        )

    # ========== Step 6: Comparison Table ==========
    print("\n" + "=" * 60)
    print("STEP 6: Comparison Table")
    print("=" * 60)

    table = create_comparison_table(
        nam_metrics,
        xgb_metrics,
        nam_bootstrap,
        xgb_bootstrap,
        nam_coverage,
        xgb_coverage,
    )
    print("\n" + table.to_string(index=False))

    latex = to_latex(table, caption="NAM vs XGBoost Comparison", label="tab:comparison")
    with open("results/tables/comparison.tex", "w") as f:
        f.write(latex)

    # ========== Step 7: Figures ==========
    print("\n" + "=" * 60)
    print("STEP 7: Generate Figures")
    print("=" * 60)

    generate_all_figures(
        nam_model=nam_model,
        xgb_model=xgb_model,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        y_prob_nam=y_prob_nam,
        y_prob_xgb=y_prob_xgb,
        shap_data=shap_data,
        nam_reliability=nam_cal["reliability_data"],
        xgb_reliability=xgb_cal["reliability_data"],
        nam_conformal_reliability=nam_conf_cal["reliability_data"],
        xgb_conformal_reliability=xgb_conf_cal["reliability_data"],
        feature_names=feature_names,
    )

    # ========== Save all results ==========
    all_results = {
        "nam_metrics": nam_metrics,
        "xgb_metrics": xgb_metrics,
        "delong_test": delong,
        "mcnemar_test": mcnemar,
        "nam_calibration": {
            k: v for k, v in nam_cal.items() if k != "reliability_data"
        },
        "xgb_calibration": {
            k: v for k, v in xgb_cal.items() if k != "reliability_data"
        },
        "nam_conformal_coverage": nam_coverage,
        "xgb_conformal_coverage": xgb_coverage,
        "nam_config": nam_config if isinstance(nam_config, dict) else dict(nam_config),
    }
    with open("results/logs/results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print("Results saved to results/logs/results.json")
    print("Figures saved to results/figures/")
    print("Tables saved to results/tables/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--skip-search",
        action="store_true",
        help="Skip hyperparameter search, use config defaults",
    )
    args = parser.parse_args()
    main(args.config, args.skip_search)
