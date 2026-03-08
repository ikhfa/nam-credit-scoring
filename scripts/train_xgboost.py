"""Standalone XGBoost training script."""

import os
import sys
import json
import numpy as np
import yaml
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.split import load_splits
from src.models.xgboost_baseline import (
    train_xgboost,
    hyperparameter_search_xgboost,
    extract_shap_values,
)


def main(config_path: str = "configs/default.yaml", skip_search: bool = False):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    seed = config["seed"]
    np.random.seed(seed)

    splits = load_splits(config["data"]["processed_dir"])
    X_train = splits["X_train"].values
    X_cal = splits["X_cal"].values
    y_train = splits["y_train"]
    y_cal = splits["y_cal"]
    X_test = splits["X_test"].values
    y_test = splits["y_test"]
    feature_names = splits["feature_names"]

    if skip_search:
        params = config["xgboost"]
    else:
        print("Running hyperparameter search...")
        params, best_auc = hyperparameter_search_xgboost(
            X_train, y_train, n_trials=50, n_folds=5, seed=seed
        )

    print(f"\nTraining final model with: {params}")
    model = train_xgboost(X_train, y_train, X_cal, y_cal, params, seed)

    # Evaluate
    from sklearn.metrics import roc_auc_score
    y_prob = model.predict_proba(X_test)[:, 1]
    print(f"Test AUC: {roc_auc_score(y_test, y_prob):.4f}")

    # Save
    os.makedirs("results/models", exist_ok=True)
    joblib.dump(model, "results/models/xgboost_best.joblib")
    with open("results/models/xgboost_params.json", "w") as f:
        json.dump(params, f, indent=2, default=str)

    # SHAP
    print("Extracting SHAP values...")
    shap_data = extract_shap_values(model, X_test, feature_names)
    np.save("results/models/shap_values.npy", shap_data["shap_values"])

    print("Model saved to results/models/xgboost_best.joblib")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--skip-search", action="store_true")
    args = parser.parse_args()
    main(args.config, args.skip_search)
