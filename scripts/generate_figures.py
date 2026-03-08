"""Generate all paper figures from saved models and results."""

import os
import sys
import json
import numpy as np
import torch
import yaml
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.split import load_splits
from src.models.nam import NAM
from src.conformal.calibration import compute_all_calibration_metrics
from src.visualization.paper_figures import generate_all_figures
from src.models.xgboost_baseline import extract_shap_values


def main(config_path: str = "configs/default.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    splits = load_splits(config["data"]["processed_dir"])
    X_train = splits["X_train"].values
    X_test = splits["X_test"].values
    y_test = splits["y_test"]
    feature_names = splits["feature_names"]

    # Load NAM
    with open("results/models/nam_config.json") as f:
        nam_config = json.load(f)

    nam_model = NAM(
        num_features=len(feature_names),
        hidden_sizes=nam_config.get("hidden_sizes", [64, 64, 64]),
        dropout=nam_config.get("dropout", 0.3),
    )
    nam_model.load_state_dict(torch.load("results/models/nam_best.pt", weights_only=True))
    nam_model.eval()

    # NAM predictions
    with torch.no_grad():
        logits, _ = nam_model(torch.FloatTensor(X_test))
        y_prob_nam = torch.sigmoid(logits).numpy().ravel()

    # Load XGBoost
    xgb_model = joblib.load("results/models/xgboost_best.joblib")
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

    # SHAP
    shap_data = extract_shap_values(xgb_model, X_test, feature_names)

    # Calibration data
    nam_cal = compute_all_calibration_metrics(y_test, y_prob_nam)
    xgb_cal = compute_all_calibration_metrics(y_test, y_prob_xgb)

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
        nam_conformal_reliability=nam_cal["reliability_data"],
        xgb_conformal_reliability=xgb_cal["reliability_data"],
        feature_names=feature_names,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
