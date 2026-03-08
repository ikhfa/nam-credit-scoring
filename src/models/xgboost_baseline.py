"""XGBoost baseline model with SHAP explanations."""

import numpy as np
import xgboost as xgb
import shap
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import optuna


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict = None,
    seed: int = 42,
) -> xgb.XGBClassifier:
    """Train an XGBoost classifier.

    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data for early stopping.
        params: XGBoost parameters. If None, uses defaults.
        seed: Random seed.

    Returns:
        Trained XGBClassifier.
    """
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos

    default_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 500,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": n_neg / n_pos,
        "random_state": seed,
        "early_stopping_rounds": 30,
        "verbosity": 0,
    }

    if params:
        default_params.update(params)

    model = xgb.XGBClassifier(**default_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    return model


def hyperparameter_search_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = 50,
    n_folds: int = 5,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[dict, float]:
    """Optuna-based hyperparameter search for XGBoost.

    Returns:
        Best params dict and best mean AUC.
    """
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos

    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "scale_pos_weight": n_neg / n_pos,
            "random_state": seed,
            "early_stopping_rounds": 30,
            "verbosity": 0,
        }

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_aucs = []

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_vl = X_train[train_idx], X_train[val_idx]
            y_tr, y_vl = y_train[train_idx], y_train[val_idx]

            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)

            y_pred = model.predict_proba(X_vl)[:, 1]
            fold_aucs.append(roc_auc_score(y_vl, y_pred))

        return np.mean(fold_aucs)

    optuna.logging.set_verbosity(
        optuna.logging.INFO if verbose else optuna.logging.WARNING
    )
    study = optuna.create_study(direction="maximize", seed=seed)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_params.update({
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "scale_pos_weight": n_neg / n_pos,
        "random_state": seed,
        "early_stopping_rounds": 30,
        "verbosity": 0,
    })

    if verbose:
        print(f"Best XGBoost AUC: {study.best_value:.4f}")
        print(f"Best params: {best_params}")

    return best_params, study.best_value


def extract_shap_values(
    model: xgb.XGBClassifier,
    X: np.ndarray,
    feature_names: list[str] = None,
) -> dict:
    """Extract SHAP values from a trained XGBoost model.

    Args:
        model: Trained XGBClassifier.
        X: Data to explain.
        feature_names: Optional feature names.

    Returns:
        Dict with shap_values, expected_value, and feature_names.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    return {
        "shap_values": shap_values,
        "expected_value": explainer.expected_value,
        "feature_names": feature_names,
        "X": X,
    }
