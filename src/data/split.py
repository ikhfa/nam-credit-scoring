"""Train/calibration/test splitting for the credit default dataset."""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.data.preprocess import (
    clean_dataframe,
    scale_features,
    FEATURE_ORDER,
    TARGET_COL,
)


def create_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.60,
    cal_ratio: float = 0.15,
    test_ratio: float = 0.25,
    seed: int = 42,
) -> dict:
    """Create stratified train/calibration/test splits.

    Args:
        df: Raw DataFrame (will be cleaned internally).
        train_ratio: Fraction for training.
        cal_ratio: Fraction for conformal calibration.
        test_ratio: Fraction for testing.
        seed: Random seed.

    Returns:
        Dictionary with keys: X_train, X_cal, X_test, y_train, y_cal, y_test,
        scaler, and feature_names.
    """
    assert abs(train_ratio + cal_ratio + test_ratio - 1.0) < 1e-6

    df = clean_dataframe(df)
    X = df[FEATURE_ORDER]
    y = df[TARGET_COL].values

    # First split: train vs (cal + test)
    holdout_ratio = cal_ratio + test_ratio
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=holdout_ratio, stratify=y, random_state=seed
    )

    # Second split: cal vs test
    cal_fraction_of_holdout = cal_ratio / holdout_ratio
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_holdout, y_holdout,
        test_size=(1 - cal_fraction_of_holdout),
        stratify=y_holdout,
        random_state=seed,
    )

    # Scale continuous features (fit on train only)
    X_train, X_cal, X_test, scaler = scale_features(X_train, X_cal, X_test)

    print(f"Split sizes — Train: {len(X_train)}, Cal: {len(X_cal)}, Test: {len(X_test)}")
    print(f"Default rates — Train: {y_train.mean():.3f}, Cal: {y_cal.mean():.3f}, Test: {y_test.mean():.3f}")

    return {
        "X_train": X_train,
        "X_cal": X_cal,
        "X_test": X_test,
        "y_train": y_train,
        "y_cal": y_cal,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": FEATURE_ORDER,
    }


def save_splits(splits: dict, output_dir: str = "data/processed") -> None:
    """Save processed splits to CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    for name in ["train", "cal", "test"]:
        X = splits[f"X_{name}"]
        y = splits[f"y_{name}"]
        df = X.copy()
        df[TARGET_COL] = y
        df.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)

    print(f"Splits saved to {output_dir}/")


def load_splits(input_dir: str = "data/processed") -> dict:
    """Load processed splits from CSV files."""
    splits = {}
    for name in ["train", "cal", "test"]:
        df = pd.read_csv(os.path.join(input_dir, f"{name}.csv"))
        splits[f"X_{name}"] = df.drop(columns=[TARGET_COL])
        splits[f"y_{name}"] = df[TARGET_COL].values

    splits["feature_names"] = list(splits["X_train"].columns)
    return splits
