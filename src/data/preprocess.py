"""Preprocess the Taiwan Credit Card Default dataset."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# Feature groups
CONTINUOUS_FEATURES = [
    "LIMIT_BAL", "AGE",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]

CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"]

ORDINAL_FEATURES = ["PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

TARGET_COL = "default"

ALL_FEATURES = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES + ORDINAL_FEATURES
# Sorted order for consistent NAM sub-network assignment
FEATURE_ORDER = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning steps to raw dataset.

    - Drop ID column if present
    - Rename PAY_0 -> PAY_1
    - Merge undocumented EDUCATION values (0, 5, 6) into 4 (Other)
    - Merge undocumented MARRIAGE value 0 into 3 (Other)
    - Rename target column
    """
    df = df.copy()

    # Drop ID column (various possible names)
    for col in ["ID", "id"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Rename PAY_0 -> PAY_1 (known naming issue)
    if "PAY_0" in df.columns:
        df = df.rename(columns={"PAY_0": "PAY_1"})

    # Fix EDUCATION: merge undocumented values into Other (4)
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})

    # Fix MARRIAGE: merge undocumented value 0 into Other (3)
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    # Rename target column
    target_candidates = ["default.payment.next.month", "default payment next month", "Y"]
    for col in target_candidates:
        if col in df.columns:
            df = df.rename(columns={col: TARGET_COL})
            break

    # Verify no missing values
    assert df.isnull().sum().sum() == 0, "Unexpected missing values found!"

    # Reorder columns
    df = df[FEATURE_ORDER + [TARGET_COL]]

    return df


def scale_features(
    X_train: pd.DataFrame,
    X_cal: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit StandardScaler on train, transform all splits. Only scales continuous features."""
    scaler = StandardScaler()

    X_train = X_train.copy()
    X_cal = X_cal.copy()
    X_test = X_test.copy()

    scaler.fit(X_train[CONTINUOUS_FEATURES])

    X_train[CONTINUOUS_FEATURES] = scaler.transform(X_train[CONTINUOUS_FEATURES])
    X_cal[CONTINUOUS_FEATURES] = scaler.transform(X_cal[CONTINUOUS_FEATURES])
    X_test[CONTINUOUS_FEATURES] = scaler.transform(X_test[CONTINUOUS_FEATURES])

    return X_train, X_cal, X_test, scaler


def compute_class_weight(y: np.ndarray) -> float:
    """Compute positive class weight for imbalanced binary classification.

    Returns weight = n_negative / n_positive.
    """
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    return n_neg / n_pos
