"""Tests for data pipeline."""

import pytest
import numpy as np
import pandas as pd

from src.data.preprocess import (
    clean_dataframe,
    FEATURE_ORDER,
    TARGET_COL,
    CONTINUOUS_FEATURES,
    CATEGORICAL_FEATURES,
    ORDINAL_FEATURES,
)
from src.data.split import create_splits


@pytest.fixture
def sample_raw_df():
    """Create a minimal sample dataset mimicking the UCI format."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "ID": range(1, n + 1),
        "LIMIT_BAL": np.random.uniform(10000, 500000, n),
        "SEX": np.random.choice([1, 2], n),
        "EDUCATION": np.random.choice([0, 1, 2, 3, 4, 5, 6], n),
        "MARRIAGE": np.random.choice([0, 1, 2, 3], n),
        "AGE": np.random.randint(21, 75, n),
        "PAY_0": np.random.choice([-2, -1, 0, 1, 2, 3], n),
        "PAY_2": np.random.choice([-2, -1, 0, 1, 2], n),
        "PAY_3": np.random.choice([-2, -1, 0, 1, 2], n),
        "PAY_4": np.random.choice([-2, -1, 0, 1, 2], n),
        "PAY_5": np.random.choice([-2, -1, 0, 1, 2], n),
        "PAY_6": np.random.choice([-2, -1, 0, 1, 2], n),
        "BILL_AMT1": np.random.uniform(-5000, 500000, n),
        "BILL_AMT2": np.random.uniform(-5000, 500000, n),
        "BILL_AMT3": np.random.uniform(-5000, 500000, n),
        "BILL_AMT4": np.random.uniform(-5000, 500000, n),
        "BILL_AMT5": np.random.uniform(-5000, 500000, n),
        "BILL_AMT6": np.random.uniform(-5000, 500000, n),
        "PAY_AMT1": np.random.uniform(0, 50000, n),
        "PAY_AMT2": np.random.uniform(0, 50000, n),
        "PAY_AMT3": np.random.uniform(0, 50000, n),
        "PAY_AMT4": np.random.uniform(0, 50000, n),
        "PAY_AMT5": np.random.uniform(0, 50000, n),
        "PAY_AMT6": np.random.uniform(0, 50000, n),
        "default.payment.next.month": np.random.choice([0, 1], n, p=[0.78, 0.22]),
    })
    return df


def test_clean_dataframe(sample_raw_df):
    df = clean_dataframe(sample_raw_df)

    # ID column removed
    assert "ID" not in df.columns

    # PAY_0 renamed to PAY_1
    assert "PAY_0" not in df.columns
    assert "PAY_1" in df.columns

    # Target renamed
    assert TARGET_COL in df.columns
    assert "default.payment.next.month" not in df.columns

    # EDUCATION values cleaned
    assert not df["EDUCATION"].isin([0, 5, 6]).any()

    # MARRIAGE values cleaned
    assert not df["MARRIAGE"].isin([0]).any()

    # Correct column order
    assert list(df.columns) == FEATURE_ORDER + [TARGET_COL]

    # No missing values
    assert df.isnull().sum().sum() == 0


def test_create_splits(sample_raw_df):
    splits = create_splits(sample_raw_df, seed=42)

    # Check keys
    assert "X_train" in splits
    assert "X_cal" in splits
    assert "X_test" in splits
    assert "y_train" in splits
    assert "y_cal" in splits
    assert "y_test" in splits

    # Check shapes
    n = len(sample_raw_df)
    total = len(splits["X_train"]) + len(splits["X_cal"]) + len(splits["X_test"])
    assert total == n

    # Check feature count
    assert splits["X_train"].shape[1] == len(FEATURE_ORDER)

    # Check no data leakage (target not in features)
    assert TARGET_COL not in splits["X_train"].columns


def test_split_stratification(sample_raw_df):
    splits = create_splits(sample_raw_df, seed=42)

    # Default rates should be approximately similar across splits
    rates = [
        splits["y_train"].mean(),
        splits["y_cal"].mean(),
        splits["y_test"].mean(),
    ]
    # Within 15% of each other (loose bound for small sample)
    assert max(rates) - min(rates) < 0.15
