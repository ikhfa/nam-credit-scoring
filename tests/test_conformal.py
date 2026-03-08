"""Tests for conformal prediction and calibration modules."""

import pytest
import numpy as np

from src.conformal.calibration import (
    expected_calibration_error,
    maximum_calibration_error,
    conformal_coverage_analysis,
)


class TestECE:
    def test_perfect_calibration(self):
        """Perfectly calibrated predictions should have ECE close to 0."""
        np.random.seed(42)
        n = 10000
        y_prob = np.random.uniform(0, 1, n)
        y_true = (np.random.uniform(0, 1, n) < y_prob).astype(int)

        ece, details = expected_calibration_error(y_true, y_prob, n_bins=10)
        assert ece < 0.05  # Should be very small

    def test_worst_case_calibration(self):
        """All predictions = 0.9, all labels = 0 -> high ECE."""
        y_prob = np.full(100, 0.9)
        y_true = np.zeros(100)

        ece, _ = expected_calibration_error(y_true, y_prob, n_bins=10)
        assert ece > 0.8

    def test_bin_details(self):
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        y_true = np.array([0, 0, 1, 1])

        ece, details = expected_calibration_error(y_true, y_prob, n_bins=10)
        assert "bins" in details
        assert len(details["bins"]) == 10

    def test_mce_geq_zero(self):
        y_prob = np.array([0.3, 0.5, 0.7])
        y_true = np.array([0, 1, 1])

        mce = maximum_calibration_error(y_true, y_prob)
        assert mce >= 0


class TestConformalCoverage:
    def test_coverage_format(self):
        n = 100
        y_true = np.random.choice([0, 1], n)
        # Simulate prediction sets: (n, 2, 3) for 3 alpha levels
        y_sets = np.random.choice([True, False], (n, 2, 3))
        alpha_levels = [0.05, 0.10, 0.20]

        results = conformal_coverage_analysis(y_true, y_sets, alpha_levels)

        assert len(results) == 3
        for r in results:
            assert "alpha" in r
            assert "empirical_coverage" in r
            assert "avg_set_size" in r
            assert 0 <= r["empirical_coverage"] <= 1
            assert 0 <= r["avg_set_size"] <= 2

    def test_full_sets_give_full_coverage(self):
        """If all prediction sets contain both classes, coverage should be 1.0."""
        n = 50
        y_true = np.random.choice([0, 1], n)
        y_sets = np.ones((n, 2, 1), dtype=bool)  # All sets = {0, 1}

        results = conformal_coverage_analysis(y_true, y_sets, [0.05])
        assert results[0]["empirical_coverage"] == 1.0
        assert results[0]["avg_set_size"] == 2.0
