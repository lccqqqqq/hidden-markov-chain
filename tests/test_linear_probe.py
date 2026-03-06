"""Tests for src/linear_probe.py"""

import numpy as np
import pytest
from linear_probe import (
    fit_probe_with_split,
    probe_layer_wise,
    probe_concatenated,
    _train_test_indices,
)


class TestFitProbeWithSplit:
    def test_perfect_probe(self):
        """When activations == targets, R² should be ~1."""
        rng = np.random.default_rng(0)
        targets = rng.random((500, 3))
        result = fit_probe_with_split(targets, targets, test_fraction=0.2, seed=0)
        assert result.train_r2 > 0.999
        assert result.test_r2 > 0.999
        assert result.test_mse < 1e-10

    def test_random_probe_low_r2(self):
        """Random activations should give low R² on random targets."""
        rng = np.random.default_rng(1)
        X = rng.random((500, 10))
        y = rng.random((500, 3))
        result = fit_probe_with_split(X, y, test_fraction=0.2, seed=1)
        assert result.test_r2 < 0.2

    def test_train_test_disjoint(self):
        """Train and test indices must not overlap."""
        train_idx, test_idx = _train_test_indices(100, 0.2, seed=42)
        assert len(set(train_idx) & set(test_idx)) == 0
        assert len(train_idx) + len(test_idx) == 100

    def test_baseline_mse_positive(self):
        """Baseline MSE (predicting train mean) should be positive for non-constant targets."""
        rng = np.random.default_rng(2)
        X = rng.random((200, 5))
        y = rng.random((200, 3))
        result = fit_probe_with_split(X, y)
        assert result.baseline_mse > 0

    def test_output_shapes(self):
        """Check prediction array shapes match train/test sizes."""
        rng = np.random.default_rng(3)
        n, d_feat, d_target = 100, 5, 3
        X = rng.random((n, d_feat))
        y = rng.random((n, d_target))
        result = fit_probe_with_split(X, y, test_fraction=0.3, seed=3)
        n_test = max(1, int(n * 0.3))
        n_train = n - n_test
        assert result.train_predictions.shape == (n_train, d_target)
        assert result.test_predictions.shape == (n_test, d_target)


class TestProbeLayerWise:
    def test_consistent_split_across_layers(self):
        """All layers should use the same train/test split."""
        rng = np.random.default_rng(4)
        n_layers, n_samples, d_model, d_target = 3, 200, 8, 3
        X = rng.random((n_layers, n_samples, d_model))
        y = rng.random((n_samples, d_target))
        results = probe_layer_wise(X, y, test_fraction=0.2, seed=4)

        assert len(results) == n_layers
        # All layers should have the same number of train/test samples
        for r in results:
            assert r.train_predictions.shape[0] == results[0].train_predictions.shape[0]
            assert r.test_predictions.shape[0] == results[0].test_predictions.shape[0]

    def test_returns_one_result_per_layer(self):
        rng = np.random.default_rng(5)
        X = rng.random((4, 100, 6))
        y = rng.random((100, 2))
        results = probe_layer_wise(X, y)
        assert len(results) == 4


class TestProbeConcatenated:
    def test_feature_dimension(self):
        """Concatenated features should have dimension n_layers * d_model."""
        rng = np.random.default_rng(6)
        n_layers, n_samples, d_model = 3, 100, 8
        X = rng.random((n_layers, n_samples, d_model))
        y = rng.random((n_samples, 2))
        result = probe_concatenated(X, y)
        # The regressor should have been fitted on n_layers * d_model features
        assert result.regressor.coef_.shape[1] == n_layers * d_model

    def test_returns_single_result(self):
        rng = np.random.default_rng(7)
        X = rng.random((2, 50, 4))
        y = rng.random((50, 3))
        result = probe_concatenated(X, y)
        assert hasattr(result, "test_r2")
