"""
Core linear probing logic for mapping transformer activations to HMM belief states.

Pure numpy/sklearn — no torch, no I/O, no side effects.
"""

from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LinearRegression


@dataclass
class ProbeResult:
    """Results from fitting a linear probe with train/test split."""
    regressor: LinearRegression
    train_predictions: np.ndarray   # (n_train, d_target)
    test_predictions: np.ndarray    # (n_test, d_target)
    train_mse: float
    test_mse: float
    train_r2: float
    test_r2: float
    baseline_mse: float             # predicting train mean on test set


def _compute_mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean((predictions - targets) ** 2))


def _train_test_indices(
    n_samples: int,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    n_test = max(1, int(n_samples * test_fraction))
    return indices[n_test:], indices[:n_test]


def fit_probe_with_split(
    activations: np.ndarray,
    targets: np.ndarray,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> ProbeResult:
    """
    Fit a linear regression from activations to targets with train/test split.

    Args:
        activations: (n_samples, d_features)
        targets: (n_samples, d_target)
        test_fraction: Fraction of data to hold out for testing
        seed: Random seed for reproducible splitting

    Returns:
        ProbeResult with fitted regressor and metrics
    """
    n_samples = activations.shape[0]
    train_idx, test_idx = _train_test_indices(n_samples, test_fraction, seed)

    X_train, X_test = activations[train_idx], activations[test_idx]
    y_train, y_test = targets[train_idx], targets[test_idx]

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    train_preds = regressor.predict(X_train)
    test_preds = regressor.predict(X_test)

    train_mse = _compute_mse(train_preds, y_train)
    test_mse = _compute_mse(test_preds, y_test)
    train_r2 = float(regressor.score(X_train, y_train))
    test_r2 = float(regressor.score(X_test, y_test))

    # Baseline: predict train mean on test set
    train_mean = y_train.mean(axis=0, keepdims=True)
    baseline_mse = _compute_mse(np.broadcast_to(train_mean, y_test.shape), y_test)

    return ProbeResult(
        regressor=regressor,
        train_predictions=train_preds,
        test_predictions=test_preds,
        train_mse=train_mse,
        test_mse=test_mse,
        train_r2=train_r2,
        test_r2=test_r2,
        baseline_mse=baseline_mse,
    )


def probe_layer_wise(
    activations_per_layer: np.ndarray,
    targets: np.ndarray,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> list[ProbeResult]:
    """
    Fit a separate linear probe per layer, using the same train/test split.

    Args:
        activations_per_layer: (n_layers, n_samples, d_model)
        targets: (n_samples, d_target)
        test_fraction: Fraction held out for testing
        seed: Random seed (same split across all layers)

    Returns:
        List of ProbeResult, one per layer
    """
    n_layers = activations_per_layer.shape[0]
    n_samples = activations_per_layer.shape[1]
    train_idx, test_idx = _train_test_indices(n_samples, test_fraction, seed)

    results = []
    for layer in range(n_layers):
        X = activations_per_layer[layer]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = targets[train_idx], targets[test_idx]

        regressor = LinearRegression()
        regressor.fit(X_train, y_train)

        train_preds = regressor.predict(X_train)
        test_preds = regressor.predict(X_test)

        train_mse = _compute_mse(train_preds, y_train)
        test_mse = _compute_mse(test_preds, y_test)
        train_r2 = float(regressor.score(X_train, y_train))
        test_r2 = float(regressor.score(X_test, y_test))

        train_mean = y_train.mean(axis=0, keepdims=True)
        baseline_mse = _compute_mse(
            np.broadcast_to(train_mean, y_test.shape), y_test
        )

        results.append(ProbeResult(
            regressor=regressor,
            train_predictions=train_preds,
            test_predictions=test_preds,
            train_mse=train_mse,
            test_mse=test_mse,
            train_r2=train_r2,
            test_r2=test_r2,
            baseline_mse=baseline_mse,
        ))

    return results


def probe_concatenated(
    activations_per_layer: np.ndarray,
    targets: np.ndarray,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> ProbeResult:
    """
    Concatenate activations across layers, then fit a single probe.

    Args:
        activations_per_layer: (n_layers, n_samples, d_model)
        targets: (n_samples, d_target)
        test_fraction: Fraction held out for testing
        seed: Random seed

    Returns:
        Single ProbeResult with d_features = n_layers * d_model
    """
    # (n_layers, n_samples, d_model) -> (n_samples, n_layers * d_model)
    n_layers, n_samples, d_model = activations_per_layer.shape
    X = activations_per_layer.transpose(1, 0, 2).reshape(n_samples, n_layers * d_model)
    return fit_probe_with_split(X, targets, test_fraction=test_fraction, seed=seed)
