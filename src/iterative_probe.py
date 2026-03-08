"""
Iterative linear probing: repeatedly fit probes and project out learned subspaces
to count independent copies of the belief state in the residual stream.

Pure numpy/sklearn — no torch, no I/O, no side effects.
"""

from dataclasses import dataclass, field
import numpy as np
from sklearn.linear_model import LinearRegression

from linear_probe import ProbeResult, fit_probe_with_split


@dataclass
class IterationResult:
    """Results from a single iteration of the iterative probe."""
    probe_result: ProbeResult
    subspace: np.ndarray          # (rank, d_model) orthonormal directions
    singular_values: np.ndarray   # (rank,) singular values of coef_


@dataclass
class IterativeProbeResult:
    """Full iterative probing result for one layer."""
    layer: int
    iterations: list[IterationResult] = field(default_factory=list)

    @property
    def n_significant(self) -> int:
        """Number of iterations with significant R²."""
        return len(self.iterations)

    @property
    def cumulative_dims(self) -> int:
        """Total number of dimensions used across all iterations."""
        return sum(it.subspace.shape[0] for it in self.iterations)

    @property
    def full_subspace(self) -> np.ndarray:
        """(cumulative_dims, d_model) — all directions concatenated."""
        if not self.iterations:
            return np.empty((0, 0))
        return np.vstack([it.subspace for it in self.iterations])


def extract_subspace(
    regressor: LinearRegression,
    rank_threshold: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract the row subspace of the regression coefficient matrix via SVD.

    Args:
        regressor: Fitted LinearRegression with coef_ of shape (d_target, d_features)
        rank_threshold: Relative threshold for considering a singular value significant.
                        Singular values below max(S) * rank_threshold are dropped.

    Returns:
        V: (rank, d_features) orthonormal basis for the row space of coef_
        S: (rank,) corresponding singular values
    """
    coef = regressor.coef_  # (d_target, d_features)
    U, S, Vt = np.linalg.svd(coef, full_matrices=False)
    # Keep directions with significant singular values
    if S[0] > 0:
        mask = S > S[0] * rank_threshold
    else:
        mask = np.zeros(len(S), dtype=bool)
    return Vt[mask], S[mask]


def project_out_subspace(X: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Project out a subspace from data.

    Args:
        X: (n_samples, d_features) data matrix
        V: (rank, d_features) orthonormal basis for the subspace to remove

    Returns:
        X_residual: (n_samples, d_features) with subspace projected out
    """
    # X_residual = X - X @ V^T @ V
    return X - (X @ V.T) @ V


def iterative_probe_single_layer(
    activations: np.ndarray,
    targets: np.ndarray,
    max_iterations: int = 10,
    r2_threshold: float = 0.05,
    test_fraction: float = 0.2,
    seed: int = 42,
    rank_threshold: float = 1e-3,
) -> IterativeProbeResult:
    """
    Iteratively probe and project out subspaces for a single layer.

    At each iteration:
    1. Fit a linear probe on the current (projected) activations
    2. If test R² < threshold, stop
    3. Extract the probe's subspace via SVD
    4. Project it out from activations
    5. Repeat

    Args:
        activations: (n_samples, d_model)
        targets: (n_samples, d_target)
        max_iterations: Maximum number of probe-project cycles
        r2_threshold: Minimum test R² to consider an iteration significant
        test_fraction: Fraction held out for testing
        seed: Random seed for train/test split
        rank_threshold: SVD rank threshold for extract_subspace

    Returns:
        IterativeProbeResult with all significant iterations
    """
    result = IterativeProbeResult(layer=-1)  # caller sets the layer
    X = activations.copy()

    for i in range(max_iterations):
        probe_result = fit_probe_with_split(
            X, targets, test_fraction=test_fraction, seed=seed,
        )

        if probe_result.test_r2 < r2_threshold:
            break

        V, S = extract_subspace(probe_result.regressor, rank_threshold=rank_threshold)

        if V.shape[0] == 0:
            break

        result.iterations.append(IterationResult(
            probe_result=probe_result,
            subspace=V,
            singular_values=S,
        ))

        X = project_out_subspace(X, V)

    return result


def iterative_probe_all_layers(
    activations_per_layer: np.ndarray,
    targets: np.ndarray,
    max_iterations: int = 10,
    r2_threshold: float = 0.05,
    test_fraction: float = 0.2,
    seed: int = 42,
    rank_threshold: float = 1e-3,
) -> list[IterativeProbeResult]:
    """
    Run iterative probing on all layers.

    Args:
        activations_per_layer: (n_layers, n_samples, d_model)
        targets: (n_samples, d_target)
        max_iterations: Maximum iterations per layer
        r2_threshold: Minimum test R² to continue
        test_fraction: Fraction held out for testing
        seed: Random seed
        rank_threshold: SVD rank threshold

    Returns:
        List of IterativeProbeResult, one per layer
    """
    n_layers = activations_per_layer.shape[0]
    results = []

    for layer_idx in range(n_layers):
        print(f"  Layer {layer_idx}:")
        layer_result = iterative_probe_single_layer(
            activations_per_layer[layer_idx],
            targets,
            max_iterations=max_iterations,
            r2_threshold=r2_threshold,
            test_fraction=test_fraction,
            seed=seed,
            rank_threshold=rank_threshold,
        )
        layer_result.layer = layer_idx

        for i, it in enumerate(layer_result.iterations):
            r = it.probe_result
            print(f"    Iter {i}: test R²={r.test_r2:.4f}, "
                  f"rank={it.subspace.shape[0]}, "
                  f"S={it.singular_values}")

        if layer_result.n_significant == 0:
            print(f"    No significant probe found (R² < {r2_threshold})")
        else:
            print(f"    Total: {layer_result.n_significant} copies, "
                  f"{layer_result.cumulative_dims} dims")

        results.append(layer_result)

    return results
