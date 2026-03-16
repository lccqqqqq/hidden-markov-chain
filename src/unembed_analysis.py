"""
Unembed decomposition analysis: how W_U reads belief-state subspaces.

Pure numpy — no torch, no I/O, no side effects.
"""

import numpy as np
from numpy.linalg import svd, norm, pinv


def compute_observation_matrix(process) -> np.ndarray:
    """O[j,i] = P(observe j | state i) = sum_k E[j,i,k]. Shape (d_vocab, n_states)."""
    return process.emission_matrices.sum(axis=2)


def wu_subspace_overlaps(W_U: np.ndarray, subspaces: list[np.ndarray]) -> np.ndarray:
    """Compute ||V_k @ W_U||_F^2 / ||W_U||_F^2 for each subspace V_k.

    Args:
        W_U: (d_model, d_vocab)
        subspaces: list of (rank_k, d_model) orthonormal row matrices

    Returns:
        (n_subspaces,) array of overlap fractions.
    """
    wu_norm_sq = norm(W_U, 'fro') ** 2
    overlaps = np.array([
        norm(V @ W_U, 'fro') ** 2 / wu_norm_sq
        for V in subspaces
    ])
    return overlaps


def wu_per_column_overlaps(W_U: np.ndarray, subspaces: list[np.ndarray]) -> np.ndarray:
    """Per-column overlap: ||V_k @ w_j||^2 / ||w_j||^2 for each subspace and token.

    Args:
        W_U: (d_model, d_vocab)
        subspaces: list of (rank_k, d_model) orthonormal row matrices

    Returns:
        (n_subspaces, d_vocab) array of per-column overlap fractions.
    """
    d_vocab = W_U.shape[1]
    col_norms_sq = np.array([norm(W_U[:, j]) ** 2 for j in range(d_vocab)])
    result = np.zeros((len(subspaces), d_vocab))
    for k, V in enumerate(subspaces):
        for j in range(d_vocab):
            proj = V @ W_U[:, j]  # (rank_k,)
            result[k, j] = norm(proj) ** 2 / col_norms_sq[j]
    return result


def effective_emission_matrix(
    W_probe_coef: np.ndarray,
    W_U: np.ndarray,
    subspace: np.ndarray,
) -> np.ndarray:
    """Compose probe and unembed through a subspace to get effective map.

    The flow is: belief → probe → activation → subspace → W_U → logits.
    We extract the effective (n_states, d_vocab) map from belief to logit
    contribution through the given subspace.

    Args:
        W_probe_coef: (n_states, d_model) probe weight matrix (regressor.coef_)
        W_U: (d_model, d_vocab) unembed weight matrix
        subspace: (rank, d_model) orthonormal rows

    Returns:
        M_eff: (n_states, d_vocab) effective belief-to-logit map through subspace
    """
    # A = W_probe @ V^T: (n_states, rank) — maps subspace coords to belief
    A = W_probe_coef @ subspace.T  # (n_states, rank)
    # B = V @ W_U: (rank, d_vocab) — maps subspace coords to logits
    B = subspace @ W_U  # (rank, d_vocab)
    # M_eff = pinv(A) @ B: maps belief to logit contribution
    # But we want belief → logit, so: pinv(A) gives (rank, n_states)
    # Then pinv(A) @ ... no, A maps rank → n_states (approximately)
    # We want: given belief b, what logit contribution comes through this subspace?
    # activation ≈ W_probe^T @ b (inverse of probe), project to subspace: V @ V^T @ W_probe^T @ b
    # logit contribution: W_U^T @ V^T @ V @ W_probe^T @ b = B^T @ A^T-pseudo @ b?
    # Actually: probe says activations ≈ coef^T @ belief (since probe is belief = coef @ actv + intercept)
    # So actv ≈ pinv(coef) @ belief, and through subspace:
    # logit = (V @ W_U)^T @ (V @ pinv(coef)^T @ b) ... let me think more carefully.
    #
    # The probe: belief_pred = coef @ actv + intercept, i.e. coef: (n_states, d_model)
    # The unembed: logit = actv @ W_U, i.e. W_U: (d_model, d_vocab)
    # Subspace projection of actv: V^T @ V @ actv
    # Logit from subspace: (V^T @ V @ actv) @ W_U = actv @ (V^T @ V @ W_U)  ... no
    # Actually (V^T @ V @ actv)^T @ ... let me just use index notation.
    #
    # actv_proj = V^T (V actv) where V is (rank, d_model), actv is (d_model,)
    # So actv_proj is (d_model,), then logit_proj = actv_proj @ W_U = (V^T (V actv))^T W_U
    # = (V actv)^T V W_U = (V actv)^T B where B = V @ W_U is (rank, d_vocab)
    # So logit_proj_j = sum_r (V actv)_r B_{r,j}
    #
    # From probe: belief ≈ coef @ actv, so actv ≈ pinv(coef) @ belief
    # V @ actv ≈ V @ pinv(coef) @ belief = (rank, n_states) @ (n_states,)
    # Then logit_proj = B^T @ (V @ pinv(coef) @ belief) = (B^T @ V @ pinv(coef)) @ belief
    #
    # M_eff = (B^T @ V @ pinv(coef))^T = pinv(coef)^T @ V^T @ B
    # Hmm, let me just define it cleanly:
    # logit_j = sum_r B_{r,j} * sum_s C_{r,s} * belief_s
    # where C = V @ pinv(coef) : (rank, n_states)
    # So M_eff[s,j] = sum_r C_{r,s} * B_{r,j} = C^T @ B
    C = subspace @ pinv(W_probe_coef)  # (rank, n_states)
    M_eff = C.T @ B  # (n_states, d_vocab)
    return M_eff


def principal_angles(subspace_A: np.ndarray, subspace_B: np.ndarray) -> np.ndarray:
    """Principal angles between two subspaces.

    Args:
        subspace_A: (dim_A, d) orthonormal rows
        subspace_B: (dim_B, d) orthonormal rows

    Returns:
        Array of principal angles in radians, length = min(dim_A, dim_B).
    """
    M = subspace_A @ subspace_B.T  # (dim_A, dim_B)
    _, S, _ = svd(M, full_matrices=False)
    # Clamp to [0, 1] for numerical safety
    S = np.clip(S, 0.0, 1.0)
    return np.arccos(S)


def wu_read_subspace(W_U: np.ndarray) -> np.ndarray:
    """Extract the column space of W_U (the 'read subspace') as orthonormal rows.

    Args:
        W_U: (d_model, d_vocab) unembed matrix

    Returns:
        V: (rank, d_model) orthonormal rows spanning the column space of W_U.
    """
    U, S, _ = svd(W_U, full_matrices=False)
    # Keep all non-negligible singular values
    mask = S > S[0] * 1e-6 if S[0] > 0 else np.zeros(len(S), dtype=bool)
    return U[:, mask].T  # (rank, d_model)


def wu_block_isometry_analysis(
    W_U: np.ndarray,
    subspaces: list[np.ndarray],
    n_random: int = 10000,
) -> dict:
    """Compare W_U blocks B_k = V_k @ W_U across subspaces via Procrustes.

    For each subspace k, compute B_k = V_k @ W_U (shape rank_k x d_vocab),
    normalize to unit Frobenius norm, and find the best orthogonal alignment
    R_k mapping B_hat_0 to B_hat_k via the Procrustes problem.

    Only subspaces with rank == rank of subspace 0 are compared (others get NaN).

    Args:
        W_U: (d_model, d_vocab) unembed weight matrix
        subspaces: list of (rank_k, d_model) orthonormal row matrices
        n_random: number of Monte Carlo samples for random baseline

    Returns:
        dict with arrays indexed by subspace k:
          - blocks: list of B_k = V_k @ W_U arrays
          - frob_norms: ||B_k||_F
          - procrustes_residuals: ||R_k B_hat_0 - B_hat_k||_F
          - aligned_cosines: trace(R_k B_hat_0 B_hat_k^T) / rank
          - det_R: det(R_k) (+1 rotation, -1 reflection)
          - random_baseline_mean: expected residual for random matrices
          - random_baseline_std: std of random baseline
    """
    n_sub = len(subspaces)
    rank0 = subspaces[0].shape[0]
    d_vocab = W_U.shape[1]

    # Compute blocks
    blocks = [V @ W_U for V in subspaces]
    frob_norms = np.array([norm(B, 'fro') for B in blocks])

    # Normalize
    B_hat = []
    for B, fn in zip(blocks, frob_norms):
        if fn > 1e-12:
            B_hat.append(B / fn)
        else:
            B_hat.append(B * 0.0)

    # Reference block
    B_ref = B_hat[0]  # (rank0, d_vocab)

    residuals = np.full(n_sub, np.nan)
    cosines = np.full(n_sub, np.nan)
    det_R = np.full(n_sub, np.nan)

    for k in range(n_sub):
        if subspaces[k].shape[0] != rank0:
            continue
        # Procrustes: find R minimizing ||R @ B_ref - B_hat_k||_F
        # SVD of B_hat_k @ B_ref^T  (rank0 x rank0)
        M = B_hat[k] @ B_ref.T
        U, S, Vt = svd(M)
        R = U @ Vt
        residuals[k] = norm(R @ B_ref - B_hat[k], 'fro')
        cosines[k] = np.trace(M @ R.T) / rank0  # = sum(S) / rank0
        det_R[k] = np.linalg.det(R)

    # Random baseline: pairs of random (rank0, d_vocab) Gaussian matrices
    rng = np.random.default_rng(42)
    rand_residuals = np.zeros(n_random)
    for i in range(n_random):
        A = rng.standard_normal((rank0, d_vocab))
        B = rng.standard_normal((rank0, d_vocab))
        A /= norm(A, 'fro')
        B /= norm(B, 'fro')
        M = B @ A.T
        U, S, Vt = svd(M)
        R = U @ Vt
        rand_residuals[i] = norm(R @ A - B, 'fro')

    return {
        "blocks": blocks,
        "frob_norms": frob_norms,
        "procrustes_residuals": residuals,
        "aligned_cosines": cosines,
        "det_R": det_R,
        "random_baseline_mean": float(rand_residuals.mean()),
        "random_baseline_std": float(rand_residuals.std()),
    }


def logit_subspace_contributions(
    h_postLN: np.ndarray,
    W_U: np.ndarray,
    subspaces: list[np.ndarray],
) -> np.ndarray:
    """Decompose logits into per-subspace contributions.

    logit_k = (V_k^T @ V_k @ h) @ W_U for each sample.

    Args:
        h_postLN: (n_samples, d_model) post-LN activations
        W_U: (d_model, d_vocab) unembed matrix
        subspaces: list of (rank_k, d_model) orthonormal row matrices

    Returns:
        (n_subspaces, n_samples, d_vocab) array of logit contributions.
    """
    result = np.zeros((len(subspaces), h_postLN.shape[0], W_U.shape[1]))
    for k, V in enumerate(subspaces):
        # Project h onto subspace: (n_samples, d_model)
        h_proj = (h_postLN @ V.T) @ V  # (n_samples, d_model)
        result[k] = h_proj @ W_U  # (n_samples, d_vocab)
    return result
