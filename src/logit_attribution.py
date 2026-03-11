"""
Direct Logit Attribution (DLA) for residual stream decomposition.

Decomposes final residual stream into additive components (embed, pos_embed,
attn_L, mlp_L) and attributes logit contributions via raw DLA, LN-corrected
DLA (Jacobian linearization), and component ablation.

Pure numpy/torch — no I/O, no side effects.
"""

import numpy as np
import torch as t


COMPONENT_NAMES = [
    "embed", "pos_embed",
    "attn_0", "mlp_0",
    "attn_1", "mlp_1",
    "attn_2", "mlp_2",
    "attn_3", "mlp_3",
]


def component_names_for_n_layers(n_layers: int) -> list[str]:
    """Return component names for a model with n_layers transformer blocks."""
    names = ["embed", "pos_embed"]
    for i in range(n_layers):
        names.append(f"attn_{i}")
        names.append(f"mlp_{i}")
    return names


def extract_components(cache, n_layers: int, position: int = -1) -> dict[str, np.ndarray]:
    """Extract additive residual stream components from a TransformerLens cache.

    Args:
        cache: TransformerLens ActivationCache from run_with_cache
        n_layers: Number of transformer blocks
        position: Sequence position to extract (-1 = last)

    Returns:
        Dict mapping component name to (batch, d_model) numpy array.
    """
    components = {}
    components["embed"] = cache["hook_embed"][:, position, :].cpu().numpy()
    components["pos_embed"] = cache["hook_pos_embed"][:, position, :].cpu().numpy()

    for layer in range(n_layers):
        components[f"attn_{layer}"] = cache[f"blocks.{layer}.hook_attn_out"][:, position, :].cpu().numpy()
        components[f"mlp_{layer}"] = cache[f"blocks.{layer}.hook_mlp_out"][:, position, :].cpu().numpy()

    return components


def verify_additivity(components: dict[str, np.ndarray], h_final: np.ndarray,
                      atol: float = 1e-4) -> float:
    """Verify that components sum to h_final. Returns max absolute error."""
    total = sum(components.values())
    max_err = np.max(np.abs(total - h_final))
    if max_err > atol:
        print(f"  WARNING: additivity error = {max_err:.6e} (threshold {atol})")
    return max_err


def layernorm_jacobian(h: np.ndarray, gamma: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Compute Jacobian of LayerNorm at point h.

    For LN(h) = gamma * (h - mu) / sigma + beta, the Jacobian is:
        J = diag(gamma/sigma) @ (I - (1/d)*11^T - (1/d)*(h-mu)(h-mu)^T/sigma^2)

    Args:
        h: (d_model,) single activation vector
        gamma: (d_model,) LN weight parameter
        eps: LN epsilon

    Returns:
        J: (d_model, d_model) Jacobian matrix
    """
    d = h.shape[0]
    mu = h.mean()
    h_centered = h - mu
    var = np.mean(h_centered ** 2)
    sigma = np.sqrt(var + eps)

    # (I - (1/d)*11^T - (1/d)*(h-mu)(h-mu)^T / sigma^2)
    ones = np.ones((d, d)) / d
    outer = np.outer(h_centered, h_centered) / (d * sigma ** 2)
    inner = np.eye(d) - ones - outer

    # diag(gamma/sigma) @ inner
    J = (gamma / sigma)[:, None] * inner
    return J


def layernorm_jacobian_batch(h_batch: np.ndarray, gamma: np.ndarray,
                              eps: float = 1e-5) -> np.ndarray:
    """Vectorized Jacobian computation for a batch of activations.

    Args:
        h_batch: (batch, d_model)
        gamma: (d_model,)
        eps: LN epsilon

    Returns:
        J_batch: (batch, d_model, d_model)
    """
    batch, d = h_batch.shape
    mu = h_batch.mean(axis=1, keepdims=True)  # (batch, 1)
    h_centered = h_batch - mu  # (batch, d)
    var = np.mean(h_centered ** 2, axis=1, keepdims=True)  # (batch, 1)
    sigma = np.sqrt(var + eps)  # (batch, 1)

    # Identity minus mean projection
    eye = np.eye(d)[None, :, :]  # (1, d, d)
    ones_mat = np.ones((1, d, d)) / d  # (1, d, d)

    # Outer product term: (batch, d, d) / (batch, 1, 1) * d
    outer = h_centered[:, :, None] * h_centered[:, None, :]  # (batch, d, d)
    outer = outer / (d * sigma[:, :, None] ** 2)  # (batch, d, d)

    inner = eye - ones_mat - outer  # (batch, d, d)

    # Scale by gamma/sigma
    scale = (gamma[None, :] / sigma)  # (batch, d)
    J_batch = scale[:, :, None] * inner  # (batch, d, d)

    return J_batch


def direct_logit_attribution(components: dict[str, np.ndarray],
                              W_U: np.ndarray) -> dict[str, np.ndarray]:
    """Raw DLA: component @ W_U, ignoring LayerNorm.

    Args:
        components: Dict of (batch, d_model) arrays
        W_U: (d_model, d_vocab) unembed matrix

    Returns:
        Dict of (batch, d_vocab) logit contributions per component.
    """
    return {name: comp @ W_U for name, comp in components.items()}


def ln_corrected_logit_attribution(
    components: dict[str, np.ndarray],
    h_final: np.ndarray,
    ln_weight: np.ndarray,
    ln_bias: np.ndarray,
    W_U: np.ndarray,
    b_U: np.ndarray | None = None,
    eps: float = 1e-5,
) -> dict[str, np.ndarray]:
    """LN-corrected DLA via Jacobian linearization.

    For each component c_i, the corrected logit contribution is:
        (J @ c_i) @ W_U
    where J is the LayerNorm Jacobian evaluated at h_final.

    The sum of all contributions equals LN(h_final) @ W_U (up to the
    constant term from LN bias and b_U).

    Args:
        components: Dict of (batch, d_model) arrays
        h_final: (batch, d_model) final residual stream
        ln_weight: (d_model,) LN gamma
        ln_bias: (d_model,) LN beta
        W_U: (d_model, d_vocab) unembed matrix
        b_U: (d_vocab,) unembed bias, optional
        eps: LN epsilon

    Returns:
        Dict of (batch, d_vocab) LN-corrected logit contributions.
        Also includes "ln_bias" and "b_U" constant contributions.
    """
    batch, d_model = h_final.shape

    # Compute Jacobian for each sample: (batch, d_model, d_model)
    J_batch = layernorm_jacobian_batch(h_final, ln_weight, eps)

    # For each component, compute (J @ c_i) @ W_U
    # J @ c_i: (batch, d_model) via einsum
    contributions = {}
    for name, comp in components.items():
        Jc = np.einsum("bij,bj->bi", J_batch, comp)  # (batch, d_model)
        contributions[name] = Jc @ W_U  # (batch, d_vocab)

    # Residual: LN(h_final) - J @ h_final, computed per sample.
    # Since LN is nonlinear, J @ h ≠ LN(h) - beta. The residual absorbs
    # both the LN bias and the nonlinear remainder.
    # LN(h) = gamma * (h - mu) / sigma + beta
    mu = h_final.mean(axis=1, keepdims=True)
    var = np.mean((h_final - mu) ** 2, axis=1, keepdims=True)
    sigma = np.sqrt(var + eps)
    ln_h = ln_weight[None, :] * (h_final - mu) / sigma + ln_bias[None, :]  # (batch, d_model)

    Jh = np.einsum("bij,bj->bi", J_batch, h_final)  # (batch, d_model)
    residual = ln_h - Jh  # (batch, d_model)
    contributions["ln_residual"] = residual @ W_U  # (batch, d_vocab)

    if b_U is not None:
        contributions["b_U"] = np.tile(b_U[None, :], (batch, 1))

    return contributions


def verify_ln_corrected_sum(contributions: dict[str, np.ndarray],
                             true_logits: np.ndarray,
                             atol: float = 1e-3) -> float:
    """Verify LN-corrected contributions sum to true logits."""
    total = sum(contributions.values())
    max_err = np.max(np.abs(total - true_logits))
    if max_err > atol:
        print(f"  WARNING: LN-corrected sum error = {max_err:.6e} (threshold {atol})")
    return max_err


def component_statistics(
    logit_contribs: dict[str, np.ndarray],
    bayes_logits: np.ndarray,
    delta: np.ndarray,
    component_names: list[str] | None = None,
) -> "pd.DataFrame":
    """Compute per-component statistics for logit contributions.

    For each component i, computes the overlap fraction:
        overlap_bayes_i = <c_i, bayes_logits> / ||bayes_logits||²
        overlap_delta_i = <c_i, delta> / ||delta||²
    averaged over samples. For the LN-corrected case where contributions
    sum exactly to the true logits, overlap_bayes sums to a value close to 1
    (exact if true logits == bayes logits).

    Args:
        logit_contribs: Dict of (batch, d_vocab) arrays
        bayes_logits: (batch, d_vocab) Bayes-optimal log-probabilities
        delta: (batch, d_vocab) nonlinear correction = bayes_logits - M_best @ beliefs
        component_names: If given, only include these components (in order)

    Returns:
        DataFrame with columns: component, overlap_bayes, overlap_delta
    """
    import pandas as pd

    if component_names is None:
        component_names = [k for k in logit_contribs.keys()]

    # Per-sample ||bayes||² and ||delta||²
    bayes_norm_sq = np.sum(bayes_logits ** 2, axis=1)  # (batch,)
    delta_norm_sq = np.sum(delta ** 2, axis=1)  # (batch,)

    rows = []
    for name in component_names:
        contrib = logit_contribs[name]  # (batch, d_vocab)

        # <c_i, bayes> / ||bayes||² per sample, then average
        dot_bayes = np.sum(contrib * bayes_logits, axis=1)  # (batch,)
        overlap_bayes = float(np.mean(dot_bayes / (bayes_norm_sq + 1e-12)))

        # <c_i, delta> / ||delta||² per sample, then average
        dot_delta = np.sum(contrib * delta, axis=1)  # (batch,)
        overlap_delta = float(np.mean(dot_delta / (delta_norm_sq + 1e-12)))

        rows.append({
            "component": name,
            "overlap_bayes": overlap_bayes,
            "overlap_delta": overlap_delta,
        })

    return pd.DataFrame(rows)


def logit_lens(
    cache,
    model,
    n_layers: int,
    position: int = -1,
) -> list[np.ndarray]:
    """Apply LN + W_U at each intermediate residual stream point.

    Args:
        cache: TransformerLens ActivationCache
        model: HookedTransformer (for LN and W_U parameters)
        n_layers: Number of transformer blocks
        position: Sequence position

    Returns:
        List of (batch, d_vocab) log-probability arrays, one per layer
        (including after embed = layer -1, and after each block).
    """
    import torch.nn.functional as F

    W_U = model.W_U.data  # (d_model, d_vocab)
    b_U = model.b_U.data if hasattr(model, 'b_U') and model.b_U is not None else None
    ln_weight = model.ln_final.w.data
    ln_bias = model.ln_final.b.data

    results = []

    # After embedding (before any blocks)
    h = cache["hook_embed"][:, position, :] + cache["hook_pos_embed"][:, position, :]
    h_ln = t.nn.functional.layer_norm(h, (h.shape[-1],), weight=ln_weight, bias=ln_bias)
    logits = h_ln @ W_U
    if b_U is not None:
        logits = logits + b_U
    results.append(F.log_softmax(logits, dim=-1).cpu().numpy())

    # After each block
    for layer in range(n_layers):
        h = cache["resid_post", layer][:, position, :]
        h_ln = t.nn.functional.layer_norm(h, (h.shape[-1],), weight=ln_weight, bias=ln_bias)
        logits = h_ln @ W_U
        if b_U is not None:
            logits = logits + b_U
        results.append(F.log_softmax(logits, dim=-1).cpu().numpy())

    return results
