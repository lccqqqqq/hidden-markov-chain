"""
E10 — gauge canonicalization before quantization
(notes/compression/singular_geometry_evaluation.md §4, Tier 4; docx §10.1).

Exact reparameterizations of the fp32 function (no LayerNorm in these models, so all of
them are exact symmetries), chosen to reduce the dynamic range the *per-output-channel*
quantization grids must span:

  relu_equalize   per MLP neuron j (ReLU positive homogeneity):
                  W_in[:, j] *= c_j, b_in[j] *= c_j, W_out[j, :] /= c_j,
                  c_j = sqrt(range_out_j / range_in_j), range_in_j = max(|W_in[:, j]|, |b_in[j]|),
                  range_out_j = max|W_out[j, :]|.      (Nagel et al., ICCV 2019, cross-layer equalization)
  qk_balance      per head h, per d_head index k (diagonal GL(d_head) gauge of q·kᵀ):
                  W_Q[h][:, k] *= a_k, b_Q[h][k] *= a_k, W_K[h][:, k] /= a_k, b_K[h][k] /= a_k,
                  a_k = sqrt(max|W_K[h][:, k]| / max|W_Q[h][:, k]|).
  vo_balance      same for (W_V[h][:, k], b_V[h][k]) vs W_O[h][k, :].
  qk_rotate       per head, A ∈ O(d_head): W_Q[h] → W_Q[h] A, b_Q[h] → b_Q[h] A,
                  W_K[h] → W_K[h] A, b_K[h] → b_K[h] A   (q·kᵀ invariant since A Aᵀ = I).
  vo_rotate       per head, A ∈ O(d_head): W_V[h] → W_V[h] A, b_V[h] → b_V[h] A,
                  W_O[h] → Aᵀ W_O[h]                    (v W_O invariant).
  all             relu_equalize, qk_balance, vo_balance, qk_rotate, vo_rotate in that order.

Rotations are found by Adam (200 steps, lr 0.02) on a Cayley parametrization
A = (I − S)(I + S)⁻¹, S skew-symmetric, all heads of a layer jointly, minimizing the
range proxy of the per-channel RTN mean-squared error under the uniform-rounding-error
model,   proxy = Σ_channels n_rows · (max − min)² ,
over the quantization channels affected by the head: the d_head columns of W_Q[h] A and
W_K[h] A (qk), or the d_head columns of W_V[h] A plus every d_model output column of the
full W_O (vo; the head's rows Aᵀ W_O[h] sit inside those columns together with the other
heads' rows). The factor 1/(12 (2^b − 1)²) is dropped, so the argmin is bit-width
independent; `bits` is accepted only for logging.

Which of these can matter under the shared per-output-channel protocol: a quantization
channel of W_Q/W_K/W_V/W_in is one column, so scaling a whole column (qk_balance, and the
W_in side of relu_equalize, and the W_V side of vo_balance) leaves its relative rounding
error unchanged — those transforms can only act through W_out / W_O, whose channels mix
rows, and through the rotations, which mix columns. qk_balance therefore is expected to
be an exact no-op for RTN and is kept as a control.

`canonicalize` asserts fp32 invariance (max |Δlogits| < 1e-4 on 256 sequences).
"""
from __future__ import annotations

import copy

import torch as t
from transformer_lens import HookedTransformer

MODES = ("none", "relu_equalize", "qk_balance", "vo_balance", "qk_rotate", "vo_rotate", "all")


@t.no_grad()
def _relu_equalize(model: HookedTransformer) -> None:
    for blk in model.blocks:
        W_in, b_in, W_out = blk.mlp.W_in, blk.mlp.b_in, blk.mlp.W_out
        r_in = t.maximum(W_in.abs().amax(dim=0), b_in.abs()).clamp_min(1e-12)   # (d_mlp,)
        r_out = W_out.abs().amax(dim=1).clamp_min(1e-12)                        # (d_mlp,)
        c = (r_out / r_in).sqrt()
        W_in.mul_(c[None, :]); b_in.mul_(c); W_out.div_(c[:, None])


@t.no_grad()
def _qk_balance(model: HookedTransformer) -> None:
    for blk in model.blocks:
        a = blk.attn
        rq = a.W_Q.abs().amax(dim=1).clamp_min(1e-12)      # (head, d_head)
        rk = a.W_K.abs().amax(dim=1).clamp_min(1e-12)
        s = (rk / rq).sqrt()
        a.W_Q.mul_(s[:, None, :]); a.b_Q.mul_(s); a.W_K.div_(s[:, None, :]); a.b_K.div_(s)


@t.no_grad()
def _vo_balance(model: HookedTransformer) -> None:
    for blk in model.blocks:
        a = blk.attn
        rv = a.W_V.abs().amax(dim=1).clamp_min(1e-12)      # (head, d_head)
        ro = a.W_O.abs().amax(dim=2).clamp_min(1e-12)      # (head, d_head)
        s = (ro / rv).sqrt()
        a.W_V.mul_(s[:, None, :]); a.b_V.mul_(s); a.W_O.div_(s[:, :, None])


def _cayley(S: t.Tensor) -> t.Tensor:
    """(head, d, d) skew-symmetric -> orthogonal, A = (I - S)(I + S)^-1."""
    S = S - S.transpose(-1, -2)
    I = t.eye(S.shape[-1], dtype=S.dtype)
    return t.linalg.solve(I + S, I - S, left=False)          # (I - S) (I + S)^-1


def _range_proxy(M: t.Tensor, dim: int) -> t.Tensor:
    """Σ_channels n_rows (max - min)² with channels along `dim` reduced away over the rows."""
    n_rows = M.shape[dim]
    return n_rows * (M.amax(dim=dim) - M.amin(dim=dim)).pow(2).sum()


def _optimize_rotation(loss_fn, n_heads: int, d_head: int, steps: int, lr: float, seed: int) -> t.Tensor:
    g = t.Generator().manual_seed(seed)
    S = t.nn.Parameter(0.01 * t.randn(n_heads, d_head, d_head, generator=g, dtype=t.float64))
    opt = t.optim.Adam([S], lr=lr)
    best, bestA = float("inf"), _cayley(S.detach())
    for _ in range(steps):
        A = _cayley(S)
        loss = loss_fn(A)
        if loss.item() < best:
            best, bestA = loss.item(), A.detach().clone()
        opt.zero_grad(); loss.backward(); opt.step()
    return bestA


@t.no_grad()
def _apply_qk_rot(a, A):
    A32 = A.to(a.W_Q.dtype)
    a.W_Q.copy_(t.einsum("hdk,hkl->hdl", a.W_Q, A32)); a.b_Q.copy_(t.einsum("hk,hkl->hl", a.b_Q, A32))
    a.W_K.copy_(t.einsum("hdk,hkl->hdl", a.W_K, A32)); a.b_K.copy_(t.einsum("hk,hkl->hl", a.b_K, A32))


@t.no_grad()
def _apply_vo_rot(a, A):
    A32 = A.to(a.W_V.dtype)
    a.W_V.copy_(t.einsum("hdk,hkl->hdl", a.W_V, A32)); a.b_V.copy_(t.einsum("hk,hkl->hl", a.b_V, A32))
    a.W_O.copy_(t.einsum("hkl,hkd->hld", A32, a.W_O))                 # Aᵀ W_O per head


def _qk_rotate(model: HookedTransformer, steps: int, lr: float, seed: int) -> None:
    for blk in model.blocks:
        a = blk.attn
        WQ, WK = a.W_Q.detach().double(), a.W_K.detach().double()

        def loss_fn(A):
            return _range_proxy(WQ @ A, dim=1) + _range_proxy(WK @ A, dim=1)   # columns = channels
        A = _optimize_rotation(loss_fn, a.W_Q.shape[0], a.W_Q.shape[2], steps, lr, seed)
        _apply_qk_rot(a, A)


def _vo_rotate(model: HookedTransformer, steps: int, lr: float, seed: int) -> None:
    for blk in model.blocks:
        a = blk.attn
        WV, WO = a.W_V.detach().double(), a.W_O.detach().double()

        def loss_fn(A):
            WO_new = t.einsum("hkl,hkd->hld", A, WO)                     # (head, d_head, d_model)
            full = WO_new.reshape(-1, WO_new.shape[-1])                  # rows = (head, d_head)
            return _range_proxy(WV @ A, dim=1) + _range_proxy(full, dim=0)
        A = _optimize_rotation(loss_fn, a.W_V.shape[0], a.W_V.shape[2], steps, lr, seed)
        _apply_vo_rot(a, A)


@t.no_grad()
def _max_logit_diff(m1: HookedTransformer, m2: HookedTransformer, tokens: t.Tensor) -> float:
    return float((m1(tokens[:, :-1]) - m2(tokens[:, :-1])).abs().max())


def canonicalize(model: HookedTransformer, mode: str, check_tokens: t.Tensor | None = None,
                 rot_steps: int = 200, rot_lr: float = 0.02, seed: int = 0, tol: float = 1e-4,
                 bits: int | None = None) -> HookedTransformer:
    """Return a canonicalized deep copy; fp32 function unchanged (asserted on check_tokens)."""
    assert mode in MODES, mode
    out = copy.deepcopy(model).eval()
    steps = ["relu_equalize", "qk_balance", "vo_balance", "qk_rotate", "vo_rotate"] if mode == "all" \
        else ([] if mode == "none" else [mode])
    for s in steps:
        if s == "relu_equalize":
            _relu_equalize(out)
        elif s == "qk_balance":
            _qk_balance(out)
        elif s == "vo_balance":
            _vo_balance(out)
        elif s == "qk_rotate":
            _qk_rotate(out, rot_steps, rot_lr, seed)
        elif s == "vo_rotate":
            _vo_rotate(out, rot_steps, rot_lr, seed)
        if check_tokens is not None:
            d = _max_logit_diff(model, out, check_tokens[:256])
            assert d < tol, f"{s}: fp32 function changed, max |Δlogits| = {d:.2e}"
    return out
