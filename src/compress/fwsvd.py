"""
FWSVD — Fisher-weighted SVD. Hsu et al., ICLR 2022, arXiv:2207.00112
("Language model compression with weighted low-rank factorization").

First parameter-count method (all quantizers keep the count fixed). Reference algorithm:
  * Empirical Fisher per weight: Î_ij = (1/N) Σ_batches (∂L/∂W_ij)², accumulated over
    calibration batches of the true cross-entropy loss.
  * Exact Fisher-weighted low-rank is intractable; the paper's row-aggregated surrogate:
    D = diag(sqrt(Σ_j Î_ij)) over the OUTPUT rows of W (d_out, d_in); SVD(D W) = U S Vᵀ;
    keep rank r; factors A = D⁻¹ U_r S_r (d_out, r), B = V_rᵀ (r, d_in);
    parameters per matrix drop from d_out·d_in to r(d_out + d_in).
  * weighted=False gives plain truncated SVD — the paper's own baseline, reported as
    method "SVD" (a defined control, not a tweak of FWSVD).
  * The paper fine-tunes after factorization; epochs=0 rows are the "w/o fine-tune"
    variant the paper also reports. Fine-tuning trains the factors A, B directly (via a
    torch parametrization computing W = A @ B), same loop settings as STE-QAT (Adam,
    lr 1e-4, batch 128, seed 0, train split).
  * Rank: uniform ratio rho across matrices, r = max(1, round(rho · min(d_out, d_in))),
    as in the paper's uniform-rank setting.

Protocol choices (logged): factorized matrices are W_Q/K/V/O (flattened heads), W_in,
W_out, W_U — same (d_out, d_in) orientations as gptq.py; W_E/W_pos stay dense (the paper
does not factorize embeddings; they are ~2% of params). Evaluation bakes W = A @ B back
into the TL weights; the parameter count is computed analytically from the ranks.
"""
from __future__ import annotations

import copy

import torch as t
from torch.nn.utils import parametrize
from transformer_lens import HookedTransformer

from compress.base import Quantizer

# name -> (to (d_out,d_in), back to TL shape) given TL tensor shape
def _to2d(name: str, w: t.Tensor) -> t.Tensor:
    last = name.split(".")[-1]
    if last in ("W_Q", "W_K", "W_V"):     # (h, d_model, d_head) -> (h*d_head, d_model)
        return w.permute(0, 2, 1).reshape(-1, w.shape[1])
    if last == "W_O":                      # (h, d_head, d_model) -> (d_model, h*d_head)
        return w.reshape(-1, w.shape[-1]).T
    return w.T                             # W_in/W_out/W_U: TL stores (d_in, d_out)

def _from2d(name: str, w2: t.Tensor, shape: t.Size) -> t.Tensor:
    last = name.split(".")[-1]
    if last in ("W_Q", "W_K", "W_V"):
        return w2.reshape(shape[0], shape[2], shape[1]).permute(0, 2, 1)
    if last == "W_O":
        return w2.T.reshape(shape)
    return w2.T


class _LowRank(t.nn.Module):
    def __init__(self, A: t.Tensor, B: t.Tensor, name: str, shape: t.Size):
        super().__init__()
        self.A, self.B = t.nn.Parameter(A), t.nn.Parameter(B)
        self.pname, self.shape = name, shape

    def forward(self, w: t.Tensor) -> t.Tensor:   # ignores w; the factors are the weights
        return _from2d(self.pname, self.A @ self.B, self.shape)


class FWSVD(Quantizer):
    name = "FWSVD"
    citation = "Hsu et al., ICLR 2022, arXiv:2207.00112"
    TARGETS = ("W_Q", "W_K", "W_V", "W_O", "W_in", "W_out", "W_U")

    def __init__(self, rank_ratio: float, weighted: bool = True, epochs: int = 0,
                 lr: float = 1e-4, batch_size: int = 128, n_fisher: int = 4096,
                 fisher_batch: int = 32, seed: int = 0):
        super().__init__(bits=16)  # bits unused; low-rank method
        assert 0 < rank_ratio <= 1
        self.rank_ratio, self.weighted, self.epochs = rank_ratio, weighted, epochs
        self.lr, self.batch_size, self.seed = lr, batch_size, seed
        self.n_fisher, self.fisher_batch = n_fisher, fisher_batch
        self.ranks: dict[str, int] = {}
        self.factors: dict[str, tuple[t.Tensor, t.Tensor]] = {}   # filled by quantize()
        if not weighted:
            self.name = "SVD"
            self.citation = "plain truncated SVD (paper's baseline)"

    @property
    def spec(self):
        s = super().spec
        s.extra = dict(rank_ratio=self.rank_ratio, weighted=self.weighted, epochs=self.epochs,
                       n_fisher=self.n_fisher, lr=self.lr)
        return s

    def _targets(self, model):
        return [(n, p) for n, p in model.named_parameters() if n.split(".")[-1] in self.TARGETS]

    def fisher(self, model: HookedTransformer, data: t.Tensor) -> dict[str, t.Tensor]:
        device = next(model.parameters()).device
        g = t.Generator().manual_seed(self.seed)
        idx = t.randperm(len(data), generator=g)[: self.n_fisher]
        names, params = zip(*self._targets(model))
        acc = [t.zeros_like(p) for p in params]
        n_batches = 0
        for bidx in idx.split(self.fisher_batch):
            batch = data[bidx].to(device)
            logits = model(batch[:, :-1])
            loss = t.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))
            grads = t.autograd.grad(loss, params)
            for a, gr in zip(acc, grads):
                a += gr.detach() ** 2
            n_batches += 1
        return {n: a / n_batches for n, a in zip(names, acc)}

    def params_after(self, model: HookedTransformer) -> int:
        total = 0
        fact = dict(self._targets(model))
        for n, p in model.named_parameters():
            if n in fact:
                d_out, d_in = _to2d(n, p.data).shape
                total += self.ranks[n] * (d_out + d_in)
            else:
                total += p.numel()
        return total

    @t.no_grad()
    def _factorize(self, name: str, w: t.Tensor, fish: t.Tensor | None):
        W2 = _to2d(name, w)
        d_out, d_in = W2.shape
        r = max(1, round(self.rank_ratio * min(d_out, d_in)))
        self.ranks[name] = r
        if self.weighted:
            d = _to2d(name, fish).sum(dim=1).clamp_min(1e-12).sqrt()
            U, S, Vh = t.linalg.svd((d[:, None] * W2).float(), full_matrices=False)
            A = (U[:, :r] * S[:r]) / d[:, None]
        else:
            U, S, Vh = t.linalg.svd(W2.float(), full_matrices=False)
            A = U[:, :r] * S[:r]
        return A, Vh[:r]

    def quantize(self, model: HookedTransformer, calib: t.Tensor | None = None) -> HookedTransformer:
        assert calib is not None, "FWSVD needs data (Fisher estimation; train split for fine-tuning)"
        model = copy.deepcopy(model)
        fisher = self.fisher(model, calib) if self.weighted else {}

        wrapped = []
        for name, p in self._targets(model):
            A, B = self._factorize(name, p.data, fisher.get(name))
            mod = model.get_submodule(name.rsplit(".", 1)[0])
            attr = name.rsplit(".", 1)[1]
            parametrize.register_parametrization(mod, attr, _LowRank(A, B, name, p.shape))
            wrapped.append((mod, attr))

        if self.epochs > 0:
            device = next(model.parameters()).device
            opt = t.optim.Adam(model.parameters(), lr=self.lr)
            g = t.Generator().manual_seed(self.seed)
            model.train()
            for _ in range(self.epochs):
                for idx in t.randperm(len(calib), generator=g).split(self.batch_size):
                    batch = calib[idx].to(device)
                    logits = model(batch[:, :-1])
                    loss = t.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))
                    opt.zero_grad(); loss.backward(); opt.step()

        for mod, attr in wrapped:
            lr_mod = mod.parametrizations[attr][0]
            self.factors[lr_mod.pname] = (lr_mod.A.detach().cpu().clone(), lr_mod.B.detach().cpu().clone())
            parametrize.remove_parametrizations(mod, attr, leave_parametrized=True)
        return model.eval()
