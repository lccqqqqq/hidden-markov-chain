"""
05_relative_indexing.py

Quadratic autoregressive model with relative (translation-invariant) indexing.
Trained on Z₂ data: blocks of (a, b, a⊕b).

Companion to version2.md.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_z2_data(batch_size, T=12):
    """Sample sequences where every 3rd element is XOR of the previous two."""
    assert T % 3 == 0
    x = torch.zeros(batch_size, T)
    for block in range(T // 3):
        i = block * 3
        x[:, i]     = torch.randint(0, 2, (batch_size,)).float()
        x[:, i + 1] = torch.randint(0, 2, (batch_size,)).float()
        x[:, i + 2] = (x[:, i] + x[:, i + 1]) % 2
    return x


class RelativeQuadraticAutoregressive(nn.Module):
    """
    Autoregressive model with translation-invariant weights.

    Parameters are indexed by relative offset (lag) rather than
    absolute position. The same weights are reused at every position.

    Logit for position t:
        Δ_t = b + Σ_{δ=1}^{K} W[δ] · x_{t-δ}
                + Σ_{δ1>δ2≥1} C[δ1,δ2] · x_{t-δ1} · x_{t-δ2}

    where K = max_lag and sums only include terms where t-δ ≥ 0.
    """

    def __init__(self, max_lag=2):
        super().__init__()
        self.max_lag = max_lag

        self.bias = nn.Parameter(torch.zeros(1))
        self.W = nn.Parameter(torch.zeros(max_lag))             # W[δ-1] for δ=1..K
        self.C = nn.Parameter(torch.zeros(max_lag, max_lag))    # C[δ1-1, δ2-1]

        # Mask: C[a,b] active only when a > b (i.e., δ1 > δ2)
        mask = torch.tril(torch.ones(max_lag, max_lag), diagonal=-1)
        self.register_buffer('quad_mask', mask)

    def forward(self, x):
        """
        x: (B, T) binary tensor
        returns: logits (B, T)
        """
        B, T = x.shape
        logits = self.bias.expand(B, T).clone()

        # Linear: W[δ] · x_{t-δ}
        for delta in range(1, min(self.max_lag, T) + 1):
            logits[:, delta:] += self.W[delta - 1] * x[:, :T - delta]

        # Quadratic: C[δ1, δ2] · x_{t-δ1} · x_{t-δ2}  for δ1 > δ2
        C_masked = self.C * self.quad_mask
        for d1 in range(2, min(self.max_lag, T) + 1):
            for d2 in range(1, d1):
                t_start = d1
                if t_start >= T:
                    continue
                x_d1 = x[:, t_start - d1 : T - d1]   # x_{t-d1}
                x_d2 = x[:, t_start - d2 : T - d2]   # x_{t-d2}
                logits[:, t_start:] += C_masked[d1 - 1, d2 - 1] * x_d1 * x_d2

        return logits


def main():
    T = 12
    max_lag = 2
    model = RelativeQuadraticAutoregressive(max_lag=max_lag)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-2)

    theoretical_min = (2/3) * torch.log(torch.tensor(2.0)).item()

    n_quad = max_lag * (max_lag - 1) // 2
    n_params = 1 + max_lag + n_quad
    print(f"T = {T}, max_lag = {max_lag}")
    print(f"Parameters: {n_params}  (bias: 1, W: {max_lag}, C: {n_quad})")
    print(f"Theoretical minimum loss: {theoretical_min:.4f}")
    print()

    for step in range(5001):
        x = sample_z2_data(batch_size=512, T=T)
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, x, reduction='mean')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 1000 == 0:
            print(f"Step {step:4d} | Loss: {loss.item():.4f} | Gap: {loss.item() - theoretical_min:.4f}")

    # --- Inspect parameters ---
    print("\n--- Learned parameters ---")
    print(f"bias       = {model.bias.item():.4f}")
    for d in range(max_lag):
        print(f"W[δ={d+1}]    = {model.W[d].item():.4f}")
    C_masked = model.C * model.quad_mask
    for d1 in range(2, max_lag + 1):
        for d2 in range(1, d1):
            print(f"C[δ1={d1},δ2={d2}] = {C_masked[d1-1, d2-1].item():.4f}")

    # --- Check XOR predictions at position 2 ---
    print("\n--- XOR predictions (position 2) ---")
    with torch.no_grad():
        test_seqs = torch.tensor([
            [0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]
        ], dtype=torch.float)
        logits = model(test_seqs)
        probs = torch.sigmoid(logits[:, 2])
        for i in range(4):
            a, b, c = test_seqs[i].tolist()
            print(f"  ({a:.0f}, {b:.0f}) → P(x₂=1) = {probs[i].item():.4f}  (target: {c:.0f})")


if __name__ == '__main__':
    main()
