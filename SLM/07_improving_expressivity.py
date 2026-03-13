"""
07_improving_expressivity.py

Two approaches to learn 3-input XOR (which quadratic models can't):
1. Cubic model: add D[t,i,j,k] x_i x_j x_k terms directly
2. Stacked model: two quadratic layers with sigmoid between them

The cubic model adds the missing 3-body term explicitly.
The stacked model decomposes 3-input XOR into two stages of 2-input XOR:
  layer 1 learns h_2 ≈ x_0 ⊕ x_1, then layer 2 learns h_2 ⊕ x_2
  via the cross-term x_i * h_j.

Companion to version2.md §3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_z2_blocks4(batch_size, T=12):
    """Blocks of 4: three random bits, 4th is their XOR."""
    assert T % 4 == 0
    x = torch.zeros(batch_size, T)
    for block in range(T // 4):
        i = block * 4
        x[:, i]     = torch.randint(0, 2, (batch_size,)).float()
        x[:, i + 1] = torch.randint(0, 2, (batch_size,)).float()
        x[:, i + 2] = torch.randint(0, 2, (batch_size,)).float()
        x[:, i + 3] = (x[:, i] + x[:, i + 1] + x[:, i + 2]) % 2
    return x


# ---------------------------------------------------------------------------
#  Approach 1: Cubic model (add 3-body terms)
# ---------------------------------------------------------------------------

class CubicAutoregressive(nn.Module):
    """
    Autoregressive model with up to 3-body interactions.

    Logit for position t:
        Δ_t = b_t + Σ_{i<t} W[t,i] x_i
                   + Σ_{i<j<t} C[t,i,j] x_i x_j
                   + Σ_{i<j<k<t} D[t,i,j,k] x_i x_j x_k
    """

    def __init__(self, T):
        super().__init__()
        self.T = T
        self.bias = nn.Parameter(torch.zeros(T))
        self.W = nn.Parameter(torch.zeros(T, T))
        self.C = nn.Parameter(torch.zeros(T, T, T))
        self.D = nn.Parameter(torch.zeros(T, T, T, T))

        # Linear mask: i < t
        linear_mask = torch.tril(torch.ones(T, T), diagonal=-1)
        self.register_buffer('linear_mask', linear_mask)

        idx = torch.arange(T)

        # Quadratic mask: i < j < t
        quad_mask = (
            (idx[None, :, None] < idx[None, None, :]) &
            (idx[None, None, :] < idx[:, None, None])
        ).float()
        self.register_buffer('quad_mask', quad_mask)

        # Cubic mask: i < j < k < t
        i_ = idx[None, :, None, None]
        j_ = idx[None, None, :, None]
        k_ = idx[None, None, None, :]
        t_ = idx[:, None, None, None]
        cubic_mask = ((i_ < j_) & (j_ < k_) & (k_ < t_)).float()
        self.register_buffer('cubic_mask', cubic_mask)

    def forward(self, x):
        B, T = x.shape

        # Linear
        W_m = self.W * self.linear_mask
        linear = x @ W_m.T

        # Quadratic
        xx = x.unsqueeze(-1) * x.unsqueeze(-2)          # (B, T, T)
        C_m = self.C * self.quad_mask
        quadratic = torch.einsum('bij, tij -> bt', xx, C_m)

        # Cubic
        x_i = x[:, :, None, None]                        # (B, T, 1, 1)
        x_j = x[:, None, :, None]                        # (B, 1, T, 1)
        x_k = x[:, None, None, :]                        # (B, 1, 1, T)
        xxx = x_i * x_j * x_k                            # (B, T, T, T)
        D_m = self.D * self.cubic_mask
        cubic = torch.einsum('bijk, tijk -> bt', xxx, D_m)

        return self.bias + linear + quadratic + cubic


# ---------------------------------------------------------------------------
#  Approach 2: Stacked model (two quadratic layers)
# ---------------------------------------------------------------------------

class StackedQuadraticAutoregressive(nn.Module):
    """
    Two-layer autoregressive model: quadratic → sigmoid → quadratic.

    Layer 1: h_t = σ(quadratic in x_{<t})
    Layer 2: logit_t = f(x_{<t}, h_{<t})  with quadratic + cross terms

    The cross-term x_i * h_j lets layer 2 combine raw inputs with
    layer 1's learned features, effectively capturing higher-order
    interactions without explicit cubic parameters.
    """

    def __init__(self, T):
        super().__init__()
        self.T = T

        # --- Layer 1: x → h ---
        self.bias1 = nn.Parameter(torch.zeros(T))
        self.W1 = nn.Parameter(torch.zeros(T, T))
        self.C1 = nn.Parameter(torch.zeros(T, T, T))

        # --- Layer 2: (x, h) → logits ---
        self.bias2 = nn.Parameter(torch.zeros(T))
        self.W2_x = nn.Parameter(torch.zeros(T, T))     # linear in x
        self.W2_h = nn.Parameter(torch.zeros(T, T))     # linear in h
        self.C2_xx = nn.Parameter(torch.zeros(T, T, T)) # x_i * x_j
        self.C2_hh = nn.Parameter(torch.zeros(T, T, T)) # h_i * h_j
        self.C2_xh = nn.Parameter(torch.zeros(T, T, T)) # x_i * h_j (cross)

        # --- Masks ---
        linear_mask = torch.tril(torch.ones(T, T), diagonal=-1)  # i < t
        self.register_buffer('linear_mask', linear_mask)

        idx = torch.arange(T)
        quad_mask = (
            (idx[None, :, None] < idx[None, None, :]) &
            (idx[None, None, :] < idx[:, None, None])
        ).float()
        self.register_buffer('quad_mask', quad_mask)

        # Cross mask: i < t AND j < t (no ordering between i,j)
        cross_mask = (
            (idx[None, :, None] < idx[:, None, None]) &
            (idx[None, None, :] < idx[:, None, None])
        ).float()
        self.register_buffer('cross_mask', cross_mask)

    def forward(self, x):
        B, T = x.shape

        # --- Layer 1: compute h = σ(quadratic in x) ---
        W1_m = self.W1 * self.linear_mask
        lin1 = x @ W1_m.T

        xx = x.unsqueeze(-1) * x.unsqueeze(-2)
        C1_m = self.C1 * self.quad_mask
        quad1 = torch.einsum('bij, tij -> bt', xx, C1_m)

        h = torch.sigmoid(self.bias1 + lin1 + quad1)     # (B, T)

        # --- Layer 2: compute logits from (x, h) ---
        # Linear terms
        W2x_m = self.W2_x * self.linear_mask
        W2h_m = self.W2_h * self.linear_mask
        lin2 = x @ W2x_m.T + h @ W2h_m.T

        # Same-type quadratics
        hh = h.unsqueeze(-1) * h.unsqueeze(-2)
        C2xx_m = self.C2_xx * self.quad_mask
        C2hh_m = self.C2_hh * self.quad_mask
        quad2_xx = torch.einsum('bij, tij -> bt', xx, C2xx_m)
        quad2_hh = torch.einsum('bij, tij -> bt', hh, C2hh_m)

        # Cross quadratic: x_i * h_j (the key new ingredient)
        xh = x.unsqueeze(-1) * h.unsqueeze(-2)           # (B, T, T): xh[b,i,j] = x_i * h_j
        C2xh_m = self.C2_xh * self.cross_mask
        cross = torch.einsum('bij, tij -> bt', xh, C2xh_m)

        logits = self.bias2 + lin2 + quad2_xx + quad2_hh + cross
        return logits


# ---------------------------------------------------------------------------
#  Training
# ---------------------------------------------------------------------------

def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_active_params(model):
    """Count parameters that aren't masked to zero."""
    total = 0
    for name, p in model.named_parameters():
        if 'bias' in name or 'W' in name or 'C' in name or 'D' in name:
            # Find the corresponding mask
            mask_name = None
            if 'D' in name:
                mask_name = 'cubic_mask'
            elif 'C' in name:
                if 'xh' in name:
                    mask_name = 'cross_mask'
                else:
                    mask_name = 'quad_mask'
            elif 'W' in name:
                mask_name = 'linear_mask'

            if mask_name and hasattr(model, mask_name):
                mask = getattr(model, mask_name)
                total += int(mask.sum().item())
            else:
                total += p.numel()
    return total


def train_model(model, name, T=12, steps=8000, lr=3e-2):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    theoretical_min = (3 / 4) * torch.log(torch.tensor(2.0)).item()

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Total tensor entries: {count_params(model)}")
    print(f"  Theoretical minimum: {theoretical_min:.4f}")
    print(f"{'='*60}")

    for step in range(steps + 1):
        x = sample_z2_blocks4(batch_size=512, T=T)
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, x, reduction='mean')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 2000 == 0:
            print(f"Step {step:5d} | Loss: {loss.item():.4f} | "
                  f"Gap: {loss.item() - theoretical_min:.4f}")

    # Check predictions at position 3
    print(f"\n--- {name}: predictions at position 3 ---")
    with torch.no_grad():
        inputs = []
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    xor = (a + b + c) % 2
                    inputs.append([a, b, c, xor] * (T // 4))
        test = torch.tensor(inputs, dtype=torch.float)
        logits = model(test)
        probs = torch.sigmoid(logits[:, 3])
        for i, (a, b, c) in enumerate(
            [(a, b, c) for a in range(2) for b in range(2) for c in range(2)]
        ):
            target = (a + b + c) % 2
            print(f"  ({a},{b},{c}) → P(x₃=1) = {probs[i].item():.4f}  "
                  f"(target: {target})")

    return model


def main():
    T = 12

    # --- Approach 1: Cubic model ---
    cubic = CubicAutoregressive(T)
    train_model(cubic, "Cubic model (3-body terms)", T=T)

    # --- Approach 2: Stacked model ---
    stacked = StackedQuadraticAutoregressive(T)
    train_model(stacked, "Stacked model (2 quadratic layers)", T=T)


if __name__ == '__main__':
    main()
