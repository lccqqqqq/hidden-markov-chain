"""
08_extrapolation.py

Test cubic and stacked models on higher-order XOR:
- 4-input XOR (blocks of 5: four random bits + their sum mod 2)
- 5-input XOR (blocks of 6: five random bits + their sum mod 2)

The cubic model (3-body terms) should fail at 4-input XOR (needs 4-body).
The stacked 2-layer model may succeed at 4-input XOR (h*h gives effective
order 4) but likely fails at 5-input XOR.

Companion to version2.md §4.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
#  Data
# ---------------------------------------------------------------------------

def sample_z2_data(batch_size, n_inputs, T):
    """Blocks of (n_inputs + 1): n_inputs random bits, then their XOR."""
    block_size = n_inputs + 1
    assert T % block_size == 0
    x = torch.zeros(batch_size, T)
    for block in range(T // block_size):
        base = block * block_size
        for k in range(n_inputs):
            x[:, base + k] = torch.randint(0, 2, (batch_size,)).float()
        x[:, base + n_inputs] = sum(x[:, base + k] for k in range(n_inputs)) % 2
    return x


# ---------------------------------------------------------------------------
#  Models (same as 07_improving_expressivity.py)
# ---------------------------------------------------------------------------

class CubicAutoregressive(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T
        self.bias = nn.Parameter(torch.zeros(T))
        self.W = nn.Parameter(torch.zeros(T, T))
        self.C = nn.Parameter(torch.zeros(T, T, T))
        self.D = nn.Parameter(torch.zeros(T, T, T, T))

        linear_mask = torch.tril(torch.ones(T, T), diagonal=-1)
        self.register_buffer('linear_mask', linear_mask)

        idx = torch.arange(T)
        quad_mask = (
            (idx[None, :, None] < idx[None, None, :]) &
            (idx[None, None, :] < idx[:, None, None])
        ).float()
        self.register_buffer('quad_mask', quad_mask)

        i_ = idx[None, :, None, None]
        j_ = idx[None, None, :, None]
        k_ = idx[None, None, None, :]
        t_ = idx[:, None, None, None]
        cubic_mask = ((i_ < j_) & (j_ < k_) & (k_ < t_)).float()
        self.register_buffer('cubic_mask', cubic_mask)

    def forward(self, x):
        B, T = x.shape
        W_m = self.W * self.linear_mask
        linear = x @ W_m.T

        xx = x.unsqueeze(-1) * x.unsqueeze(-2)
        C_m = self.C * self.quad_mask
        quadratic = torch.einsum('bij, tij -> bt', xx, C_m)

        x_i = x[:, :, None, None]
        x_j = x[:, None, :, None]
        x_k = x[:, None, None, :]
        xxx = x_i * x_j * x_k
        D_m = self.D * self.cubic_mask
        cubic = torch.einsum('bijk, tijk -> bt', xxx, D_m)

        return self.bias + linear + quadratic + cubic


class StackedQuadraticAutoregressive(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

        self.bias1 = nn.Parameter(torch.zeros(T))
        self.W1 = nn.Parameter(torch.zeros(T, T))
        self.C1 = nn.Parameter(torch.zeros(T, T, T))

        self.bias2 = nn.Parameter(torch.zeros(T))
        self.W2_x = nn.Parameter(torch.zeros(T, T))
        self.W2_h = nn.Parameter(torch.zeros(T, T))
        self.C2_xx = nn.Parameter(torch.zeros(T, T, T))
        self.C2_hh = nn.Parameter(torch.zeros(T, T, T))
        self.C2_xh = nn.Parameter(torch.zeros(T, T, T))

        linear_mask = torch.tril(torch.ones(T, T), diagonal=-1)
        self.register_buffer('linear_mask', linear_mask)

        idx = torch.arange(T)
        quad_mask = (
            (idx[None, :, None] < idx[None, None, :]) &
            (idx[None, None, :] < idx[:, None, None])
        ).float()
        self.register_buffer('quad_mask', quad_mask)

        cross_mask = (
            (idx[None, :, None] < idx[:, None, None]) &
            (idx[None, None, :] < idx[:, None, None])
        ).float()
        self.register_buffer('cross_mask', cross_mask)

    def forward(self, x):
        B, T = x.shape

        W1_m = self.W1 * self.linear_mask
        lin1 = x @ W1_m.T
        xx = x.unsqueeze(-1) * x.unsqueeze(-2)
        C1_m = self.C1 * self.quad_mask
        quad1 = torch.einsum('bij, tij -> bt', xx, C1_m)
        h = torch.sigmoid(self.bias1 + lin1 + quad1)

        W2x_m = self.W2_x * self.linear_mask
        W2h_m = self.W2_h * self.linear_mask
        lin2 = x @ W2x_m.T + h @ W2h_m.T

        hh = h.unsqueeze(-1) * h.unsqueeze(-2)
        C2xx_m = self.C2_xx * self.quad_mask
        C2hh_m = self.C2_hh * self.quad_mask
        quad2_xx = torch.einsum('bij, tij -> bt', xx, C2xx_m)
        quad2_hh = torch.einsum('bij, tij -> bt', hh, C2hh_m)

        xh = x.unsqueeze(-1) * h.unsqueeze(-2)
        C2xh_m = self.C2_xh * self.cross_mask
        cross = torch.einsum('bij, tij -> bt', xh, C2xh_m)

        return self.bias2 + lin2 + quad2_xx + quad2_hh + cross


# ---------------------------------------------------------------------------
#  Training & evaluation
# ---------------------------------------------------------------------------

def train_and_test(model, name, n_inputs, T, steps=8000, lr=3e-2):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    block_size = n_inputs + 1
    theoretical_min = (n_inputs / block_size) * torch.log(torch.tensor(2.0)).item()

    print(f"\n  {name}")
    print(f"  Theoretical minimum: {theoretical_min:.4f}")

    for step in range(steps + 1):
        x = sample_z2_data(batch_size=512, n_inputs=n_inputs, T=T)
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, x, reduction='mean')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 2000 == 0:
            print(f"  Step {step:5d} | Loss: {loss.item():.4f} | "
                  f"Gap: {loss.item() - theoretical_min:.4f}")

    # Check predictions at the first XOR position
    xor_pos = n_inputs
    with torch.no_grad():
        n_combos = 2 ** n_inputs
        inputs = []
        for val in range(n_combos):
            bits = [(val >> k) & 1 for k in range(n_inputs)]
            xor_bit = sum(bits) % 2
            block = bits + [xor_bit]
            seq = (block * ((T // block_size) + 1))[:T]
            inputs.append(seq)
        test = torch.tensor(inputs, dtype=torch.float)
        logits = model(test)
        probs = torch.sigmoid(logits[:, xor_pos])
        preds = (logits[:, xor_pos] > 0).float()
        targets = test[:, xor_pos]
        accuracy = (preds == targets).float().mean().item()

        # Print first 8 examples
        for i in range(min(8, n_combos)):
            bits = [int(inputs[i][k]) for k in range(n_inputs)]
            target = int(inputs[i][xor_pos])
            print(f"    {tuple(bits)} → P=1: {probs[i].item():.4f} (target: {target})")
        if n_combos > 8:
            print(f"    ... ({n_combos - 8} more cases)")
        print(f"  Accuracy: {accuracy:.2%}")

    return accuracy


def main():
    for n_inputs, T in [(4, 10), (5, 12)]:
        print(f"\n{'='*60}")
        print(f"  {n_inputs}-input XOR  (blocks of {n_inputs+1}, T={T})")
        print(f"{'='*60}")

        cubic = CubicAutoregressive(T)
        train_and_test(cubic, "Cubic (3-body terms)", n_inputs, T)

        stacked = StackedQuadraticAutoregressive(T)
        train_and_test(stacked, "Stacked 2-layer quadratic", n_inputs, T)


if __name__ == '__main__':
    main()
