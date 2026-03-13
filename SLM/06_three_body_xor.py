"""
06_three_body_xor.py

Can a quadratic autoregressive model learn 3-input XOR?

Data: blocks of 4 — three random bits, then their sum mod 2.
Model: same absolute-indexing quadratic model from 02_train_full.py.

Spoiler: the quadratic logit can't represent 3-input XOR.
The 3-body term x_i * x_j * x_k is missing, and the sigmoid
(being monotonic) can't compensate.

Companion to version2.md §2.
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


class QuadraticAutoregressive(nn.Module):
    """Same model as in 02_train_full.py — absolute indexing, pairwise interactions."""

    def __init__(self, T):
        super().__init__()
        self.T = T
        self.bias = nn.Parameter(torch.zeros(T))
        self.W = nn.Parameter(torch.zeros(T, T))
        self.C = nn.Parameter(torch.zeros(T, T, T))

        linear_mask = torch.tril(torch.ones(T, T), diagonal=-1)
        self.register_buffer('linear_mask', linear_mask)

        idx = torch.arange(T)
        quad_mask = (
            (idx[None, :, None] < idx[None, None, :]) &
            (idx[None, None, :] < idx[:, None, None])
        ).float()
        self.register_buffer('quad_mask', quad_mask)

    def forward(self, x):
        B, T = x.shape
        W_masked = self.W * self.linear_mask
        linear = x @ W_masked.T

        xx = x.unsqueeze(-1) * x.unsqueeze(-2)
        C_masked = self.C * self.quad_mask
        quadratic = torch.einsum('bij, tij -> bt', xx, C_masked)

        return self.bias + linear + quadratic


def main():
    T = 12
    model = QuadraticAutoregressive(T)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-2)

    # 3 random + 1 deterministic per block → (3/4) ln 2
    theoretical_min = (3 / 4) * torch.log(torch.tensor(2.0)).item()
    print(f"T = {T} ({T // 4} blocks of 4)")
    print(f"Theoretical minimum loss: {theoretical_min:.4f}")
    print()

    for step in range(5001):
        x = sample_z2_blocks4(batch_size=512, T=T)
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, x, reduction='mean')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 1000 == 0:
            print(f"Step {step:4d} | Loss: {loss.item():.4f} | "
                  f"Gap: {loss.item() - theoretical_min:.4f}")

    # --- Check predictions at the first XOR position (position 3) ---
    print("\n--- Predictions at position 3 (XOR of 0,1,2) ---")
    with torch.no_grad():
        # All 8 input combinations
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

    # --- Key weights ---
    print("\n--- Key parameters at position 3 ---")
    with torch.no_grad():
        print(f"  bias[3]    = {model.bias[3].item():.2f}")
        print(f"  W[3,0]     = {model.W[3, 0].item():.2f}")
        print(f"  W[3,1]     = {model.W[3, 1].item():.2f}")
        print(f"  W[3,2]     = {model.W[3, 2].item():.2f}")
        print(f"  C[3,0,1]   = {model.C[3, 0, 1].item():.2f}")
        print(f"  C[3,0,2]   = {model.C[3, 0, 2].item():.2f}")
        print(f"  C[3,1,2]   = {model.C[3, 1, 2].item():.2f}")


if __name__ == '__main__':
    main()
