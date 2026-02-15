"""
01_linear_fails.py — Demonstrates that a linear autoregressive model cannot learn XOR.

A linear model (bias + W, no quadratic term C) predicts each bit via:
    P(x_t = 1 | x_{<t}) = σ(b_t + Σ_{i<t} W_{t,i} x_i)

Since XOR is not linearly separable, the model cannot capture the constraint
x_{t} = x_{t-2} ⊕ x_{t-1} at every third position. The loss plateaus well
above the theoretical minimum of (2/3) ln 2 ≈ 0.462.

Reference: z2_quadratic_tutorial.md §2
"""

import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Progress bar (no external deps)
# ---------------------------------------------------------------------------

def progress_bar(step, total, loss, extra="", width=30):
    frac = step / total
    filled = int(width * frac)
    bar = "█" * filled + "░" * (width - filled)
    sys.stderr.write(f"\r  {bar} {step:>5d}/{total} | loss {loss:.4f} {extra}")
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# Data & Model
# ---------------------------------------------------------------------------

def sample_z2_data(batch_size, T=12):
    """Sample sequences where every 3rd element is XOR of the previous two."""
    assert T % 3 == 0
    x = torch.zeros(batch_size, T)
    for block in range(T // 3):
        i = block * 3
        x[:, i]     = torch.randint(0, 2, (batch_size,)).float()
        x[:, i + 1] = torch.randint(0, 2, (batch_size,)).float()
        x[:, i + 2] = (x[:, i] + x[:, i + 1]) % 2  # XOR
    return x


class LinearAutoregressive(nn.Module):
    """Autoregressive model with only bias + linear weights (no quadratic term)."""
    def __init__(self, T):
        super().__init__()
        self.T = T
        self.bias = nn.Parameter(torch.zeros(T))
        self.W = nn.Parameter(torch.zeros(T, T))

        linear_mask = torch.tril(torch.ones(T, T), diagonal=-1)
        self.register_buffer('linear_mask', linear_mask)

    def forward(self, x):
        W_masked = self.W * self.linear_mask
        linear = x @ W_masked.T
        return self.bias + linear


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = 12
    steps = 5000
    model = LinearAutoregressive(T).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-2)

    theoretical_min = (2 / 3) * torch.log(torch.tensor(2.0)).item()
    ln2 = torch.log(torch.tensor(2.0)).item()
    n_params = sum(p.numel() for p in model.parameters())

    print("=" * 60)
    print("  LINEAR AUTOREGRESSIVE MODEL  (no quadratic term)")
    print("=" * 60)
    print(f"  Sequence length T     = {T}  ({T // 3} blocks of 3)")
    print(f"  Device                = {device}")
    print(f"  Parameters            = {n_params}")
    print(f"  Theoretical min loss  = {theoretical_min:.4f}  ((2/3) ln 2)")
    print(f"  Expected plateau      = {ln2:.4f}  (ln 2 — all positions predict 50/50)")
    print()

    # --- Training ---
    print("Training:")
    losses = []
    t0 = time.time()

    for step in range(steps + 1):
        x = sample_z2_data(batch_size=512, T=T).to(device)
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, x, reduction='mean')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if step % 50 == 0 or step == steps:
            progress_bar(step, steps, loss.item(),
                         extra=f"| gap {loss.item() - theoretical_min:+.4f}")

    elapsed = time.time() - t0
    sys.stderr.write("\n")
    print(f"  Done in {elapsed:.1f}s\n")

    # --- Loss trajectory (sampled) ---
    print("Loss trajectory:")
    for s in range(0, steps + 1, 500):
        l = losses[s]
        bar_len = int(max(0, (l - theoretical_min) * 80))
        bar = "▓" * bar_len
        print(f"  Step {s:5d} | {l:.4f} | {bar}")
    print()

    # --- XOR accuracy ---
    with torch.no_grad():
        x = sample_z2_data(1000, T).to(device)
        logits = model(x)
        preds = (logits[:, 2::3] > 0).float()
        targets = x[:, 2::3]
        xor_acc = (preds == targets).float().mean().item()

        # Per-position-type loss
        bce = F.binary_cross_entropy_with_logits(logits, x, reduction='none')
        random_pos_loss = bce[:, 0::3].mean().item()   # positions 0,3,6,...
        random_pos_loss2 = bce[:, 1::3].mean().item()   # positions 1,4,7,...
        xor_pos_loss = bce[:, 2::3].mean().item()        # positions 2,5,8,...

    print("Per-position-type loss breakdown:")
    print(f"  Positions 0 mod 3 (random a): {random_pos_loss:.4f}  (optimal: {ln2:.4f})")
    print(f"  Positions 1 mod 3 (random b): {random_pos_loss2:.4f}  (optimal: {ln2:.4f})")
    print(f"  Positions 2 mod 3 (XOR c):    {xor_pos_loss:.4f}  (optimal: 0.0000)")
    print()
    print(f"XOR prediction accuracy: {xor_acc:.4f}  (random chance = 0.50)")
    print()

    # --- Verdict ---
    final_loss = losses[-1]
    print("=" * 60)
    print(f"  RESULT: Loss stuck at ~{final_loss:.3f}  (ln 2 = {ln2:.3f})")
    print(f"  The linear model cannot capture XOR.")
    print(f"  Gap to theoretical min: {final_loss - theoretical_min:+.3f} nats")
    print("=" * 60)


if __name__ == "__main__":
    main()
