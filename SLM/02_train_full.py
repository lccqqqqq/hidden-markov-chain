"""
02_train_full.py — Core demo: train a quadratic autoregressive model on Z₂ data.

Covers:
  §5.1  Data generation (sample_z2_data)
  §5.2  QuadraticAutoregressive model class
  §5.3  Training loop with loss printing
  §5.4  Inspecting learned parameters
  §8.2  XOR accuracy verification

Reference: z2_quadratic_tutorial.md §5
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
# §5.1 Data generation
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


# ---------------------------------------------------------------------------
# §5.2 Model class
# ---------------------------------------------------------------------------

class QuadraticAutoregressive(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T
        self.bias = nn.Parameter(torch.zeros(T))
        self.W = nn.Parameter(torch.zeros(T, T))       # linear weights
        self.C = nn.Parameter(torch.zeros(T, T, T))     # quadratic weights

        # Linear mask: W[t, i] is active only if i < t
        linear_mask = torch.tril(torch.ones(T, T), diagonal=-1)
        self.register_buffer('linear_mask', linear_mask)

        # Quadratic mask: C[t, i, j] is active only if i < j < t
        idx = torch.arange(T)
        quad_mask = (
            (idx[None, :, None] < idx[None, None, :]) &   # i < j
            (idx[None, None, :] < idx[:, None, None])      # j < t
        ).float()
        self.register_buffer('quad_mask', quad_mask)

    def forward(self, x):
        B, T = x.shape

        # Linear contribution
        W_masked = self.W * self.linear_mask
        linear = x @ W_masked.T

        # Quadratic contribution
        xx = x.unsqueeze(-1) * x.unsqueeze(-2)
        C_masked = self.C * self.quad_mask
        quadratic = torch.einsum('bij, tij -> bt', xx, C_masked)

        logits = self.bias + linear + quadratic
        return logits


# ---------------------------------------------------------------------------
# §5.3 Training loop
# ---------------------------------------------------------------------------

def train(model, T, steps=5000, batch_size=512, lr=3e-2):
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    theoretical_min = (2 / 3) * torch.log(torch.tensor(2.0)).item()

    print("Training:")
    losses = []
    t0 = time.time()

    for step in range(steps + 1):
        x = sample_z2_data(batch_size=batch_size, T=T).to(device)
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

    # Loss trajectory
    print("Loss trajectory:")
    for s in range(0, steps + 1, 500):
        l = losses[s]
        bar_len = int(max(0, (l - theoretical_min) * 80))
        bar = "▓" * bar_len
        print(f"  Step {s:5d} | {l:.4f} | {bar}")
    print()

    return theoretical_min, losses


# ---------------------------------------------------------------------------
# §5.4 Inspecting learned parameters
# ---------------------------------------------------------------------------

def inspect_parameters(model, T):
    print("Learned parameters")
    print("-" * 50)
    with torch.no_grad():
        # Biases at "random" positions
        print("\n  Biases at random positions (expect ~0):")
        for t in range(T):
            if t % 3 != 2:
                v = model.bias[t].item()
                print(f"    bias[{t:2d}] = {v:+.3f}")

        # XOR positions: key weights
        print("\n  XOR position weights (expect b ~ -L, W ~ 2L, C ~ -4L):")
        for block in range(T // 3):
            t = block * 3 + 2
            i, j = block * 3, block * 3 + 1
            b = model.bias[t].item()
            w_i = model.W[t, i].item()
            w_j = model.W[t, j].item()
            c = model.C[t, i, j].item()
            L = -b  # infer L from bias
            print(f"\n    Position {t} (XOR of {i},{j}):  L ~ {L:.1f}")
            print(f"      bias[{t}]      = {b:+8.2f}   (theory: -L = {-L:.2f})")
            print(f"      W[{t},{i}]      = {w_i:+8.2f}   (theory: 2L = {2*L:.2f})")
            print(f"      W[{t},{j}]      = {w_j:+8.2f}   (theory: 2L = {2*L:.2f})")
            print(f"      C[{t},{i},{j}]    = {c:+8.2f}   (theory: -4L = {-4*L:.2f})")

        # Check cross-block weights are near zero
        print("\n  Cross-block weights (should be ~0):")
        max_cross = 0.0
        for block in range(1, T // 3):
            t = block * 3 + 2
            for prev_block in range(block):
                for pi in range(prev_block * 3, prev_block * 3 + 2):
                    w = abs(model.W[t, pi].item())
                    max_cross = max(max_cross, w)
        print(f"    Max |W| across block boundaries: {max_cross:.4f}")
    print()


# ---------------------------------------------------------------------------
# §8.2 XOR accuracy verification
# ---------------------------------------------------------------------------

def verify_xor_accuracy(model, T, n_samples=10000):
    print("Verification")
    print("-" * 50)
    ln2 = torch.log(torch.tensor(2.0)).item()
    theoretical_min = (2 / 3) * ln2

    with torch.no_grad():
        device = next(model.parameters()).device
        x = sample_z2_data(n_samples, T).to(device)
        logits = model(x)

        # XOR accuracy
        preds = (logits[:, 2::3] > 0).float()
        targets = x[:, 2::3]
        xor_acc = (preds == targets).float().mean().item()

        # Per-position-type loss
        bce = F.binary_cross_entropy_with_logits(logits, x, reduction='none')
        loss_a = bce[:, 0::3].mean().item()
        loss_b = bce[:, 1::3].mean().item()
        loss_c = bce[:, 2::3].mean().item()
        total = bce.mean().item()

    print(f"\n  Per-position-type loss (over {n_samples} samples):")
    print(f"    Positions 0 mod 3 (random a): {loss_a:.4f}  (optimal: {ln2:.4f})")
    print(f"    Positions 1 mod 3 (random b): {loss_b:.4f}  (optimal: {ln2:.4f})")
    print(f"    Positions 2 mod 3 (XOR c):    {loss_c:.4f}  (optimal: 0.0000)")
    print(f"    Overall:                      {total:.4f}  (optimal: {theoretical_min:.4f})")
    print()
    print(f"  XOR prediction accuracy: {xor_acc:.4f}  (expect ~1.0)")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = 12
    theoretical_min = (2 / 3) * torch.log(torch.tensor(2.0)).item()
    model = QuadraticAutoregressive(T).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    print("=" * 60)
    print("  QUADRATIC AUTOREGRESSIVE MODEL  (full)")
    print("=" * 60)
    print(f"  Sequence length T     = {T}  ({T // 3} blocks of 3)")
    print(f"  Device                = {device}")
    print(f"  Parameters            = {n_params}  (b: {T}, W: {T*T}, C: {T**3})")
    print(f"  Theoretical min loss  = {theoretical_min:.4f}  ((2/3) ln 2)")
    print()

    train(model, T)
    inspect_parameters(model, T)
    verify_xor_accuracy(model, T)

    print("=" * 60)
    print("  Done. The model learned XOR from data using quadratic features.")
    print("=" * 60)


if __name__ == "__main__":
    main()
