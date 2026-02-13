"""
04_exact_training.py — Exact enumeration training (no sampling noise).

For small T, we can enumerate all 2^T binary sequences, compute the true
distribution's probability for each, and compute the exact expected loss.
This eliminates sampling noise entirely, giving smooth convergence.

Reference: z2_quadratic_tutorial.md §7
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
    sys.stderr.write(f"\r  {bar} {step:>5d}/{total} | loss {loss:.6f} {extra}")
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# Model (same as 02_train_full.py)
# ---------------------------------------------------------------------------

class QuadraticAutoregressive(nn.Module):
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


# ---------------------------------------------------------------------------
# §7.2 True distribution
# ---------------------------------------------------------------------------

def true_log_probs(T):
    """Compute log P_true(x) for all 2^T binary sequences."""
    N = 2 ** T
    all_x = torch.zeros(N, T)
    for i in range(T):
        all_x[:, i] = (torch.arange(N) >> i) % 2

    # Check which sequences satisfy XOR constraint in every block
    valid = torch.ones(N, dtype=torch.bool)
    for block in range(T // 3):
        i = block * 3
        xor_ok = ((all_x[:, i] + all_x[:, i + 1]) % 2 == all_x[:, i + 2])
        valid &= xor_ok

    log_p = torch.full((N,), float('-inf'))
    log_p[valid] = -(T / 3) * torch.log(torch.tensor(4.0))
    return all_x, log_p


# ---------------------------------------------------------------------------
# §7.3 Exact expected loss
# ---------------------------------------------------------------------------

def exact_loss(model, T):
    """Compute the exact expected NLL under the true distribution."""
    device = next(model.parameters()).device
    all_x, true_log_p = true_log_probs(T)
    all_x = all_x.to(device)
    true_p = torch.exp(true_log_p).to(device)

    logits = model(all_x)

    # Per-position log P_model(x_t | x_{<t})
    log_p_model = -F.binary_cross_entropy_with_logits(
        logits, all_x, reduction='none'
    )

    # Total log P_model(x) = sum over positions
    log_p_model_total = log_p_model.sum(dim=1)

    # Expected NLL per position
    nll = -(true_p * log_p_model_total).sum() / T
    return nll


# ---------------------------------------------------------------------------
# §7.4 Training with exact loss
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = 9  # 3 blocks, 2^9 = 512 sequences
    steps = 2000
    model = QuadraticAutoregressive(T).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-2)

    theoretical_min = (2 / 3) * torch.log(torch.tensor(2.0)).item()
    n_params = sum(p.numel() for p in model.parameters())

    # Count valid sequences
    _, log_p = true_log_probs(T)
    n_valid = (log_p > float('-inf')).sum().item()
    n_total = 2 ** T

    print("=" * 60)
    print("  EXACT ENUMERATION TRAINING  (no sampling)")
    print("=" * 60)
    print(f"  Sequence length T     = {T}  ({T // 3} blocks of 3)")
    print(f"  Total sequences       = {n_total}  (2^{T})")
    print(f"  Valid sequences       = {n_valid}  (satisfy XOR in all blocks)")
    print(f"  Device                = {device}")
    print(f"  Parameters            = {n_params}")
    print(f"  Theoretical min loss  = {theoretical_min:.6f}  ((2/3) ln 2)")
    print()

    # --- Training ---
    print("Training (exact gradients, no sampling noise):")
    losses = []
    t0 = time.time()

    for step in range(steps + 1):
        loss = exact_loss(model, T)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if step % 20 == 0 or step == steps:
            progress_bar(step, steps, loss.item(),
                         extra=f"| gap {loss.item() - theoretical_min:+.6f}")

    elapsed = time.time() - t0
    sys.stderr.write("\n")
    print(f"  Done in {elapsed:.1f}s\n")

    # --- Loss trajectory ---
    print("Loss trajectory (note: perfectly smooth — no sampling noise):")
    for s in range(0, steps + 1, 200):
        l = losses[s]
        bar_len = int(max(0, (l - theoretical_min) * 120))
        bar = "▓" * bar_len
        print(f"  Step {s:5d} | {l:.6f} | {bar}")
    print()

    # --- Compare: how close to theoretical min? ---
    final_loss = losses[-1]
    print("Convergence analysis:")
    print(f"  Initial loss:     {losses[0]:.6f}  (= ln 2, uniform predictions)")
    print(f"  Loss at step 500: {losses[500]:.6f}")
    print(f"  Loss at step 1k:  {losses[1000]:.6f}")
    print(f"  Final loss:       {final_loss:.6f}")
    print(f"  Theoretical min:  {theoretical_min:.6f}")
    print(f"  Final gap:        {final_loss - theoretical_min:.6f}")
    print()

    # --- Inspect parameters at XOR positions ---
    print("Learned XOR weights:")
    with torch.no_grad():
        for block in range(T // 3):
            t = block * 3 + 2
            i, j = block * 3, block * 3 + 1
            b = model.bias[t].item()
            w_i = model.W[t, i].item()
            w_j = model.W[t, j].item()
            c = model.C[t, i, j].item()
            print(f"  Position {t}: bias={b:+.2f}, W[{i}]={w_i:+.2f}, "
                  f"W[{j}]={w_j:+.2f}, C[{i},{j}]={c:+.2f}")
    print()

    print("=" * 60)
    print(f"  Smooth convergence to {final_loss:.4f} — no sampling noise.")
    print("=" * 60)


if __name__ == "__main__":
    main()
