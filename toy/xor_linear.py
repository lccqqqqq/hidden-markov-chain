"""
Toy AND Linear Model — Forward vs Backward Complexity

Forward: (a, b) -> c = a AND b  (deterministic)
Backward: (c, b) -> a           (stochastic when c=0, b=0)

Optimal cross-entropy losses:
  Forward:  0 nats (deterministic, approached asymptotically)
  Backward: (1/2) ln(2) ≈ 0.347 nats
"""

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

# Data: all 4 tuples (a, b, c=a AND b)
a_vals = np.array([0, 0, 1, 1])
b_vals = np.array([0, 1, 0, 1])
c_vals = np.array([0, 0, 0, 1])

# Forward: X=(a,b), y=c
X_fwd = np.stack([a_vals, b_vals], axis=1).astype(np.float64)  # (4, 2)
y_fwd = c_vals  # (4,)

# Backward: X=(c,b), y=a
X_bwd = np.stack([c_vals, b_vals], axis=1).astype(np.float64)  # (4, 2)
y_bwd = a_vals  # (4,)

# Analytical optimal losses
optimal_fwd = 0.0
optimal_bwd = 0.5 * np.log(2)  # ≈ 0.347 nats
print(f"Analytical optimal losses:")
print(f"  Forward:  {optimal_fwd:.6f} nats")
print(f"  Backward: {optimal_bwd:.6f} nats")
print()


def softmax(logits):
    """Stable softmax over last axis."""
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def cross_entropy_loss(logits, y):
    """Mean cross-entropy loss. logits: (N, C), y: (N,) integer labels."""
    log_probs = logits - np.log(np.exp(logits - logits.max(axis=-1, keepdims=True)).sum(axis=-1, keepdims=True)) - logits.max(axis=-1, keepdims=True) + logits.max(axis=-1, keepdims=True)
    # Simpler: use log-sum-exp
    lse = logits.max(axis=-1) + np.log(np.exp(logits - logits.max(axis=-1, keepdims=True)).sum(axis=-1))
    correct_logits = logits[np.arange(len(y)), y]
    return np.mean(lse - correct_logits)


def train(X, y, n_steps=5000, lr=0.1):
    """Train a 2-class linear model with SGD (full-batch). Returns W, b, losses."""
    np.random.seed(42)
    N, D = X.shape
    C = 2  # binary classification
    W = np.random.randn(D, C) * 0.01  # (2, 2)
    b = np.zeros(C)                     # (2,)

    losses = []
    for step in range(n_steps):
        # Forward
        logits = X @ W + b  # (N, C)
        probs = softmax(logits)  # (N, C)

        # Loss
        loss = cross_entropy_loss(logits, y)
        losses.append(loss)

        # Gradient of cross-entropy w.r.t. logits: probs - one_hot(y)
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(N), y] = 1.0
        dlogits = (probs - one_hot) / N  # (N, C)

        # Gradients
        dW = X.T @ dlogits  # (D, C)
        db = dlogits.sum(axis=0)  # (C,)

        # Update
        W -= lr * dW
        b -= lr * db

    return W, b, losses


# Train both (more steps for clean power-law tails)
N_STEPS = 20000
print("Training forward model (a,b) -> c ...")
W_fwd, b_fwd, losses_fwd = train(X_fwd, y_fwd, n_steps=N_STEPS, lr=1.0)
print(f"  Final loss: {losses_fwd[-1]:.6f}")
print(f"  W = {W_fwd}")
print(f"  b = {b_fwd}")
print()

print("Training backward model (c,b) -> a ...")
W_bwd, b_bwd, losses_bwd = train(X_bwd, y_bwd, n_steps=N_STEPS, lr=1.0)
print(f"  Final loss: {losses_bwd[-1]:.6f}")
print(f"  W = {W_bwd}")
print(f"  b = {b_bwd}")
print()

losses_fwd = np.array(losses_fwd)
losses_bwd = np.array(losses_bwd)
steps = np.arange(1, N_STEPS + 1)

# --- Power-law fit: L(t) - L* ~ t^(-alpha) ---
# Fit in log-log space: log(L - L*) = -alpha * log(t) + const
# Use the tail region where transient is over

def fit_power_law(t, excess_loss, fit_start, fit_end):
    """Fit log(excess_loss) = -alpha * log(t) + c in [fit_start, fit_end]."""
    mask = (t >= fit_start) & (t <= fit_end) & (excess_loss > 0)
    log_t = np.log(t[mask])
    log_L = np.log(excess_loss[mask])
    # Linear regression
    A = np.stack([log_t, np.ones_like(log_t)], axis=1)
    coeffs = np.linalg.lstsq(A, log_L, rcond=None)[0]
    alpha = -coeffs[0]
    const = coeffs[1]
    return alpha, const

# Forward: L_fwd -> 0, so excess = L_fwd itself
fit_start, fit_end = 100, N_STEPS
alpha_fwd, c_fwd = fit_power_law(steps, losses_fwd, fit_start, fit_end)
print(f"Forward power-law exponent:  alpha = {alpha_fwd:.4f}")
print(f"  (fit range: steps {fit_start}–{fit_end})")

# Backward: L_bwd -> L*, so excess = L_bwd - L*
excess_bwd = losses_bwd - optimal_bwd
alpha_bwd, c_bwd = fit_power_law(steps, excess_bwd, fit_start, fit_end)
print(f"Backward power-law exponent: alpha = {alpha_bwd:.4f}")
print(f"  (fit range: steps {fit_start}–{fit_end})")
print()

# --- Plots ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: linear scale loss curves
ax = axes[0]
ax.plot(steps, losses_fwd, label="Forward: (a,b)→c", color="C0")
ax.plot(steps, losses_bwd, label="Backward: (c,b)→a", color="C1")
ax.axhline(optimal_fwd, color="C0", linestyle="--", alpha=0.5, label=f"Optimal fwd = {optimal_fwd:.3f}")
ax.axhline(optimal_bwd, color="C1", linestyle="--", alpha=0.5, label=f"Optimal bwd = {optimal_bwd:.3f}")
ax.set_xlabel("Training step")
ax.set_ylabel("Cross-entropy loss (nats)")
ax.set_title("Loss curves")
ax.legend()
ax.set_ylim(bottom=-0.02)

# Right: log-log of excess loss with power-law fits
ax = axes[1]
t_fit = np.linspace(fit_start, fit_end, 200)
ax.plot(steps, losses_fwd, label="Forward: L(t)", color="C0", alpha=0.6)
ax.plot(t_fit, np.exp(c_fwd) * t_fit**(-alpha_fwd), "--", color="C0",
        label=f"Fwd fit: $t^{{-{alpha_fwd:.2f}}}$")
ax.plot(steps, excess_bwd, label="Backward: L(t) − L*", color="C1", alpha=0.6)
# Only plot fit where excess > 0
ax.plot(t_fit, np.exp(c_bwd) * t_fit**(-alpha_bwd), "--", color="C1",
        label=f"Bwd fit: $t^{{-{alpha_bwd:.2f}}}$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Training step")
ax.set_ylabel("Excess loss (nats)")
ax.set_title("Power-law convergence")
ax.legend()

fig.suptitle("Linear model: AND forward vs backward", fontsize=14)
fig.tight_layout()
fig.savefig("toy/loss_curves.pdf")
print("Figure saved to toy/loss_curves.pdf")
