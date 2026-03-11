"""
Where does the belief-to-logit conversion happen?

For each layer, we measure:
1. R²(h → b)           — can we decode the belief state?
2. R²(h → log(O^T b))  — can we decode the Bayes-optimal logits?
3. R²(h → δ)           — can we decode the NONLINEAR RESIDUAL
                          δ = log(O^T b) - M_best · b?
                          (This isolates the part that ISN'T linear in b)

If layers pre-compute log-probs, δ should become more decodable at later layers.
If layers only store b, δ should be near-zero everywhere.

Also: for each layer, compute W_U^T · h and compare to both O^T b and log(O^T b).

Usage:
    python tests/scratch/test_where_logprob_computed.py
"""
#%%
import sys
import os

os.chdir(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, "src")

import numpy as np
import torch as t
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from activation_extraction import load_model_from_dir, get_process_from_config
from unembed_analysis import compute_observation_matrix
import yaml

# ── Settings ────────────────────────────────────────────────────────────────
MODEL_DIR = "models/mess3/20260306_150920_L4_d64_H1_full_LN"
BATCH_SIZE = 10_000
SEQ_LENGTH = 10
DEVICE = "cpu"
SEED = 42

np.random.seed(SEED)
t.manual_seed(SEED)

# ── Load and generate ───────────────────────────────────────────────────────
config_path = os.path.join(MODEL_DIR, "config.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

process = get_process_from_config(config)
sequences = process.generate_data(batch_size=BATCH_SIZE, length=SEQ_LENGTH)
belief_states = process.mixed_state_presentation(sequences.numpy())
b_last = belief_states[:, -1, :]  # (N, 3)

O = compute_observation_matrix(process)  # (d_vocab, n_states)
true_probs = (O.T @ b_last.T).T          # (N, 3) = O^T b
log_probs = np.log(np.clip(true_probs, 1e-10, None))  # (N, 3) = log(O^T b)

# Compute the nonlinear residual: δ = log(O^T b) - best_linear_approx(b)
reg_best = LinearRegression().fit(b_last, log_probs)
log_probs_linear = reg_best.predict(b_last)
delta = log_probs - log_probs_linear  # (N, 3) — the nonlinear part

print(f"Observation matrix O:\n{O}\n")
print(f"δ (nonlinear residual) stats:")
print(f"  mean: {delta.mean():.6f}, std: {delta.std():.6f}")
print(f"  max |δ|: {np.abs(delta).max():.6f}")
print(f"  R² of linear approx log(O^T b) ≈ M·b: {r2_score(log_probs, log_probs_linear):.6f}")
print()

# ── Extract activations ─────────────────────────────────────────────────────
model = load_model_from_dir(MODEL_DIR, device=DEVICE)
model.eval()

# Get W_U for direct logit comparison
W_U = model.W_U.detach().cpu().numpy()    # (d_model, d_vocab)
b_U = model.b_U.detach().cpu().numpy()    # (d_vocab,)

with t.no_grad():
    _, cache = model.run_with_cache(sequences.to(DEVICE))

n_layers = config["model"]["n_layer"]

# ── Probe helper ────────────────────────────────────────────────────────────
def probe_r2(features, targets):
    if targets.ndim == 1:
        targets = targets[:, None]
    X_tr, X_te, y_tr, y_te = train_test_split(
        features, targets, test_size=0.2, random_state=SEED)
    reg = Ridge(alpha=1.0)
    reg.fit(X_tr, y_tr)
    return r2_score(y_te, reg.predict(X_te))


# ── Per-layer analysis ──────────────────────────────────────────────────────
print("=" * 85)
print(f"{'Layer':<8s}  {'R²(b)':>8s}  {'R²(logp)':>9s}  {'R²(δ)':>8s}  "
      f"{'corr(Wh,Ob)':>12s}  {'corr(Wh,logOb)':>15s}")
print("=" * 85)

for layer in range(n_layers):
    h = cache[f"blocks.{layer}.hook_resid_post"][:, -1, :].cpu().numpy()

    r2_b = probe_r2(h, b_last)
    r2_logp = probe_r2(h, log_probs)
    r2_delta = probe_r2(h, delta)

    # What would W_U read from this layer's activations?
    # (This isn't quite right since LN + later layers intervene,
    #  but it shows the "potential" logit content)
    logits_from_h = h @ W_U + b_U  # (N, 3)

    # Correlation with O^T b (probability space)
    corr_prob = np.mean([
        np.corrcoef(logits_from_h[:, j], true_probs[:, j])[0, 1]
        for j in range(3)
    ])
    # Correlation with log(O^T b) (logit space)
    corr_logp = np.mean([
        np.corrcoef(logits_from_h[:, j], log_probs[:, j])[0, 1]
        for j in range(3)
    ])

    print(f"  {layer:<6d}  {r2_b:>8.4f}  {r2_logp:>9.4f}  {r2_delta:>8.4f}  "
          f"{corr_prob:>12.4f}  {corr_logp:>15.4f}")

# Post-LN
h_postln = cache["ln_final.hook_normalized"][:, -1, :].cpu().numpy()
r2_b = probe_r2(h_postln, b_last)
r2_logp = probe_r2(h_postln, log_probs)
r2_delta = probe_r2(h_postln, delta)

logits_postln = h_postln @ W_U + b_U
corr_prob = np.mean([
    np.corrcoef(logits_postln[:, j], true_probs[:, j])[0, 1]
    for j in range(3)
])
corr_logp = np.mean([
    np.corrcoef(logits_postln[:, j], log_probs[:, j])[0, 1]
    for j in range(3)
])
print(f"  {'postLN':<6s}  {r2_b:>8.4f}  {r2_logp:>9.4f}  {r2_delta:>8.4f}  "
      f"{corr_prob:>12.4f}  {corr_logp:>15.4f}")

print("=" * 85)

# ── Model's actual output ───────────────────────────────────────────────────
print("\nModel's actual output logits (from postLN @ W_U):")
actual_probs = np.exp(logits_postln) / np.exp(logits_postln).sum(axis=1, keepdims=True)
kl_to_true = np.mean(np.sum(true_probs * np.log(
    np.clip(true_probs, 1e-10, None) / np.clip(actual_probs, 1e-10, None)
), axis=1))
print(f"  Mean KL(O^T b || softmax(model logits)): {kl_to_true:.6f} nats")

corr_final = [
    np.corrcoef(actual_probs[:, j], true_probs[:, j])[0, 1]
    for j in range(3)
]
print(f"  Per-token corr(model probs, O^T b): {corr_final}")

del model, cache
print("\nDone.")
