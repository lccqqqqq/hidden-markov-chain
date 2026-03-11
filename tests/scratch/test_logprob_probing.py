"""
Quick test: probe post-LN activations for log(O^T b) vs b.

Compares R² of linear probes for:
  1. Belief state b (the standard target)
  2. Bayes-optimal log-probs: log(O^T b)

If the residual stream stores b linearly, both should have similar R²
(since log(O^T b) ≈ linear in b for near-diagonal O).
If the residual stream pre-computes log-probs, the log-prob probe
might have *higher* R².

Usage:
    python tests/scratch/test_logprob_probing.py
"""
#%%
import sys
import os

os.chdir(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, "src")

import numpy as np
import torch as t
from sklearn.linear_model import Ridge
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
RIDGE_ALPHA = 1.0
TEST_FRACTION = 0.2

np.random.seed(SEED)
t.manual_seed(SEED)

# ── Load model and generate data ────────────────────────────────────────────
config_path = os.path.join(MODEL_DIR, "config.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

process = get_process_from_config(config)
sequences = process.generate_data(batch_size=BATCH_SIZE, length=SEQ_LENGTH)
belief_states = process.mixed_state_presentation(sequences.numpy())  # (N, T, n_states)
b_last = belief_states[:, -1, :]  # (N, n_states) — belief at last position

# Observation matrix O: (d_vocab, n_states)
O = compute_observation_matrix(process)
print(f"Observation matrix O:\n{O}\n")

# ── Compute targets ─────────────────────────────────────────────────────────
# Target 1: belief state b
target_b = b_last  # (N, 3)

# Target 2: Bayes-optimal log-probs log(O^T b)
probs = (O.T @ target_b.T).T  # (N, d_vocab) — this is O^T b for each sample
# Clip to avoid log(0)
probs = np.clip(probs, 1e-10, None)
target_logprob = np.log(probs)  # (N, d_vocab)

# Also compute: how linear is log(O^T b) in b?
# Fit b -> log(O^T b) with linear regression to see residual nonlinearity
from sklearn.linear_model import LinearRegression
reg_lin = LinearRegression()
reg_lin.fit(target_b, target_logprob)
logprob_pred_from_b = reg_lin.predict(target_b)
r2_logprob_from_b = r2_score(target_logprob, logprob_pred_from_b)
print(f"R² of log(O^T b) predicted linearly from b: {r2_logprob_from_b:.6f}")
print(f"  (If close to 1, the two targets are nearly linearly related)\n")

# ── Extract post-LN activations ─────────────────────────────────────────────
print("Loading model and extracting post-LN activations...")
model = load_model_from_dir(MODEL_DIR, device=DEVICE)
model.eval()

with t.no_grad():
    _, cache = model.run_with_cache(sequences.to(DEVICE))
    h_postln = cache["ln_final.hook_normalized"][:, -1, :].cpu().numpy()  # (N, 64)

del model, cache
print(f"Activation shape: {h_postln.shape}\n")

# ── Probe for both targets ──────────────────────────────────────────────────
def probe_r2(features, targets, label=""):
    X_tr, X_te, y_tr, y_te = train_test_split(
        features, targets, test_size=TEST_FRACTION, random_state=SEED
    )
    reg = Ridge(alpha=RIDGE_ALPHA)
    reg.fit(X_tr, y_tr)
    y_pred = reg.predict(X_te)
    r2 = r2_score(y_te, y_pred)
    return r2, reg

print("=" * 60)
print(f"{'Target':<30s}  {'dims':>4s}  {'R²':>10s}")
print("=" * 60)

r2_belief, reg_belief = probe_r2(h_postln, target_b, "belief")
print(f"  {'Belief state b':<28s}  {target_b.shape[1]:>4d}  {r2_belief:>10.6f}")

r2_logprob, reg_logprob = probe_r2(h_postln, target_logprob, "logprob")
print(f"  {'log(O^T b)':<28s}  {target_logprob.shape[1]:>4d}  {r2_logprob:>10.6f}")

# Also probe for the raw probabilities O^T b
r2_prob, reg_prob = probe_r2(h_postln, probs, "prob")
print(f"  {'O^T b (probabilities)':<28s}  {probs.shape[1]:>4d}  {r2_prob:>10.6f}")

print("=" * 60)

# ── Also do per-layer probing ────────────────────────────────────────────────
print("\n\nPer-layer comparison:")
print("=" * 60)
print(f"{'Layer':<8s}  {'R²(b)':>10s}  {'R²(log)':>10s}  {'R²(prob)':>10s}")
print("=" * 60)

# Re-load model to get per-layer activations
model = load_model_from_dir(MODEL_DIR, device=DEVICE)
model.eval()

with t.no_grad():
    _, cache = model.run_with_cache(sequences.to(DEVICE))

n_layers = config["model"]["n_layer"]
for layer in range(n_layers):
    h_layer = cache[f"blocks.{layer}.hook_resid_post"][:, -1, :].cpu().numpy()
    r2_b, _ = probe_r2(h_layer, target_b)
    r2_lp, _ = probe_r2(h_layer, target_logprob)
    r2_p, _ = probe_r2(h_layer, probs)
    print(f"  {layer:<6d}  {r2_b:>10.6f}  {r2_lp:>10.6f}  {r2_p:>10.6f}")

# Post-LN (already computed)
print(f"  {'postLN':<6s}  {r2_belief:>10.6f}  {r2_logprob:>10.6f}  {r2_prob:>10.6f}")

print("=" * 60)

del model, cache
print("\nDone.")
