"""
Control experiment: Why does a randomly-initialized network give R² ~ 0.9
for belief-state probing?

Hypothesis: The high R² comes from the fact that a random network is a
*deterministic* function of the input.  Even though the function is random,
it maps each unique input sequence to a fixed 64-dim vector.  A linear probe
can exploit these consistent (albeit arbitrary) embeddings to reconstruct
the belief state, which is also a deterministic function of the same input.

This script tests 4 conditions on the SAME set of input sequences:
  1. Random-init network activations  → expect R² ~ 0.9  (input-dependent)
  2. One-hot encoding of input tokens → expect R² ~ 0.97 (perfect input representation)
  3. Purely random features (IID Gaussian, NO input dependence) → expect R² ~ 0
  4. Shuffled activations (break input-activation correspondence) → expect R² ~ 0

Conditions 3 & 4 are the key controls.  If R² drops to ~0, it confirms the
high R² in condition 1 is due to the input-dependence of the activations,
NOT overfitting or some artifact.

Usage:
    python tests/scratch/test_random_baseline_r2.py
"""
#%%
import sys
import os

os.chdir(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, "src")

import numpy as np
import torch as t
import yaml
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from activation_extraction import (
    extract_residual_stream,
    prepare_probe_data,
    get_process_from_config,
)
from utils import initialize_transformer_from_yaml


# ── Settings ────────────────────────────────────────────────────────────────
MODEL_DIR = "models/mess3/20260308_144221_L4_d64_H1_full_LN"
BATCH_SIZE = 10_000
SEQ_LENGTH = 10
TEST_FRACTION = 0.2
PROBE_LAYER = 3          # last layer, where we typically probe
RIDGE_ALPHA = 1.0
DEVICE = "cpu"
SEED = 42

np.random.seed(SEED)
t.manual_seed(SEED)


# ── Helper: fit a Ridge probe and return test R² ────────────────────────────
def probe_r2(features: np.ndarray, targets: np.ndarray) -> float:
    """Fit Ridge regression on features → targets, return test R²."""
    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=TEST_FRACTION, random_state=SEED
    )
    reg = Ridge(alpha=RIDGE_ALPHA)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    return r2_score(y_test, y_pred)


# ── Step 1: Generate data and belief states ─────────────────────────────────
config_path = os.path.join(MODEL_DIR, "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

process = get_process_from_config(config)
sequences = process.generate_data(batch_size=BATCH_SIZE, length=SEQ_LENGTH)
belief_states = process.mixed_state_presentation(sequences.numpy())

# Targets: belief state at the last position, shape (n_samples, n_states)
targets = belief_states[:, -1, :]
print(f"Data: {BATCH_SIZE} sequences, length {SEQ_LENGTH}")
print(f"Target shape: {targets.shape}  (belief state dimension = {targets.shape[1]})")
print()


# ── Condition 1: Random-init network activations ───────────────────────────
# A fresh transformer with random weights.  Activations are a deterministic
# (but arbitrary) nonlinear function of the input sequence.
print("=" * 60)
print("Condition 1: Random-init network activations")
print("=" * 60)

model = initialize_transformer_from_yaml(config_path)
model.eval()
model.to(DEVICE)

raw = extract_residual_stream(model, sequences, device=DEVICE)
layer_actvs, _ = prepare_probe_data(raw, belief_states, use_last_pos=True)
random_net_features = layer_actvs[PROBE_LAYER]  # (n_samples, d_model=64)
del model, raw

r2_random_net = probe_r2(random_net_features, targets)
print(f"  R² = {r2_random_net:.4f}")
print(f"  → These features ARE input-dependent (deterministic function of tokens)")
print()


# ── Condition 2: One-hot encoding of input ──────────────────────────────────
# The "gold standard" input representation: one-hot over (position × token).
# This preserves ALL information about the input sequence.
print("=" * 60)
print("Condition 2: One-hot encoding of input tokens")
print("=" * 60)

vocab_size = config["model"]["vocab_size"]
seq_np = sequences.numpy()  # (n_samples, seq_length)
# Flatten one-hot: each sample → vector of length seq_length * vocab_size
onehot_features = np.zeros((BATCH_SIZE, SEQ_LENGTH * vocab_size))
for pos in range(SEQ_LENGTH):
    for tok in range(vocab_size):
        onehot_features[:, pos * vocab_size + tok] = (seq_np[:, pos] == tok).astype(float)

r2_onehot = probe_r2(onehot_features, targets)
print(f"  R² = {r2_onehot:.4f}")
print(f"  → Upper bound: perfect input representation, linear probe")
print()


# ── Condition 3: Purely random features (no input dependence) ───────────────
# IID Gaussian vectors with the SAME shape as condition 1, but NO relation
# to the input.  If R² ~ 0 here, it proves the high R² in condition 1
# is due to input-dependence, not overfitting on 64 features.
print("=" * 60)
print("Condition 3: Purely random features (IID Gaussian)")
print("=" * 60)

random_features = np.random.randn(BATCH_SIZE, random_net_features.shape[1])

r2_random = probe_r2(random_features, targets)
print(f"  R² = {r2_random:.4f}")
print(f"  → NO input dependence — probe has nothing to learn from")
print()


# ── Condition 4: Shuffled activations ───────────────────────────────────────
# Take the SAME activations from condition 1, but randomly permute across
# samples.  This preserves the marginal distribution of activations but
# breaks the correspondence between input and activation.
print("=" * 60)
print("Condition 4: Shuffled activations (break input↔activation mapping)")
print("=" * 60)

shuffle_idx = np.random.permutation(BATCH_SIZE)
shuffled_features = random_net_features[shuffle_idx]

r2_shuffled = probe_r2(shuffled_features, targets)
print(f"  R² = {r2_shuffled:.4f}")
print(f"  → Same activation vectors, but wrong samples — no input correspondence")
print()


# ── Summary ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  {'Condition':<45} {'R²':>8}")
print(f"  {'-'*45} {'-'*8}")
print(f"  {'1. Random-init network (input-dependent)':<45} {r2_random_net:>8.4f}")
print(f"  {'2. One-hot input (upper bound)':<45} {r2_onehot:>8.4f}")
print(f"  {'3. IID Gaussian (no input dependence)':<45} {r2_random:>8.4f}")
print(f"  {'4. Shuffled activations (broken mapping)':<45} {r2_shuffled:>8.4f}")
print()
print("Interpretation:")
print("  If conditions 3 & 4 give R² ≈ 0 while condition 1 gives R² ~ 0.9,")
print("  it confirms: the random network's R² comes from being a deterministic")
print("  function of the input, NOT from overfitting or probe expressiveness.")
print("  The network acts as a 'random feature map' (cf. reservoir computing).")
