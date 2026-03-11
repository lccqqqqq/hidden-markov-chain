"""
Is the belief state a "special" function for linear probing at random init?

We take the SAME Mess3 sequences and random-init transformer, and probe for
many different deterministic functions of the input — not just the belief state.
If they ALL give high R², it shows that belief-state decodability at random
init is a generic property of random feature maps, not evidence that the
network has "learned" anything about HMM structure.

Target functions (all map input sequence → low-dimensional output):

  Belief-state related:
    1. True belief state          — the real Bayesian posterior (2D on simplex)
    2. Wrong-model belief state   — belief update with a WRONG transition matrix

  Simple statistics:
    3. Token counts               — (count of each token) / seq_length  (2D)
    4. Last-token one-hot         — identity of the last token (2D on simplex)
    5. Positional token average   — position-weighted average of token values (1D)

  Arbitrary nonlinear:
    6. Bigram counts              — counts of all 9 possible bigrams (8D)
    7. Trigram hash               — hash of last 3 tokens mapped to 2D
    8. Modular arithmetic         — (cumulative sum mod 3) at last position (2D)

All targets are deterministic functions of the same input and map to
low-dimensional spaces (1D–8D).  The reservoir computing prediction is
that ALL of them will have high R² from a random-init transformer.

Usage:
    python tests/scratch/test_belief_state_not_special.py
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
RIDGE_ALPHA = 1.0
PROBE_LAYER = 3
DEVICE = "cpu"
SEED = 42

np.random.seed(SEED)
t.manual_seed(SEED)


# ── Helper ──────────────────────────────────────────────────────────────────
def probe_r2(features: np.ndarray, targets: np.ndarray) -> float:
    if targets.ndim == 1:
        targets = targets[:, None]
    X_tr, X_te, y_tr, y_te = train_test_split(
        features, targets, test_size=TEST_FRACTION, random_state=SEED
    )
    reg = Ridge(alpha=RIDGE_ALPHA)
    reg.fit(X_tr, y_tr)
    y_pred = reg.predict(X_te)
    return r2_score(y_te, y_pred)


# ── Step 1: Generate Mess3 data ─────────────────────────────────────────────
config_path = os.path.join(MODEL_DIR, "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

process = get_process_from_config(config)
sequences = process.generate_data(batch_size=BATCH_SIZE, length=SEQ_LENGTH)
seq_np = sequences.numpy()
vocab_size = config["model"]["vocab_size"]  # 3 for Mess3

print(f"Mess3 data: {BATCH_SIZE} sequences, length {SEQ_LENGTH}, vocab {vocab_size}")
print()


# ── Step 2: Compute all target functions ────────────────────────────────────
targets = {}

# --- 1. True belief state (the "special" function from the paper) ---
# This is the Bayesian posterior P(hidden state | observations so far).
# For Mess3 with 3 hidden states, this is 2D (lives on 2-simplex).
belief_states = process.mixed_state_presentation(seq_np)
targets["1. True belief state"] = belief_states[:, -1, :]  # (N, 3)

# --- 2. Wrong-model belief state ---
# Run the same Bayesian update but with a WRONG transition matrix.
# If belief-state R² is "special", this should give lower R²; if it's
# just "any smooth function of input", it should be similarly high.
E_true = process.emission_matrices  # (d_vocab, n_states, n_states)
# Create a wrong model: randomly permute the transition structure
rng = np.random.RandomState(123)
perm = rng.permutation(process.num_hidden_states)
E_wrong = E_true[:, perm, :][:, :, perm]  # permute states
# Normalize to valid emission matrices (each column sums to 1 over vocab×next_state)
E_wrong = E_wrong / E_wrong.sum(axis=(0, 2), keepdims=True) * E_wrong.sum(axis=(0, 2), keepdims=True).mean()

# Manual belief update with wrong model
n_states = process.num_hidden_states
wrong_beliefs = np.zeros((BATCH_SIZE, SEQ_LENGTH, n_states))
T_wrong = E_wrong.sum(axis=0)  # (n_states, n_states): transition matrix
O_wrong = E_wrong.sum(axis=2)  # (d_vocab, n_states): observation matrix
for i in range(BATCH_SIZE):
    b = np.ones(n_states) / n_states  # uniform prior
    for pos in range(SEQ_LENGTH):
        tok = seq_np[i, pos]
        # Update: b' ∝ O[tok, :] * (T @ b)  — wrong model's update
        b_pred = T_wrong @ b
        b = O_wrong[tok, :] * b_pred
        b = b / (b.sum() + 1e-15)
        wrong_beliefs[i, pos, :] = b
targets["2. Wrong-model belief"] = wrong_beliefs[:, -1, :]  # (N, 3)

# --- 3. Token counts ---
# Fraction of each token in the sequence.  This is a LINEAR function of
# the one-hot input.  For vocab=3, this is 2D (sums to 1).
token_counts = np.zeros((BATCH_SIZE, vocab_size))
for tok in range(vocab_size):
    token_counts[:, tok] = (seq_np == tok).sum(axis=1) / SEQ_LENGTH
targets["3. Token counts"] = token_counts  # (N, 3)

# --- 4. Last-token one-hot ---
# Just the identity of the last token.  Extremely simple function.
last_onehot = np.zeros((BATCH_SIZE, vocab_size))
for tok in range(vocab_size):
    last_onehot[:, tok] = (seq_np[:, -1] == tok).astype(float)
targets["4. Last token (one-hot)"] = last_onehot  # (N, 3)

# --- 5. Position-weighted average ---
# A simple 1D function: sum_t (t/T) * x_t, normalized.
positions = np.arange(SEQ_LENGTH) / SEQ_LENGTH
pos_avg = (seq_np * positions[None, :]).sum(axis=1) / (vocab_size - 1)
targets["5. Position-weighted avg"] = pos_avg  # (N,)

# --- 6. Bigram counts ---
# Count of each of the vocab² possible bigrams.  This is a nonlinear
# function (depends on pairs of adjacent tokens).  8D for vocab=3.
n_bigrams = vocab_size * vocab_size
bigram_counts = np.zeros((BATCH_SIZE, n_bigrams))
for pos in range(SEQ_LENGTH - 1):
    bigram_idx = seq_np[:, pos] * vocab_size + seq_np[:, pos + 1]
    for b in range(n_bigrams):
        bigram_counts[:, b] += (bigram_idx == b).astype(float)
bigram_counts /= (SEQ_LENGTH - 1)  # normalize
targets["6. Bigram counts"] = bigram_counts  # (N, 9)

# --- 7. Trigram hash → 2D ---
# Map the last 3 tokens to a 2D point via a fixed random projection.
# This is an arbitrary nonlinear function with no meaningful structure.
hash_rng = np.random.RandomState(999)
# Create a lookup table: for each (t1, t2, t3) → random 2D point
trigram_table = hash_rng.randn(vocab_size, vocab_size, vocab_size, 2)
trigram_hash = np.zeros((BATCH_SIZE, 2))
for i in range(BATCH_SIZE):
    t1, t2, t3 = seq_np[i, -3], seq_np[i, -2], seq_np[i, -1]
    trigram_hash[i] = trigram_table[t1, t2, t3]
targets["7. Trigram hash (2D)"] = trigram_hash  # (N, 2)

# --- 8. Modular arithmetic ---
# (cumulative sum of tokens mod 3) at last position, as one-hot.
# This is a highly nonlinear function of all tokens.
cum_mod = seq_np.sum(axis=1) % vocab_size
mod_onehot = np.zeros((BATCH_SIZE, vocab_size))
for v in range(vocab_size):
    mod_onehot[:, v] = (cum_mod == v).astype(float)
targets["8. Modular sum mod 3"] = mod_onehot  # (N, 3)


# ── Step 3: Extract random-init transformer activations ─────────────────────
print("Creating random-init transformer...")
model = initialize_transformer_from_yaml(config_path)
model.eval()
model.to(DEVICE)

with t.no_grad():
    raw = extract_residual_stream(model, sequences, device=DEVICE)
    layer_actvs, _ = prepare_probe_data(raw, belief_states, use_last_pos=True)
    features = layer_actvs[PROBE_LAYER]  # (N, 64)
del model, raw

print(f"  Activation shape: {features.shape}")
print()


# ── Step 4: Probe each target ───────────────────────────────────────────────
print("=" * 65)
print(f"{'Target function':<30s}  {'dim':>4s}  {'R² (random)':>11s}  {'R² (shuffled)':>13s}")
print("=" * 65)

for name, target_vals in targets.items():
    r2_real = probe_r2(features, target_vals)

    # Shuffled control: break input-activation correspondence
    shuffle_idx = np.random.permutation(BATCH_SIZE)
    r2_shuf = probe_r2(features[shuffle_idx], target_vals)

    dim = target_vals.shape[1] if target_vals.ndim > 1 else 1
    print(f"  {name:<28s}  {dim:>4d}  {r2_real:>11.4f}  {r2_shuf:>13.4f}")

print("=" * 65)
print()
print("Interpretation:")
print("  ALL target functions should have high R² from the random-init")
print("  transformer (column 3), while shuffled controls are ~0 (column 4).")
print()
print("  This shows that belief-state decodability at random init is NOT")
print("  special — ANY low-dimensional deterministic function of the input")
print("  is linearly decodable from a random transformer's activations.")
print("  The random network acts as a generic reservoir / random feature map.")
