"""
Compare decodability of true vs wrong-model belief states
in random-init vs trained transformer.

At random init, both should be equally decodable (reservoir effect).
After training, true belief states should be much more decodable
than wrong-model ones.

Usage:
    python tests/scratch/test_wrong_model_trained_vs_random.py
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

from activation_extraction import (
    load_model_from_dir,
    get_process_from_config,
    extract_residual_stream,
    prepare_probe_data,
)
from utils import initialize_transformer_from_yaml
import yaml

# ── Settings ────────────────────────────────────────────────────────────────
MODEL_DIR = "models/mess3/20260306_150920_L4_d64_H1_full_LN"
BATCH_SIZE = 10_000
SEQ_LENGTH = 10
DEVICE = "cpu"
SEED = 42
RIDGE_ALPHA = 1.0
TEST_FRACTION = 0.2
N_WRONG_MODELS = 10

np.random.seed(SEED)
t.manual_seed(SEED)

# ── Load config and generate data ───────────────────────────────────────────
config_path = os.path.join(MODEL_DIR, "config.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

process = get_process_from_config(config)
sequences = process.generate_data(batch_size=BATCH_SIZE, length=SEQ_LENGTH)
seq_np = sequences.numpy()
belief_states = process.mixed_state_presentation(seq_np)
true_beliefs_last = belief_states[:, -1, :]  # (N, 3)

n_states = process.num_hidden_states
vocab_size = config["model"]["vocab_size"]


# ── Generate wrong-model belief states ──────────────────────────────────────
def compute_wrong_beliefs(seq_np, n_hidden, d_vocab, rng_seed):
    """Compute belief states under a random HMM with given n_hidden states."""
    rng = np.random.RandomState(rng_seed)
    # Random transition matrix (rows sum to 1)
    T = rng.dirichlet(np.ones(n_hidden), size=n_hidden)  # (n_hidden, n_hidden)
    # Random observation matrix (columns sum to 1 over vocab)
    O = rng.dirichlet(np.ones(d_vocab), size=n_hidden).T  # (d_vocab, n_hidden)

    N, L = seq_np.shape
    beliefs = np.zeros((N, L, n_hidden))
    for i in range(N):
        b = np.ones(n_hidden) / n_hidden
        for pos in range(L):
            tok = seq_np[i, pos]
            b_pred = T.T @ b  # predict
            b = O[tok, :] * b_pred  # update
            b = b / (b.sum() + 1e-15)
            beliefs[i, pos, :] = b
    return beliefs[:, -1, :]  # (N, n_hidden)


print(f"Computing {N_WRONG_MODELS} wrong-model belief states...")
wrong_beliefs = []
for i in range(N_WRONG_MODELS):
    # Use 3 hidden states (same as true model) for fair comparison
    wb = compute_wrong_beliefs(seq_np, n_states, vocab_size, rng_seed=1000 + i)
    wrong_beliefs.append(wb)
    print(f"  Wrong model {i}: belief range [{wb.min():.3f}, {wb.max():.3f}]")


# ── Probe helper ────────────────────────────────────────────────────────────
def probe_r2(features, targets):
    if targets.ndim == 1:
        targets = targets[:, None]
    X_tr, X_te, y_tr, y_te = train_test_split(
        features, targets, test_size=TEST_FRACTION, random_state=SEED)
    reg = Ridge(alpha=RIDGE_ALPHA)
    reg.fit(X_tr, y_tr)
    return r2_score(y_te, reg.predict(X_te))


# ── Extract activations for both models ─────────────────────────────────────
print("\nExtracting activations...")

# Random init
model_rand = initialize_transformer_from_yaml(config_path)
model_rand.eval().to(DEVICE)
raw_rand = extract_residual_stream(model_rand, sequences, device=DEVICE)
layer_actvs_rand, _ = prepare_probe_data(raw_rand, belief_states, use_last_pos=True)
del model_rand, raw_rand

# Trained model
model_trained = load_model_from_dir(MODEL_DIR, device=DEVICE)
raw_trained = extract_residual_stream(model_trained, sequences, device=DEVICE)
layer_actvs_trained, _ = prepare_probe_data(raw_trained, belief_states, use_last_pos=True)
del model_trained, raw_trained

n_layers = layer_actvs_rand.shape[0]

# ── Probe and compare ───────────────────────────────────────────────────────
print("\n" + "=" * 75)
print(f"{'Target':<25s}  {'Model':>10s}", end="")
for l in range(n_layers):
    print(f"  {'L'+str(l)+' R²':>8s}", end="")
print()
print("=" * 75)

for model_label, actvs in [("Random", layer_actvs_rand), ("Trained", layer_actvs_trained)]:
    # True belief state
    r2s = [probe_r2(actvs[l], true_beliefs_last) for l in range(n_layers)]
    print(f"  {'True belief':<23s}  {model_label:>10s}", end="")
    for r2 in r2s:
        print(f"  {r2:>8.4f}", end="")
    print()

    # Wrong-model belief states (average over N_WRONG_MODELS)
    wrong_r2s_per_layer = [[] for _ in range(n_layers)]
    for wb in wrong_beliefs:
        for l in range(n_layers):
            wrong_r2s_per_layer[l].append(probe_r2(actvs[l], wb))

    means = [np.mean(wrong_r2s_per_layer[l]) for l in range(n_layers)]
    stds = [np.std(wrong_r2s_per_layer[l]) for l in range(n_layers)]
    print(f"  {'Wrong belief (mean)':<23s}  {model_label:>10s}", end="")
    for m in means:
        print(f"  {m:>8.4f}", end="")
    print()
    print(f"  {'Wrong belief (std)':<23s}  {model_label:>10s}", end="")
    for s in stds:
        print(f"  {s:>8.4f}", end="")
    print()

print("=" * 75)
print()
print("Interpretation:")
print("  At random init, true and wrong beliefs should have similar R².")
print("  After training, true beliefs should be much more decodable than wrong ones.")
print("  The gap (true - wrong) measures how much training specializes the representation.")
