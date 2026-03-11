"""
Sweep over many random "wrong" HMM models to show that belief-state
decodability from a random-init transformer is generic.

For each trial, we:
  1. Generate a random emission matrix E_wrong (d_vocab × n_states × n_states)
  2. Compute belief states under this wrong model on the SAME Mess3 sequences
  3. Probe a random-init transformer's activations for these wrong belief states

If the R² is consistently high across many random models, it confirms that
the decodability is a property of the random feature map, not of the
"correct" HMM structure.

Usage:
    python tests/scratch/test_wrong_model_sweep.py
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
N_RANDOM_MODELS = 20  # number of random wrong HMMs to test

# Parameters for random HMMs
N_STATES_OPTIONS = [2, 3, 4, 5]  # vary the number of hidden states too

np.random.seed(SEED)
t.manual_seed(SEED)


# ── Helpers ─────────────────────────────────────────────────────────────────
def probe_r2(features: np.ndarray, targets: np.ndarray) -> float:
    if targets.ndim == 1:
        targets = targets[:, None]
    X_tr, X_te, y_tr, y_te = train_test_split(
        features, targets, test_size=TEST_FRACTION, random_state=SEED
    )
    reg = Ridge(alpha=RIDGE_ALPHA)
    reg.fit(X_tr, y_tr)
    return r2_score(y_te, reg.predict(X_te))


def random_emission_matrix(d_vocab: int, n_states: int, rng: np.random.RandomState) -> np.ndarray:
    """Generate a random emission matrix E[j, i, k] = P(observe j, transition to k | state i).

    Uses Dirichlet distribution to ensure proper normalization:
    for each state i, the joint distribution over (j, k) sums to 1.
    """
    E = np.zeros((d_vocab, n_states, n_states))
    for i in range(n_states):
        # Draw from Dirichlet: joint distribution over (vocab × next_state)
        flat = rng.dirichlet(np.ones(d_vocab * n_states))
        E[:, i, :] = flat.reshape(d_vocab, n_states)
    return E


def compute_belief_states(sequences: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Compute belief states under a given emission matrix.

    Args:
        sequences: (batch, seq_length) integer token sequences
        E: (d_vocab, n_states, n_states) emission matrix

    Returns:
        beliefs: (batch, seq_length, n_states) belief state at each position
    """
    batch_size, seq_length = sequences.shape
    n_states = E.shape[1]
    T = E.sum(axis=0)  # (n_states, n_states) transition matrix
    O = E.sum(axis=2)  # (d_vocab, n_states) observation matrix

    beliefs = np.zeros((batch_size, seq_length, n_states))
    for i in range(batch_size):
        b = np.ones(n_states) / n_states  # uniform prior
        for pos in range(seq_length):
            tok = sequences[i, pos]
            # Bayesian update: b' ∝ O[tok, :] * (T @ b)
            b_pred = T @ b
            b = O[tok, :] * b_pred
            norm = b.sum()
            if norm > 1e-15:
                b = b / norm
            else:
                b = np.ones(n_states) / n_states  # reset if degenerate
            beliefs[i, pos, :] = b
    return beliefs


# ── Step 1: Generate Mess3 data ─────────────────────────────────────────────
config_path = os.path.join(MODEL_DIR, "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

process = get_process_from_config(config)
d_vocab = config["model"]["vocab_size"]  # 3

sequences = process.generate_data(batch_size=BATCH_SIZE, length=SEQ_LENGTH)
seq_np = sequences.numpy()
print(f"Mess3 data: {BATCH_SIZE} sequences, length {SEQ_LENGTH}, vocab {d_vocab}")


# ── Step 2: Extract random-init transformer activations ─────────────────────
print("Creating random-init transformer...")
model = initialize_transformer_from_yaml(config_path)
model.eval()
model.to(DEVICE)

belief_states_true = process.mixed_state_presentation(seq_np)
with t.no_grad():
    raw = extract_residual_stream(model, sequences, device=DEVICE)
    layer_actvs, _ = prepare_probe_data(raw, belief_states_true, use_last_pos=True)
    features = layer_actvs[PROBE_LAYER]
del model, raw
print(f"  Activation shape: {features.shape}")
print()


# ── Step 3: Probe true belief state ────────────────────────────────────────
true_targets = belief_states_true[:, -1, :]
r2_true = probe_r2(features, true_targets)
print(f"True Mess3 belief state (n_states=3):  R² = {r2_true:.4f}")
print()


# ── Step 4: Sweep over random wrong models ──────────────────────────────────
print("=" * 70)
print(f"{'Model':<35s}  {'n_states':>8s}  {'R²':>8s}")
print("=" * 70)
print(f"  {'True Mess3':<33s}  {3:>8d}  {r2_true:>8.4f}")
print("-" * 70)

rng = np.random.RandomState(777)
all_r2 = []

for trial in range(N_RANDOM_MODELS):
    # Pick a random number of hidden states
    n_states = rng.choice(N_STATES_OPTIONS)

    # Generate random emission matrix
    E_wrong = random_emission_matrix(d_vocab, n_states, rng)

    # Compute belief states under this wrong model
    wrong_beliefs = compute_belief_states(seq_np, E_wrong)
    wrong_targets = wrong_beliefs[:, -1, :]  # (N, n_states)

    # Probe
    r2 = probe_r2(features, wrong_targets)
    all_r2.append(r2)

    print(f"  Random model {trial+1:2d}                  {n_states:>8d}  {r2:>8.4f}")

print("=" * 70)
print()

all_r2 = np.array(all_r2)
print(f"Summary over {N_RANDOM_MODELS} random wrong models:")
print(f"  R² mean  = {all_r2.mean():.4f}")
print(f"  R² std   = {all_r2.std():.4f}")
print(f"  R² min   = {all_r2.min():.4f}")
print(f"  R² max   = {all_r2.max():.4f}")
print()
print("Interpretation:")
print("  If R² is consistently high across all random models, then")
print("  belief-state decodability at random init is a GENERIC property")
print("  of any smooth, low-dimensional function of the input — not")
print("  evidence that the network has learned HMM structure.")
