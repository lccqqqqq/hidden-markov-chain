"""
Demonstration: Random transformers act as reservoir-like feature maps.

NO HMM or belief states involved.  We use plain IID random token sequences
and probe whether various deterministic target functions can be linearly
decoded from the activations of a randomly-initialized transformer.

Target functions (all deterministic functions of the input sequence):
  1. Token frequency  — frac of token 0 in sequence       (linear in input)
  2. Bigram indicator — whether bigram (0,1) appears       (nonlinear, positional)
  3. Parity           — (sum of all tokens) mod 2          (highly nonlinear)
  4. Last-2 product   — x[-1] * x[-2] (scaled to [0,1])   (local nonlinear)

For each target, we compare 4 feature conditions:
  A. Random-init transformer activations  (input-dependent random features)
  B. One-hot encoding of input tokens     (perfect input representation)
  C. IID Gaussian features                (no input dependence — control)
  D. Shuffled activations                 (broken input correspondence — control)

Prediction: A and B give high R², C and D give R² ≈ 0.
This demonstrates the reservoir computing property of random transformers
without any domain-specific structure.

Usage:
    python tests/scratch/test_random_transformer_reservoir.py
"""
#%%
import sys
import os

os.chdir(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, "src")

import numpy as np
import torch as t
from transformer_lens import HookedTransformer, HookedTransformerConfig
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ── Settings ────────────────────────────────────────────────────────────────
VOCAB_SIZE = 5
SEQ_LENGTH = 8
BATCH_SIZE = 10_000
TEST_FRACTION = 0.2
RIDGE_ALPHA = 1.0
SEED = 42
DEVICE = "cpu"

# Transformer architecture (small, similar to our Mess3 model)
MODEL_CONFIG = dict(
    n_layers=4,
    d_model=64,
    d_head=64,
    n_heads=1,
    d_vocab=VOCAB_SIZE,
    n_ctx=SEQ_LENGTH,
    d_mlp=256,
    act_fn="relu",
    normalization_type="LN",
    attn_only=False,
)

np.random.seed(SEED)
t.manual_seed(SEED)


# ── Helper: fit Ridge probe and return test R² ─────────────────────────────
def probe_r2(features: np.ndarray, targets: np.ndarray) -> float:
    """Fit Ridge regression, return test R².  Handles both 1D and 2D targets."""
    if targets.ndim == 1:
        targets = targets[:, None]
    X_tr, X_te, y_tr, y_te = train_test_split(
        features, targets, test_size=TEST_FRACTION, random_state=SEED
    )
    reg = Ridge(alpha=RIDGE_ALPHA)
    reg.fit(X_tr, y_tr)
    y_pred = reg.predict(X_te)
    return r2_score(y_te, y_pred)


# ── Step 1: Generate random IID token sequences ────────────────────────────
# No HMM — just uniform random tokens.  The sequences have no temporal
# structure; any decodability comes purely from the feature map.
print("Generating random IID token sequences...")
sequences = t.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LENGTH))
seq_np = sequences.numpy()
print(f"  Shape: {sequences.shape}, vocab: {VOCAB_SIZE}, length: {SEQ_LENGTH}")
print()


# ── Step 2: Define target functions ─────────────────────────────────────────
# Each target is a deterministic function of the input sequence.
# They range from linear to highly nonlinear.

targets = {}

# 1. Token frequency: fraction of token 0 in the sequence
#    This is a LINEAR function of the one-hot input representation.
targets["token_freq"] = (seq_np == 0).mean(axis=1).astype(np.float64)

# 2. Bigram indicator: does the bigram (0, 1) appear anywhere?
#    This is NONLINEAR — it depends on pairs of adjacent positions.
has_bigram = np.zeros(BATCH_SIZE, dtype=bool)
for pos in range(SEQ_LENGTH - 1):
    has_bigram |= (seq_np[:, pos] == 0) & (seq_np[:, pos + 1] == 1)
targets["bigram_(0,1)"] = has_bigram.astype(np.float64)

# 3. Parity: (sum of all tokens) mod 2
#    Highly nonlinear — a classic hard function for linear methods on raw input.
targets["parity"] = (seq_np.sum(axis=1) % 2).astype(np.float64)

# 4. Last-2 product: x[-1] * x[-2], normalized to [0,1]
#    A local nonlinear interaction between the last two positions.
max_product = (VOCAB_SIZE - 1) ** 2
targets["last2_product"] = (seq_np[:, -1] * seq_np[:, -2]).astype(np.float64) / max_product

print("Target functions:")
for name, vals in targets.items():
    print(f"  {name:20s}  mean={vals.mean():.3f}  std={vals.std():.3f}")
print()


# ── Step 3: Create random-init transformer and extract activations ──────────
print("Creating random-init transformer...")
config = HookedTransformerConfig(**MODEL_CONFIG)
model = HookedTransformer(config)
model.eval()
model.to(DEVICE)

# Extract last-position activations from the final residual stream layer
with t.no_grad():
    _, cache = model.run_with_cache(sequences.to(DEVICE))
    # resid_post at the last layer, last sequence position
    random_net_activations = cache[f"blocks.{MODEL_CONFIG['n_layers']-1}.hook_resid_post"][:, -1, :]
    random_net_features = random_net_activations.cpu().numpy()

del model, cache
print(f"  Activation shape: {random_net_features.shape}")
print()


# ── Step 4: Prepare alternative feature sets ────────────────────────────────

# One-hot encoding: each sample → vector of length seq_length * vocab_size
onehot_features = np.zeros((BATCH_SIZE, SEQ_LENGTH * VOCAB_SIZE), dtype=np.float64)
for pos in range(SEQ_LENGTH):
    for tok in range(VOCAB_SIZE):
        onehot_features[:, pos * VOCAB_SIZE + tok] = (seq_np[:, pos] == tok).astype(float)

# IID Gaussian: same shape as transformer activations, NO input dependence
iid_features = np.random.randn(BATCH_SIZE, random_net_features.shape[1])

# Shuffled: same activations, randomly permuted across samples
shuffle_idx = np.random.permutation(BATCH_SIZE)
shuffled_features = random_net_features[shuffle_idx]


# ── Step 5: Probe all combinations ─────────────────────────────────────────
feature_sets = {
    "A. Random transformer": random_net_features,
    "B. One-hot input":      onehot_features,
    "C. IID Gaussian":       iid_features,
    "D. Shuffled activs":    shuffled_features,
}

print("=" * 72)
print(f"{'Target':<20s}", end="")
for feat_name in feature_sets:
    print(f"  {feat_name:>16s}", end="")
print()
print("=" * 72)

for target_name, target_vals in targets.items():
    print(f"{target_name:<20s}", end="")
    for feat_name, feat_vals in feature_sets.items():
        r2 = probe_r2(feat_vals, target_vals)
        print(f"  {r2:>16.4f}", end="")
    print()

print("=" * 72)
print()
print("Interpretation:")
print("  Columns A & B should show high R² (input-dependent features).")
print("  Columns C & D should show R² ≈ 0   (no input dependence).")
print()
print("  If A ≈ B >> C ≈ D ≈ 0, the random transformer acts as a reservoir:")
print("  it creates a rich nonlinear feature map of the input from which")
print("  linear probes can decode arbitrary deterministic functions.")
print()
print("  Note: 'parity' may have lower R² even for A & B, since it is a")
print("  highly nonlinear function that is hard for linear probes regardless")
print("  of the feature representation.")
