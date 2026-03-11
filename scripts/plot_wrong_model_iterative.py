"""
Iterative probing for wrong-model belief states across training stages.

Produces a 2×2 figure (like Fig 6) showing R² vs probe iteration at 4 training
stages, comparing true vs wrong-model belief states. For a well-trained model,
the wrong-model copies should decay much faster than the true ones.

Usage:
    python scripts/plot_wrong_model_iterative.py
"""
import sys
import os

os.chdir(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from activation_extraction import (
    load_model_from_dir,
    get_process_from_config,
    extract_residual_stream,
    prepare_probe_data,
)
from iterative_probe import iterative_probe_single_layer
from utils import initialize_transformer_from_yaml

# ── Config ──────────────────────────────────────────────────────────────────
EARLY_MODEL_DIR = "models/mess3/20260308_144221_L4_d64_H1_full_LN"
FINAL_MODEL_DIR = "models/mess3/20260306_150920_L4_d64_H1_full_LN"
FIG_DIR = "figures/wrong_model_probing"
os.makedirs(FIG_DIR, exist_ok=True)

BATCH_SIZE = 10_000
SEQ_LENGTH = 10
MAX_ITERATIONS = 32
R2_THRESHOLD = -1
TEST_FRACTION = 0.2
DEVICE = "cpu"
SEED = 42
PROBE_LAYER = 3  # focus on the causally important layer
N_WRONG_MODELS = 5

np.random.seed(SEED)

PANELS = [
    {"label": "Step 0 (random init)", "model_dir": EARLY_MODEL_DIR,
     "checkpoint": None, "random_init": True},
    {"label": "Step 100", "model_dir": EARLY_MODEL_DIR,
     "checkpoint": "checkpoint_step_100.pt", "random_init": False},
    {"label": "Step 2000", "model_dir": EARLY_MODEL_DIR,
     "checkpoint": "checkpoint_step_2000.pt", "random_init": False},
    {"label": "Best model (~46k steps)", "model_dir": FINAL_MODEL_DIR,
     "checkpoint": "best_model.pt", "random_init": False},
]


# ── Generate data and wrong-model beliefs ───────────────────────────────────
config_path = os.path.join(EARLY_MODEL_DIR, "config.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

process = get_process_from_config(config)
print(f"Generating {BATCH_SIZE} sequences of length {SEQ_LENGTH}")
sequences = process.generate_data(batch_size=BATCH_SIZE, length=SEQ_LENGTH, use_tqdm=True)
seq_np = sequences.numpy()
belief_states = process.mixed_state_presentation(seq_np)
true_beliefs_last = belief_states[:, -1, :]  # (N, 3)

n_states = process.num_hidden_states
vocab_size = config["model"]["vocab_size"]


def compute_wrong_beliefs(seq_np, n_hidden, d_vocab, rng_seed):
    """Compute belief states under a random HMM."""
    rng = np.random.RandomState(rng_seed)
    T = rng.dirichlet(np.ones(n_hidden), size=n_hidden)
    O = rng.dirichlet(np.ones(d_vocab), size=n_hidden).T
    N, L = seq_np.shape
    beliefs = np.zeros((N, L, n_hidden))
    for i in range(N):
        b = np.ones(n_hidden) / n_hidden
        for pos in range(L):
            tok = seq_np[i, pos]
            b_pred = T.T @ b
            b = O[tok, :] * b_pred
            b = b / (b.sum() + 1e-15)
            beliefs[i, pos, :] = b
    return beliefs[:, -1, :]


print(f"Computing {N_WRONG_MODELS} wrong-model belief states...")
wrong_beliefs_list = []
for i in range(N_WRONG_MODELS):
    wb = compute_wrong_beliefs(seq_np, n_states, vocab_size, rng_seed=1000 + i)
    wrong_beliefs_list.append(wb)
    print(f"  Wrong model {i}: done")


# ── Run iterative probing ───────────────────────────────────────────────────
def extract_layer_activations(model_dir, checkpoint, random_init, sequences, belief_states):
    """Extract layer-3 activations."""
    cp = os.path.join(model_dir, "config.yaml")
    if random_init:
        model = initialize_transformer_from_yaml(cp)
        model.eval().to(DEVICE)
    else:
        model = load_model_from_dir(model_dir, checkpoint_filename=checkpoint, device=DEVICE)

    raw = extract_residual_stream(model, sequences, device=DEVICE)
    layer_actvs, _ = prepare_probe_data(raw, belief_states, use_last_pos=True)
    del model, raw
    return layer_actvs[PROBE_LAYER]  # (N, 64)


def run_iterative_probe(activations, targets):
    """Run iterative probing, return list of R² values."""
    result = iterative_probe_single_layer(
        activations, targets,
        max_iterations=MAX_ITERATIONS,
        r2_threshold=R2_THRESHOLD,
        test_fraction=TEST_FRACTION,
    )
    return [it.probe_result.test_r2 for it in result.iterations]


# Collect results
all_results = []

for pidx, panel in enumerate(PANELS):
    print(f"\n{'='*60}")
    print(f"Computing: {panel['label']}")
    print(f"{'='*60}")

    actvs = extract_layer_activations(
        panel["model_dir"], panel["checkpoint"],
        panel["random_init"], sequences, belief_states,
    )

    # True beliefs
    print("  Probing true beliefs...")
    r2s_true = run_iterative_probe(actvs, true_beliefs_last)
    for i, r2 in enumerate(r2s_true):
        all_results.append({
            "panel": pidx, "label": panel["label"],
            "iteration": i, "test_r2": r2, "target": "True model",
        })

    # Wrong-model beliefs (run each, collect mean ± std)
    wrong_r2s_all = []
    for widx, wb in enumerate(wrong_beliefs_list):
        print(f"  Probing wrong model {widx}...")
        r2s_wrong = run_iterative_probe(actvs, wb)
        wrong_r2s_all.append(r2s_wrong)
        for i, r2 in enumerate(r2s_wrong):
            all_results.append({
                "panel": pidx, "label": panel["label"],
                "iteration": i, "test_r2": r2,
                "target": f"Wrong model {widx}",
            })

    # Also store mean wrong for plotting
    wrong_arr = np.array(wrong_r2s_all)  # (N_WRONG_MODELS, n_iters)
    for i in range(wrong_arr.shape[1]):
        all_results.append({
            "panel": pidx, "label": panel["label"],
            "iteration": i, "test_r2": wrong_arr[:, i].mean(),
            "target": "Wrong model (mean)",
        })

df = pd.DataFrame(all_results)


# ── Plot 2×2 ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True, sharex=True)

for pidx, panel in enumerate(PANELS):
    row, col = divmod(pidx, 2)
    ax = axes[row, col]
    sub = df[df["panel"] == pidx]

    # True model — thick line
    true_sub = sub[sub["target"] == "True model"].sort_values("iteration")
    ax.plot(true_sub["iteration"], true_sub["test_r2"], "o-",
            color="steelblue", linewidth=2, markersize=3, label="True model", zorder=5)

    # Individual wrong models — thin gray lines
    for widx in range(N_WRONG_MODELS):
        wsub = sub[sub["target"] == f"Wrong model {widx}"].sort_values("iteration")
        ax.plot(wsub["iteration"], wsub["test_r2"], "-",
                color="gray", alpha=0.3, linewidth=0.8,
                label="Wrong models" if widx == 0 else None)

    # Mean wrong model — dashed red
    mean_sub = sub[sub["target"] == "Wrong model (mean)"].sort_values("iteration")
    ax.plot(mean_sub["iteration"], mean_sub["test_r2"], "s--",
            color="firebrick", linewidth=1.5, markersize=2.5, label="Wrong (mean)", zorder=4)

    ax.set_title(panel["label"], fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    if row == 1:
        ax.set_xlabel("Probe iteration")
    if col == 0:
        ax.set_ylabel(f"Test R² (Layer {PROBE_LAYER})")

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3,
           bbox_to_anchor=(0.5, 1.02), fontsize=9)
fig.suptitle(f"True vs wrong-model belief states: iterative probing (Layer {PROBE_LAYER})",
             y=1.06, fontsize=13)
fig.tight_layout()
out_path = os.path.join(FIG_DIR, "wrong_model_random_vs_trained.pdf")
fig.savefig(out_path, bbox_inches="tight")
print(f"\nSaved {out_path}")
plt.close(fig)

print("\nDone.")
