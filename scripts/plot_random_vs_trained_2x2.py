"""
Generate a 2x2 figure showing R² vs probe iteration at 4 training stages:
  (0,0) Step 0 (random init)
  (0,1) Step 100
  (1,0) Step 2000
  (1,1) Best model (fully trained, ~46k steps)

All panels computed fresh with 32 iterations for consistency.
- Random init: model architecture without trained weights
- Step 100 & 2000: from 20260308_144221 (fine-grained early checkpoints)
- Best model: from 20260306_150920 (fully trained, 10 epochs)
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
import matplotlib.cm as cm
import yaml

from activation_extraction import (
    load_model_from_dir,
    get_process_from_config,
    extract_residual_stream,
    prepare_probe_data,
)
from iterative_probe import iterative_probe_all_layers
from utils import initialize_transformer_from_yaml


# ── Config ──────────────────────────────────────────────────────────────────
EARLY_MODEL_DIR = "models/mess3/20260308_144221_L4_d64_H1_full_LN"
FINAL_MODEL_DIR = "models/mess3/20260306_150920_L4_d64_H1_full_LN"
FIG_DIR = "figures/iterative_probe_emergence"
os.makedirs(FIG_DIR, exist_ok=True)

BATCH_SIZE = 10000
SEQ_LENGTH = 10
MAX_ITERATIONS = 32
R2_THRESHOLD = -1  # no cutoff
TEST_FRACTION = 0.2
DEVICE = "cpu"

# The 4 panels: (model_dir, checkpoint_filename_or_None, label)
PANELS = [
    # (0,0): random init — use EARLY_MODEL_DIR for architecture, no weights
    {"label": "Step 0 (random init)", "model_dir": EARLY_MODEL_DIR,
     "checkpoint": None, "random_init": True},
    # (0,1): step 100
    {"label": "Step 100", "model_dir": EARLY_MODEL_DIR,
     "checkpoint": "checkpoint_step_100.pt", "random_init": False},
    # (1,0): step 2000
    {"label": "Step 2000", "model_dir": EARLY_MODEL_DIR,
     "checkpoint": "checkpoint_step_2000.pt", "random_init": False},
    # (1,1): best model (fully trained)
    {"label": "Best model (~46k steps)", "model_dir": FINAL_MODEL_DIR,
     "checkpoint": "best_model.pt", "random_init": False},
]


# ── Compute probing for one model ───────────────────────────────────────────
def compute_probing(model_dir, checkpoint, random_init, sequences, belief_states):
    """Run iterative probing and return DataFrame with layer, iteration, test_r2."""
    config_path = os.path.join(model_dir, "config.yaml")

    if random_init:
        model = initialize_transformer_from_yaml(config_path)
        model.eval()
        model.to(DEVICE)
    else:
        model = load_model_from_dir(model_dir, checkpoint_filename=checkpoint,
                                     device=DEVICE)

    raw_actvs = extract_residual_stream(model, sequences, device=DEVICE)
    layer_actvs, targets = prepare_probe_data(raw_actvs, belief_states, use_last_pos=True)
    del raw_actvs, model

    results = iterative_probe_all_layers(
        layer_actvs, targets,
        max_iterations=MAX_ITERATIONS,
        r2_threshold=R2_THRESHOLD,
        test_fraction=TEST_FRACTION,
    )

    rows = []
    for lr in results:
        for i, it in enumerate(lr.iterations):
            rows.append({
                "layer": lr.layer,
                "iteration": i,
                "test_r2": it.probe_result.test_r2,
            })
    return pd.DataFrame(rows)


# ── Main ────────────────────────────────────────────────────────────────────
# Generate shared data once (use EARLY_MODEL_DIR config — same process for all)
config_path = os.path.join(EARLY_MODEL_DIR, "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

process = get_process_from_config(config)
print(f"Generating {BATCH_SIZE} sequences of length {SEQ_LENGTH} from {process.__class__.__name__}")
sequences = process.generate_data(batch_size=BATCH_SIZE, length=SEQ_LENGTH, use_tqdm=True)
belief_states = process.mixed_state_presentation(sequences.numpy())

# Compute probing for each panel
panel_data = []
for panel in PANELS:
    print(f"\n{'='*60}")
    print(f"Computing: {panel['label']}")
    print(f"{'='*60}")
    df = compute_probing(
        panel["model_dir"], panel["checkpoint"],
        panel["random_init"], sequences, belief_states,
    )
    panel_data.append(df)

# ── Plot 2x2 ────────────────────────────────────────────────────────────────
layers = sorted(panel_data[0]["layer"].unique())
n_layers = len(layers)
layer_colors = cm.viridis(np.linspace(0.15, 0.85, n_layers))

fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True, sharex=True)

for idx, (panel, data) in enumerate(zip(PANELS, panel_data)):
    row, col = divmod(idx, 2)
    ax = axes[row, col]

    for i, layer in enumerate(layers):
        sub = data[data["layer"] == layer].sort_values("iteration")
        if len(sub) > 0:
            ax.plot(sub["iteration"], sub["test_r2"], "o-",
                    color=layer_colors[i], label=f"Layer {layer}",
                    markersize=3, linewidth=1.5)

    ax.set_title(panel["label"], fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    if row == 1:
        ax.set_xlabel("Probe iteration")
    if col == 0:
        ax.set_ylabel("Test R²")

# Single legend
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=n_layers,
           bbox_to_anchor=(0.5, 1.02), fontsize=9)

fig.suptitle("Emergence of belief-state copies during training", y=1.06, fontsize=13)
fig.tight_layout()
out_path = os.path.join(FIG_DIR, "random_vs_trained.pdf")
fig.savefig(out_path, bbox_inches="tight")
print(f"\nSaved {out_path}")
plt.close(fig)
