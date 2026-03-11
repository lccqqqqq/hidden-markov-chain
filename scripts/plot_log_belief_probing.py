"""
Iterative probing with log(b) targets instead of b.

Produces two figures:
  1. log_belief_r2_vs_iteration.pdf — R² vs probe iteration per layer (like Fig 2)
  2. log_belief_random_vs_trained.pdf — 2×2 training dynamics (like Fig 6)

Also overlays b vs log(b) comparison for the best model.

Usage:
    python scripts/plot_log_belief_probing.py
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
FIG_DIR = "figures/log_belief_probing"
os.makedirs(FIG_DIR, exist_ok=True)

BATCH_SIZE = 10_000
SEQ_LENGTH = 10
MAX_ITERATIONS = 32
R2_THRESHOLD = -1  # no cutoff
TEST_FRACTION = 0.2
DEVICE = "cpu"
SEED = 42

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


# ── Helpers ─────────────────────────────────────────────────────────────────

def compute_probing(model_dir, checkpoint, random_init, sequences,
                    targets_b, targets_logb):
    """Run iterative probing for both b and log(b), return DataFrames."""
    config_path = os.path.join(model_dir, "config.yaml")

    if random_init:
        model = initialize_transformer_from_yaml(config_path)
        model.eval()
        model.to(DEVICE)
    else:
        model = load_model_from_dir(model_dir, checkpoint_filename=checkpoint,
                                     device=DEVICE)

    raw_actvs = extract_residual_stream(model, sequences, device=DEVICE)
    # prepare_probe_data returns (layer_actvs, _) — we supply our own targets
    layer_actvs, _ = prepare_probe_data(raw_actvs, belief_states, use_last_pos=True)
    del raw_actvs, model

    def run_probe(tgts, label):
        results = iterative_probe_all_layers(
            layer_actvs, tgts,
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
                    "target": label,
                })
        return pd.DataFrame(rows)

    df_b = run_probe(targets_b, "b")
    df_logb = run_probe(targets_logb, "log(b)")
    return pd.concat([df_b, df_logb], ignore_index=True), layer_actvs


# ── Generate shared data ────────────────────────────────────────────────────
config_path = os.path.join(EARLY_MODEL_DIR, "config.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

process = get_process_from_config(config)
print(f"Generating {BATCH_SIZE} sequences of length {SEQ_LENGTH}")
sequences = process.generate_data(batch_size=BATCH_SIZE, length=SEQ_LENGTH, use_tqdm=True)
belief_states = process.mixed_state_presentation(sequences.numpy())

# Targets: last-position belief state and its log
b_last = belief_states[:, -1, :]  # (N, 3)
LOG_EPS = 1e-10
logb_last = np.log(np.clip(b_last, LOG_EPS, None))  # (N, 3)

print(f"Belief state range: [{b_last.min():.4f}, {b_last.max():.4f}]")
print(f"Log-belief range:   [{logb_last.min():.2f}, {logb_last.max():.2f}]")

# Check how linear log(b) is in b
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
reg = LinearRegression().fit(b_last, logb_last)
print(f"R² of log(b) predicted linearly from b: {r2_score(logb_last, reg.predict(b_last)):.4f}")
print()


# ── Figure 1: R² vs iteration for best model (b vs log(b) overlay) ─────────
print("=" * 60)
print("Computing: Best model — b vs log(b) comparison")
print("=" * 60)

df_best, _ = compute_probing(
    FINAL_MODEL_DIR, "best_model.pt", False,
    sequences, b_last, logb_last,
)

layers = sorted(df_best["layer"].unique())
n_layers = len(layers)
layer_colors = cm.viridis(np.linspace(0.15, 0.85, n_layers))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

for target_label, ax, title in [
    ("b", axes[0], r"Target: belief state $b$"),
    ("log(b)", axes[1], r"Target: $\log(b)$"),
]:
    sub = df_best[df_best["target"] == target_label]
    for i, layer in enumerate(layers):
        lsub = sub[sub["layer"] == layer].sort_values("iteration")
        ax.plot(lsub["iteration"], lsub["test_r2"], "o-",
                color=layer_colors[i], label=f"Layer {layer}",
                markersize=3, linewidth=1.5)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Probe iteration")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

axes[0].set_ylabel("Test R²")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=n_layers,
           bbox_to_anchor=(0.5, 1.02), fontsize=9)
fig.suptitle("Iterative probing: belief state vs log-belief state (best model)",
             y=1.08, fontsize=13)
fig.tight_layout()
out1 = os.path.join(FIG_DIR, "log_belief_r2_vs_iteration.pdf")
fig.savefig(out1, bbox_inches="tight")
print(f"\nSaved {out1}")
plt.close(fig)


# ── Figure 2: 2×2 training dynamics with log(b) target ──────────────────────
panel_data = []
# Reuse best model result for panel 3
for pidx, panel in enumerate(PANELS):
    if pidx == 3:
        # Already computed above
        df_logb_only = df_best[df_best["target"] == "log(b)"].copy()
        panel_data.append(df_logb_only)
        continue
    print(f"\n{'='*60}")
    print(f"Computing: {panel['label']}")
    print(f"{'='*60}")
    df_both, _ = compute_probing(
        panel["model_dir"], panel["checkpoint"],
        panel["random_init"], sequences, b_last, logb_last,
    )
    panel_data.append(df_both[df_both["target"] == "log(b)"].copy())

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

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=n_layers,
           bbox_to_anchor=(0.5, 1.02), fontsize=9)
fig.suptitle(r"Emergence of $\log(b)$ copies during training", y=1.06, fontsize=13)
fig.tight_layout()
out2 = os.path.join(FIG_DIR, "log_belief_random_vs_trained.pdf")
fig.savefig(out2, bbox_inches="tight")
print(f"\nSaved {out2}")
plt.close(fig)

print("\nDone.")
