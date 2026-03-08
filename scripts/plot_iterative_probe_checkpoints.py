"""
Visualization of iterative probing across training checkpoints.

Reads all_checkpoints_iterative_metrics.csv and produces 4 PDF figures
showing how belief-state copy quality evolves during training.

Usage:
    python scripts/plot_iterative_probe_checkpoints.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_DIR = "models/mess3/20260308_141638_L4_d64_H1_full_LN"
RESULTS_DIR = os.path.join(MODEL_DIR, "iterative_probe_checkpoints")
FIGURES_DIR = "figures/iterative_probe_emergence"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

csv_path = os.path.join(RESULTS_DIR, "all_checkpoints_iterative_metrics.csv")
if not os.path.exists(csv_path):
    print(f"ERROR: {csv_path} not found. Run iterative_probe_multi_checkpoint.py first.")
    sys.exit(1)

df = pd.read_csv(csv_path)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Parse epochs and layers
epochs = sorted(df["epoch"].unique())
layers = sorted(df["layer"].unique())
n_layers = len(layers)
n_epochs = len(epochs)

# Map epoch number to display label
epoch_labels = {}
for e in epochs:
    names = df[df["epoch"] == e]["name"].unique()
    epoch_labels[e] = names[0] if len(names) > 0 else str(e)

layer_colors = cm.viridis(np.linspace(0.15, 0.85, n_layers))

print(f"Loaded {len(df)} rows: {n_epochs} checkpoints x {n_layers} layers")
print(f"Epochs: {[epoch_labels[e] for e in epochs]}")

# ---------------------------------------------------------------------------
# (a) Heatmap: R² vs epoch x iteration for Layer 3 (best layer)
# ---------------------------------------------------------------------------

focus_layer = layers[-1]  # Layer 3
sub = df[df["layer"] == focus_layer]
iterations = sorted(sub["iteration"].unique())
n_iter = len(iterations)

r2_matrix = np.full((n_iter, n_epochs), np.nan)
for j, epoch in enumerate(epochs):
    for i, it in enumerate(iterations):
        row = sub[(sub["epoch"] == epoch) & (sub["iteration"] == it)]
        if len(row) > 0:
            r2_matrix[i, j] = row["test_r2"].values[0]

fig, ax = plt.subplots(figsize=(8, 4.5))
im = ax.imshow(r2_matrix, aspect="auto", cmap="RdYlGn", origin="lower",
               interpolation="nearest", vmin=0.92, vmax=1.0)

ax.set_xticks(range(n_epochs))
ax.set_xticklabels([epoch_labels[e] for e in epochs], rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(0, n_iter, 2))
ax.set_yticklabels([str(iterations[i]) for i in range(0, n_iter, 2)])
ax.set_xlabel("Training checkpoint")
ax.set_ylabel("Probe iteration")
ax.set_title(f"Test R² per probe iteration (Layer {focus_layer})")

fig.colorbar(im, ax=ax, label="Test R²", shrink=0.8)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "r2_heatmap_vs_epoch.pdf"), bbox_inches="tight")
print("Saved r2_heatmap_vs_epoch.pdf")
plt.close(fig)

# ---------------------------------------------------------------------------
# (b) Line plot: R² of first probe (iteration 0) vs epoch, per layer
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(6, 4))

for i, layer in enumerate(layers):
    first_r2 = []
    plot_epochs = []
    for epoch in epochs:
        row = df[(df["epoch"] == epoch) & (df["layer"] == layer) & (df["iteration"] == 0)]
        if len(row) > 0:
            first_r2.append(row["test_r2"].values[0])
            plot_epochs.append(epoch)

    ax.plot(plot_epochs, first_r2, "o-", color=layer_colors[i],
            label=f"Layer {layer}", markersize=5)

ax.set_xlabel("Epoch")
ax.set_ylabel("Test R² (first probe)")
ax.set_title("First copy decodability during training")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(epochs)
ax.set_xticklabels([epoch_labels[e] for e in epochs], rotation=45, ha="right", fontsize=8)
# Zoom to show the variation
ymin = ax.get_ylim()[0]
ax.set_ylim(max(ymin, 0.99), 1.001)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "first_copy_r2_vs_epoch.pdf"), bbox_inches="tight")
print("Saved first_copy_r2_vs_epoch.pdf")
plt.close(fig)

# ---------------------------------------------------------------------------
# (c) Line plot: R² at last iteration vs epoch, per layer
#     Shows how the "worst copy" quality improves during training
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(6, 4))

max_iter = df["iteration"].max()

for i, layer in enumerate(layers):
    last_r2 = []
    plot_epochs = []
    for epoch in epochs:
        row = df[(df["epoch"] == epoch) & (df["layer"] == layer) & (df["iteration"] == max_iter)]
        if len(row) > 0:
            last_r2.append(row["test_r2"].values[0])
            plot_epochs.append(epoch)

    ax.plot(plot_epochs, last_r2, "o-", color=layer_colors[i],
            label=f"Layer {layer}", markersize=5)

ax.set_xlabel("Epoch")
ax.set_ylabel(f"Test R² (iteration {max_iter})")
ax.set_title(f"Last-copy quality during training (iteration {max_iter})")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(epochs)
ax.set_xticklabels([epoch_labels[e] for e in epochs], rotation=45, ha="right", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "last_copy_r2_vs_epoch.pdf"), bbox_inches="tight")
print("Saved last_copy_r2_vs_epoch.pdf")
plt.close(fig)

# ---------------------------------------------------------------------------
# (d) R² profiles at selected epochs (staircase formation)
# ---------------------------------------------------------------------------

# Pick 4 epochs spread across training
if n_epochs >= 4:
    selected_indices = [0, n_epochs // 3, 2 * n_epochs // 3, n_epochs - 1]
else:
    selected_indices = list(range(n_epochs))

# Deduplicate while preserving order
seen = set()
selected_indices = [i for i in selected_indices if not (i in seen or seen.add(i))]
selected_epochs = [epochs[i] for i in selected_indices]

fig, axes = plt.subplots(1, len(selected_epochs), figsize=(3.5 * len(selected_epochs), 3.5),
                         sharey=True)
if len(selected_epochs) == 1:
    axes = [axes]

for ax, epoch in zip(axes, selected_epochs):
    for i, layer in enumerate(layers):
        sub = df[(df["epoch"] == epoch) & (df["layer"] == layer)]
        if len(sub) > 0:
            ax.plot(sub["iteration"], sub["test_r2"], "o-",
                    color=layer_colors[i], label=f"Layer {layer}", markersize=4)

    ax.set_xlabel("Iteration")
    ax.set_title(epoch_labels[epoch])
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("Test R²")
axes[-1].legend(loc="lower left", fontsize=7)

fig.suptitle("R² vs iteration at selected training checkpoints", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "r2_profiles_selected_epochs.pdf"), bbox_inches="tight")
print("Saved r2_profiles_selected_epochs.pdf")
plt.close(fig)

print("\nAll figures saved to", FIGURES_DIR)
