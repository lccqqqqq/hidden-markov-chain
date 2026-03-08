import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

# Load both datasets
early_dir = "models/mess3/20260308_144221_L4_d64_H1_full_LN/iterative_probe_checkpoints"
dense_dir = "models/mess3/20260308_141638_L4_d64_H1_full_LN/iterative_probe_checkpoints"
fig_dir = "figures/iterative_probe_emergence"
os.makedirs(fig_dir, exist_ok=True)

early_df = pd.read_csv(os.path.join(early_dir, "all_checkpoints_iterative_metrics.csv"))
dense_df = pd.read_csv(os.path.join(dense_dir, "all_checkpoints_iterative_metrics.csv"))

# Combine: early (steps 100-2000) + dense (steps 2000-46000+)
# Remove step_002000 from dense to avoid duplicate
dense_df = dense_df[dense_df["epoch"] > 2000]
combined = pd.concat([early_df, dense_df], ignore_index=True)
combined = combined.sort_values(["epoch", "layer", "iteration"])

# Save combined
combined.to_csv(os.path.join(fig_dir, "combined_iterative_metrics.csv"), index=False)

epochs = sorted(combined["epoch"].unique())
layers = sorted(combined["layer"].unique())
n_layers = len(layers)
n_epochs = len(epochs)

epoch_labels = {}
for e in epochs:
    names = combined[combined["epoch"] == e]["name"].unique()
    epoch_labels[e] = names[0] if len(names) > 0 else str(e)

layer_colors = cm.viridis(np.linspace(0.15, 0.85, n_layers))

print(f"Combined: {len(combined)} rows, {n_epochs} checkpoints")

# ---- (a) Zoomed heatmap: first 2000 steps, Layer 3 ----
early_epochs = [e for e in epochs if e <= 2000]
focus_layer = layers[-1]
sub = combined[(combined["layer"] == focus_layer) & (combined["epoch"].isin(early_epochs))]
iterations = sorted(sub["iteration"].unique())
n_iter = len(iterations)
n_early = len(early_epochs)

r2_matrix = np.full((n_iter, n_early), np.nan)
for j, epoch in enumerate(early_epochs):
    for i, it in enumerate(iterations):
        row = sub[(sub["epoch"] == epoch) & (sub["iteration"] == it)]
        if len(row) > 0:
            r2_matrix[i, j] = row["test_r2"].values[0]

fig, ax = plt.subplots(figsize=(8, 4.5))
im = ax.imshow(r2_matrix, aspect="auto", cmap="RdYlGn", origin="lower",
               interpolation="nearest", vmin=0.0, vmax=1.0)
ax.set_xticks(range(0, n_early, 2))
ax.set_xticklabels([epoch_labels[early_epochs[i]] for i in range(0, n_early, 2)],
                   rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(0, n_iter, 2))
ax.set_yticklabels([str(iterations[i]) for i in range(0, n_iter, 2)])
ax.set_xlabel("Training step")
ax.set_ylabel("Probe iteration")
ax.set_title(f"Onset of belief-state copies (Layer {focus_layer}, first 2000 steps)")
fig.colorbar(im, ax=ax, label="Test R²", shrink=0.8)
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, "onset_r2_heatmap.pdf"), bbox_inches="tight")
print("Saved onset_r2_heatmap.pdf")
plt.close(fig)

# ---- (b) First copy R² vs step: full range with zoomed inset ----
fig, ax = plt.subplots(figsize=(7, 4.5))

for i, layer in enumerate(layers):
    first_r2 = []
    plot_epochs = []
    for epoch in epochs:
        row = combined[(combined["epoch"] == epoch) & (combined["layer"] == layer) & (combined["iteration"] == 0)]
        if len(row) > 0:
            first_r2.append(row["test_r2"].values[0])
            plot_epochs.append(epoch)
    ax.plot(plot_epochs, first_r2, "-", color=layer_colors[i],
            label=f"Layer {layer}", linewidth=1.5)

ax.set_xlabel("Training step")
ax.set_ylabel("Test R² (first probe)")
ax.set_title("First copy emergence: full training trajectory")
ax.legend()
ax.grid(True, alpha=0.3)

# Inset: zoom on first 2000 steps
axins = ax.inset_axes([0.45, 0.15, 0.5, 0.5])
for i, layer in enumerate(layers):
    first_r2 = []
    plot_epochs = []
    for epoch in early_epochs:
        row = combined[(combined["epoch"] == epoch) & (combined["layer"] == layer) & (combined["iteration"] == 0)]
        if len(row) > 0:
            first_r2.append(row["test_r2"].values[0])
            plot_epochs.append(epoch)
    axins.plot(plot_epochs, first_r2, "o-", color=layer_colors[i], markersize=3, linewidth=1)

axins.set_xlabel("Step", fontsize=8)
axins.set_ylabel("R²", fontsize=8)
axins.set_title("First 2000 steps", fontsize=8)
axins.tick_params(labelsize=7)
axins.grid(True, alpha=0.3)
ax.indicate_inset_zoom(axins, edgecolor="gray")

fig.tight_layout()
fig.savefig(os.path.join(fig_dir, "first_copy_r2_full_with_inset.pdf"), bbox_inches="tight")
print("Saved first_copy_r2_full_with_inset.pdf")
plt.close(fig)

# ---- (c) R² profiles at very early steps ----
# Pick 4 early checkpoints
if len(early_epochs) >= 4:
    sel_idx = [0, len(early_epochs)//3, 2*len(early_epochs)//3, len(early_epochs)-1]
else:
    sel_idx = list(range(len(early_epochs)))
seen = set()
sel_idx = [i for i in sel_idx if not (i in seen or seen.add(i))]
sel_epochs = [early_epochs[i] for i in sel_idx]

fig, axes = plt.subplots(1, len(sel_epochs), figsize=(3.5 * len(sel_epochs), 3.5), sharey=True)
if len(sel_epochs) == 1:
    axes = [axes]

for ax, epoch in zip(axes, sel_epochs):
    for i, layer in enumerate(layers):
        sub = combined[(combined["epoch"] == epoch) & (combined["layer"] == layer)]
        if len(sub) > 0:
            ax.plot(sub["iteration"], sub["test_r2"], "o-",
                    color=layer_colors[i], label=f"Layer {layer}", markersize=4)
    ax.set_xlabel("Iteration")
    ax.set_title(epoch_labels[epoch])
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("Test R²")
axes[-1].legend(loc="lower left", fontsize=7)
fig.suptitle("R² profiles at early training steps", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, "r2_profiles_early_steps.pdf"), bbox_inches="tight")
print("Saved r2_profiles_early_steps.pdf")
plt.close(fig)

# ---- (d) Number of copies (R² > threshold) vs step, zoomed ----
fig, ax = plt.subplots(figsize=(7, 4))
for threshold in [0.5, 0.9, 0.99]:
    for i, layer in enumerate(layers):
        counts = []
        plot_epochs = []
        for epoch in epochs:
            sub = combined[(combined["epoch"] == epoch) & (combined["layer"] == layer)]
            if len(sub) > 0:
                counts.append((sub["test_r2"] > threshold).sum())
                plot_epochs.append(epoch)
        style = "-" if threshold == 0.9 else ("--" if threshold == 0.5 else ":")
        alpha = 1.0 if threshold == 0.9 else 0.5
        label = f"L{layer}" if threshold == 0.9 else None
        ax.plot(plot_epochs, counts, style, color=layer_colors[i],
                linewidth=1.5, alpha=alpha, label=label)

# Add threshold labels manually
ax.text(epochs[-1]*1.02, 15.5, "R²>0.5", fontsize=7, color="gray", style="italic")
ax.text(epochs[-1]*1.02, 14, "R²>0.9", fontsize=7, color="gray")
ax.text(epochs[-1]*1.02, 12, "R²>0.99", fontsize=7, color="gray", style="italic")

ax.set_xlabel("Training step")
ax.set_ylabel("Number of decodable copies")
ax.set_title("Emergence of decodable copies during training")
ax.legend(title="R²>0.9", fontsize=8, title_fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, "n_copies_vs_step.pdf"), bbox_inches="tight")
print("Saved n_copies_vs_step.pdf")
plt.close(fig)

print("\nAll combined figures saved to", fig_dir)

# ---- Summary: when does first R² exceed thresholds? ----
print("\n=== Emergence Summary ===")
for layer in layers:
    print(f"\nLayer {layer}:")
    for thresh_name, thresh in [("0.5", 0.5), ("0.9", 0.9), ("0.99", 0.99)]:
        first_step = None
        for epoch in epochs:
            row = combined[(combined["epoch"] == epoch) & (combined["layer"] == layer) & (combined["iteration"] == 0)]
            if len(row) > 0 and row["test_r2"].values[0] > thresh:
                first_step = epoch
                break
        if first_step is not None:
            print(f"  First R² > {thresh_name} at step {first_step} ({epoch_labels[first_step]})")
        else:
            print(f"  First R² never exceeds {thresh_name}")
