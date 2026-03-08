"""
Summary plots for iterative probing and counterfactual importance results.
Run interactively cell-by-cell in an IDE (e.g., VS Code with #%% markers).
"""

# %% Imports and data loading
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Ensure we're in the project root regardless of where the IDE starts
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

results_dir = "models/mess3/20260306_150920_L4_d64_H1_full_LN/iterative_probe_results"

probe_df = pd.read_csv(f"{results_dir}/iterative_metrics.csv")
cf_df = pd.read_csv(f"{results_dir}/counterfactual_metrics.csv")

n_layers = probe_df["layer"].nunique()
layer_colors = cm.viridis(np.linspace(0.15, 0.85, n_layers))

print(f"Layers: {n_layers}, max iterations: {probe_df['iteration'].max() + 1}")
print(f"Counterfactual ablation types: {cf_df['ablation_type'].unique()}")

# %% Plot 1: Test R² vs iteration (one line per layer)
fig, ax = plt.subplots(figsize=(6, 4))

for layer in range(n_layers):
    ld = probe_df[probe_df["layer"] == layer]
    ax.plot(ld["iteration"], ld["test_r2"], "o-",
            color=layer_colors[layer], label=f"Layer {layer}", markersize=5)

ax.set_xlabel("Iteration (probe-project cycle)")
ax.set_ylabel("Test R²")
ax.set_title("Iterative probing: R² after successive projections")
ax.legend()
ax.set_ylim(0.98, 1.0)
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()

# %% Plot 2: Test MSE vs cumulative dimensions projected out
fig, ax = plt.subplots(figsize=(6, 4))

for layer in range(n_layers):
    ld = probe_df[probe_df["layer"] == layer]
    ax.plot(ld["cumulative_dims"], ld["test_mse"], "o-",
            color=layer_colors[layer], label=f"Layer {layer}", markersize=5)

ax.set_xlabel("Cumulative dimensions projected out")
ax.set_ylabel("Test MSE")
ax.set_title("Probe MSE vs. dimensions removed")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()

# %% Plot 3: Counterfactual KL — subspace ablation per iteration (grouped bar chart)
sub_df = cf_df[(cf_df["ablation_type"] == "subspace")].copy()

fig, ax = plt.subplots(figsize=(8, 4))

bar_width = 0.18
iterations = sorted(sub_df["iteration"].unique())

for i, layer in enumerate(range(n_layers)):
    ld = sub_df[sub_df["layer"] == layer]
    x = np.array(iterations) + i * bar_width
    ax.bar(x, ld["mean_kl"].values, bar_width,
           color=layer_colors[layer], label=f"Layer {layer}", alpha=0.85,
           yerr=ld["std_kl"].values, capsize=2, error_kw={"linewidth": 0.8})

ax.set_xlabel("Iteration")
ax.set_ylabel("Mean KL(clean || ablated)")
ax.set_title("Counterfactual importance of each probe subspace")
ax.set_xticks(np.array(iterations) + bar_width * (n_layers - 1) / 2)
ax.set_xticklabels(iterations)
ax.legend()
ax.set_yscale("log")
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
plt.show()

# %% Plot 4: Cumulative-progressive KL vs dimensions ablated
cprog = cf_df[cf_df["ablation_type"] == "cumulative_progressive"].copy()

if cprog.empty:
    print("No cumulative_progressive data found — skipping Plot 4.")
else:
    fig, ax = plt.subplots(figsize=(7, 4))

    for layer in range(n_layers):
        ld = cprog[cprog["layer"] == layer].sort_values("iteration")
        cum_dims = (ld["iteration"].values + 1) * 2
        ax.plot(cum_dims, ld["mean_kl"].values, "o-",
                color=layer_colors[layer], label=f"Layer {layer}", markersize=3)
        ax.fill_between(cum_dims,
                        ld["mean_kl"].values - ld["std_kl"].values,
                        ld["mean_kl"].values + ld["std_kl"].values,
                        color=layer_colors[layer], alpha=0.15)

    ax.set_xlabel("Cumulative dimensions ablated")
    ax.set_ylabel("KL(clean || ablated)")
    ax.set_title("Cumulative-progressive ablation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()

# %% Plot 5: Combined — R² (left) and KL (right) vs iteration, one panel per layer
fig, axes = plt.subplots(1, n_layers, figsize=(14, 3.5), sharey=False)

for layer, ax in enumerate(axes):
    ld_probe = probe_df[probe_df["layer"] == layer]
    ld_cf = sub_df[sub_df["layer"] == layer]

    color_r2 = layer_colors[layer]
    color_kl = "firebrick"

    ax.plot(ld_probe["iteration"], ld_probe["test_r2"], "o-",
            color=color_r2, markersize=4, label="Test R²")
    ax.set_ylim(0.98, 1.0)
    ax.set_xlabel("Iteration")
    if layer == 0:
        ax.set_ylabel("Test R²", color=color_r2)
    ax.tick_params(axis="y", labelcolor=color_r2)

    ax2 = ax.twinx()
    ax2.bar(ld_cf["iteration"].values, ld_cf["mean_kl"].values,
            alpha=0.4, color=color_kl, label="KL")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-6, 0.1)
    if layer == n_layers - 1:
        ax2.set_ylabel("KL(clean || ablated)", color=color_kl)
    ax2.tick_params(axis="y", labelcolor=color_kl)

    ax.set_title(f"Layer {layer}")
    ax.grid(True, alpha=0.2)

fig.suptitle("Iterative probing: decodability (R²) vs. causal importance (KL)", y=1.02)
fig.tight_layout()
plt.show()

# %% Plot 6: Singular values across iterations (heatmap per layer)
fig, axes = plt.subplots(1, n_layers, figsize=(14, 3.5), sharey=True)

for layer, ax in enumerate(axes):
    ld = probe_df[probe_df["layer"] == layer]
    sv_matrix = []
    for _, row in ld.iterrows():
        svs = [float(x) for x in row["singular_values"].split(",")]
        sv_matrix.append(svs)
    sv_matrix = np.array(sv_matrix)  # (n_iter, rank)

    im = ax.imshow(sv_matrix.T, aspect="auto", cmap="magma",
                   origin="lower", interpolation="nearest")
    ax.set_xlabel("Iteration")
    ax.set_title(f"Layer {layer}")
    if layer == 0:
        ax.set_ylabel("SVD component")
    ax.set_yticks(range(sv_matrix.shape[1]))
    ax.set_yticklabels([f"σ{i}" for i in range(sv_matrix.shape[1])])

fig.colorbar(im, ax=axes[-1], label="Singular value", shrink=0.8)
fig.suptitle("Probe coefficient singular values per iteration", y=1.02)
fig.tight_layout()
plt.show()
