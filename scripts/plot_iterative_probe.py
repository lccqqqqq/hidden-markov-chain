#!/usr/bin/env python3
"""Generate 5 publication-quality PDF figures from iterative probe results."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import os

# Paths
BASE_DIR = '/mnt/users/clin/workspace/hidden-markov-chain'
RESULTS_DIR = os.path.join(
    BASE_DIR,
    'models/mess3/20260306_150920_L4_d64_H1_full_LN/iterative_probe_results'
)
FIG_DIR = os.path.join(BASE_DIR, 'figures/iterative_probe')
os.makedirs(FIG_DIR, exist_ok=True)

# Load data
iter_df = pd.read_csv(os.path.join(RESULTS_DIR, 'iterative_metrics.csv'))
cf_df = pd.read_csv(os.path.join(RESULTS_DIR, 'counterfactual_metrics.csv'))

# Common setup
layers = sorted(iter_df['layer'].unique())
n_layers = len(layers)
layer_colors = cm.viridis(np.linspace(0.15, 0.85, n_layers))

# ============================================================
# Fig 1: r2_vs_iteration.pdf
# ============================================================
fig, ax = plt.subplots(figsize=(6, 4))
for i, layer in enumerate(layers):
    subset = iter_df[iter_df['layer'] == layer].sort_values('iteration')
    ax.plot(subset['iteration'], subset['test_r2'],
            marker='o', markersize=4, color=layer_colors[i],
            label=f'Layer {layer}')
ax.set_xlabel('Iteration')
ax.set_ylabel('Test R²')
ax.set_ylim(0.98, 1.0)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'r2_vs_iteration.pdf'))
plt.close(fig)
print('Saved r2_vs_iteration.pdf')

# ============================================================
# Fig 2: mse_vs_dims.pdf
# ============================================================
fig, ax = plt.subplots(figsize=(6, 4))
for i, layer in enumerate(layers):
    subset = iter_df[iter_df['layer'] == layer].sort_values('cumulative_dims')
    ax.plot(subset['cumulative_dims'], subset['test_mse'],
            marker='o', markersize=4, color=layer_colors[i],
            label=f'Layer {layer}')
ax.set_xlabel('Cumulative dimensions projected out')
ax.set_ylabel('Test MSE')
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'mse_vs_dims.pdf'))
plt.close(fig)
print('Saved mse_vs_dims.pdf')

# ============================================================
# Fig 3: kl_per_iteration.pdf — grouped bar chart
# ============================================================
subspace_df = cf_df[cf_df['ablation_type'] == 'subspace'].copy()
iterations = sorted(subspace_df['iteration'].unique())
n_iters = len(iterations)

fig, ax = plt.subplots(figsize=(8, 4))
bar_width = 0.18
x = np.arange(n_iters)

for i, layer in enumerate(layers):
    subset = subspace_df[subspace_df['layer'] == layer].sort_values('iteration')
    offset = (i - (n_layers - 1) / 2) * bar_width
    ax.bar(x + offset, subset['mean_kl'].values, bar_width,
           yerr=subset['std_kl'].values, capsize=2,
           color=layer_colors[i], label=f'Layer {layer}')

ax.set_yscale('log')
ax.set_xlabel('Iteration')
ax.set_ylabel('Mean KL')
ax.set_xticks(x)
ax.set_xticklabels(iterations)
ax.legend()
ax.grid(alpha=0.3, which='both')
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'kl_per_iteration.pdf'))
plt.close(fig)
print('Saved kl_per_iteration.pdf')

# ============================================================
# Fig 4: cumulative_kl.pdf — bar chart
# ============================================================
cumulative_df = cf_df[cf_df['ablation_type'] == 'cumulative'].copy()
cumulative_df = cumulative_df.sort_values('layer')

fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(n_layers)
ax.bar(x, cumulative_df['mean_kl'].values,
       yerr=cumulative_df['std_kl'].values, capsize=3,
       color=layer_colors)
ax.set_xlabel('Layer')
ax.set_ylabel('Cumulative KL')
ax.set_xticks(x)
ax.set_xticklabels([str(l) for l in layers])
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'cumulative_kl.pdf'))
plt.close(fig)
print('Saved cumulative_kl.pdf')

# ============================================================
# Fig 5: combined_r2_kl.pdf — 4 subplots, dual y-axes
# ============================================================
fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

for i, layer in enumerate(layers):
    ax_left = axes[i]

    # Left y-axis: R² line plot
    iter_subset = iter_df[iter_df['layer'] == layer].sort_values('iteration')
    color_line = layer_colors[i]
    ax_left.plot(iter_subset['iteration'], iter_subset['test_r2'],
                 marker='o', markersize=4, color=color_line, zorder=3)
    ax_left.set_ylim(0.98, 1.0)
    ax_left.set_xlabel('Iteration')
    if i == 0:
        ax_left.set_ylabel('Test R²')
    ax_left.tick_params(axis='y', labelcolor=color_line)
    ax_left.grid(alpha=0.3)
    ax_left.set_title(f'Layer {layer}')

    # Right y-axis: KL bar chart
    ax_right = ax_left.twinx()
    sub_kl = subspace_df[subspace_df['layer'] == layer].sort_values('iteration')
    ax_right.bar(sub_kl['iteration'].values, sub_kl['mean_kl'].values,
                 color='firebrick', alpha=0.4, zorder=1)
    ax_right.set_yscale('log')
    ax_right.set_ylim(1e-6, 0.1)
    if i == n_layers - 1:
        ax_right.set_ylabel('KL (subspace ablation)')
    ax_right.tick_params(axis='y', labelcolor='firebrick')

fig.suptitle('Test R² and Subspace Ablation KL per Layer', y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'combined_r2_kl.pdf'), bbox_inches='tight')
plt.close(fig)
print('Saved combined_r2_kl.pdf')

print('\nAll 5 figures saved to:', FIG_DIR)
