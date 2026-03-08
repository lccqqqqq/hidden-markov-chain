import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import os

# Paths
base = '/mnt/users/clin/workspace/hidden-markov-chain'
results_dir = os.path.join(base, 'models/mess3/20260306_150920_L4_d64_H1_full_LN/iterative_probe_results')
out_dir = os.path.join(base, 'figures/iterative_probe')
os.makedirs(out_dir, exist_ok=True)

iter_df = pd.read_csv(os.path.join(results_dir, 'iterative_metrics.csv'))
cf_df = pd.read_csv(os.path.join(results_dir, 'counterfactual_metrics.csv'))

layers = sorted(iter_df['layer'].unique())
n_layers = len(layers)
layer_colors = cm.viridis(np.linspace(0.15, 0.85, n_layers))

# --- Fig 1: R² vs iteration ---
fig, ax = plt.subplots(figsize=(6, 4))
for i, layer in enumerate(layers):
    d = iter_df[iter_df['layer'] == layer]
    ax.plot(d['iteration'], d['test_r2'], 'o-', color=layer_colors[i], label=f'Layer {layer}', markersize=3)
ax.set_xlabel('Iteration')
ax.set_ylabel('Test R²')
ax.set_ylim(0.0, 1.05)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(out_dir, 'r2_vs_iteration.pdf'))
plt.close(fig)
print('Saved r2_vs_iteration.pdf')

# --- Fig 2: MSE vs cumulative dims ---
fig, ax = plt.subplots(figsize=(6, 4))
for i, layer in enumerate(layers):
    d = iter_df[iter_df['layer'] == layer]
    ax.plot(d['cumulative_dims'], d['test_mse'], 'o-', color=layer_colors[i], label=f'Layer {layer}', markersize=3)
ax.set_xlabel('Cumulative dimensions projected out')
ax.set_ylabel('Test MSE')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(out_dir, 'mse_vs_dims.pdf'))
plt.close(fig)
print('Saved mse_vs_dims.pdf')

# --- Fig 3: KL per iteration (grouped bar, subspace only) ---
sub_df = cf_df[cf_df['ablation_type'] == 'subspace'].copy()
iterations = sorted(sub_df['iteration'].unique())
n_iters = len(iterations)
bar_width = 0.18
fig, ax = plt.subplots(figsize=(10, 4))
x = np.arange(n_iters)
for i, layer in enumerate(layers):
    d = sub_df[sub_df['layer'] == layer].sort_values('iteration')
    offset = (i - (n_layers - 1) / 2) * bar_width
    ax.bar(x + offset, d['mean_kl'].values, bar_width,
           yerr=d['std_kl'].values, capsize=2,
           color=layer_colors[i], label=f'Layer {layer}')
ax.set_yscale('log')
ax.set_xlabel('Iteration')
ax.set_ylabel('Mean KL (subspace ablation)')
ax.set_xticks(x)
ax.set_xticklabels([str(it) for it in iterations], rotation=45 if n_iters > 16 else 0, ha='right' if n_iters > 16 else 'center')
ax.legend()
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
fig.savefig(os.path.join(out_dir, 'kl_per_iteration.pdf'))
plt.close(fig)
print('Saved kl_per_iteration.pdf')

# --- Fig 4: Cumulative KL per layer (bar chart, backwards compat) ---
cum_prog_df = cf_df[cf_df['ablation_type'] == 'cumulative_progressive'].copy()
if cum_prog_df.empty:
    # Fall back to old "cumulative" type if progressive data not yet available
    cum_prog_df = cf_df[cf_df['ablation_type'] == 'cumulative'].copy()

# Bar chart: final cumulative value per layer
fig, ax = plt.subplots(figsize=(5, 4))
for i, layer in enumerate(layers):
    d = cum_prog_df[cum_prog_df['layer'] == layer]
    max_kl = d['mean_kl'].max()
    max_std = d.loc[d['mean_kl'].idxmax(), 'std_kl']
    ax.bar(i, max_kl, yerr=max_std, capsize=4, color=layer_colors[i])
ax.set_xticks(range(n_layers))
ax.set_xticklabels([f'Layer {l}' for l in layers])
ax.set_ylabel('Cumulative KL (all dims)')
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
fig.savefig(os.path.join(out_dir, 'cumulative_kl.pdf'))
plt.close(fig)
print('Saved cumulative_kl.pdf')

# --- Fig 6: Cumulative-progressive KL vs dims ablated ---
if not cf_df[cf_df['ablation_type'] == 'cumulative_progressive'].empty:
    cprog = cf_df[cf_df['ablation_type'] == 'cumulative_progressive'].copy()
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, layer in enumerate(layers):
        d = cprog[cprog['layer'] == layer].sort_values('iteration')
        # Compute cumulative dims from iteration index (each iter removes rank-2)
        cum_dims = (d['iteration'].values + 1) * 2
        ax.plot(cum_dims, d['mean_kl'].values, 'o-',
                color=layer_colors[i], label=f'Layer {layer}', markersize=3)
        ax.fill_between(cum_dims,
                        d['mean_kl'].values - d['std_kl'].values,
                        d['mean_kl'].values + d['std_kl'].values,
                        color=layer_colors[i], alpha=0.15)
    ax.set_xlabel('Cumulative dimensions ablated')
    ax.set_ylabel('KL(clean || ablated)')
    ax.set_title('Cumulative-progressive ablation')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'cumulative_progressive_kl.pdf'))
    plt.close(fig)
    print('Saved cumulative_progressive_kl.pdf')

# --- Fig 5: Combined R² + KL, 4 subplots ---
fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
for i, layer in enumerate(layers):
    ax1 = axes[i]
    d_iter = iter_df[iter_df['layer'] == layer].sort_values('iteration')
    d_sub = sub_df[sub_df['layer'] == layer].sort_values('iteration')
    iters = d_iter['iteration'].values

    # Left y-axis: R²
    ax1.plot(iters, d_iter['test_r2'].values, color=layer_colors[i], marker='o', markersize=2)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Test R²')

    # Right y-axis: KL bars
    ax2 = ax1.twinx()
    ax2.bar(d_sub['iteration'].values, d_sub['mean_kl'].values,
            color='firebrick', alpha=0.4, width=0.8)
    ax2.set_yscale('log')
    ax2.set_ylim(1e-6, 0.1)
    ax2.set_ylabel('KL (subspace)')

    ax1.set_title(f'Layer {layer}')

fig.suptitle('Iterative Probe: R² and Subspace KL per Iteration', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(out_dir, 'combined_r2_kl.pdf'), bbox_inches='tight')
plt.close(fig)
print('Saved combined_r2_kl.pdf')

# Verify
for f in ['r2_vs_iteration.pdf', 'mse_vs_dims.pdf', 'kl_per_iteration.pdf', 'cumulative_kl.pdf', 'combined_r2_kl.pdf', 'cumulative_progressive_kl.pdf']:
    path = os.path.join(out_dir, f)
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f'  {f}: {size} bytes')
    else:
        print(f'  {f}: MISSING!')
