"""
Summary plots for unembed decomposition analysis.
Run interactively cell-by-cell in an IDE (e.g., VS Code with #%% markers).
"""

# %% Imports and data loading
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ensure we're in the project root regardless of where the IDE starts
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

results_dir = "models/mess3/20260306_150920_L4_d64_H1_full_LN/unembed_analysis"

overlap_df = pd.read_csv(f"{results_dir}/wu_overlap.csv")
ablation_df = pd.read_csv(f"{results_dir}/wu_ablation.csv")
logit_df = pd.read_csv(f"{results_dir}/logit_contributions.csv")
emission_data = np.load(f"{results_dir}/effective_emission.npz")
probe_df = pd.read_csv(f"{results_dir}/postln_iterative_metrics.csv")

n_subspaces = len(overlap_df)
print(f"Number of subspaces: {n_subspaces}")
print(f"Parseval check (sum of overlaps): {overlap_df['overlap_fraction'].sum():.6f}")

# %% Plot 0: Post-LN iterative probing — R² and MSE vs iteration
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.plot(probe_df["iteration"], probe_df["test_r2"], "o-", color="steelblue", markersize=4)
ax.set_xlabel("Iteration (probe-project cycle)")
ax.set_ylabel("Test R²")
ax.set_title("Post-LN iterative probing: R² after successive projections")
ax.set_ylim(0, 1.05)
ax.axhline(0.05, color="gray", linestyle="--", alpha=0.5, label="R² threshold")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(probe_df["cumulative_dims"], probe_df["test_mse"], "o-", color="steelblue", markersize=4)
ax.set_xlabel("Cumulative dimensions projected out")
ax.set_ylabel("Test MSE")
ax.set_title("Post-LN probe MSE vs. dimensions removed")
ax.grid(True, alpha=0.3)

fig.tight_layout()
plt.show()

# %% Plot 1: W_U overlap bar chart
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: overall overlap fraction
ax = axes[0]
n_show = min(n_subspaces, 20)
ax.bar(range(n_show), overlap_df["overlap_fraction"].values[:n_show],
       color="steelblue", alpha=0.8)
ax.set_xlabel("Subspace index (probe iteration)")
ax.set_ylabel(r"$\|V_k W_U\|_F^2 / \|W_U\|_F^2$")
ax.set_title(r"$W_U$ overlap with each probe subspace")
ax.axhline(1.0 / n_subspaces, color="gray", linestyle="--", alpha=0.5,
           label=f"Uniform (1/{n_subspaces})")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

# Right: per-token breakdown for top subspaces
ax = axes[1]
token_cols = [c for c in overlap_df.columns if c.startswith("overlap_token")]
d_vocab = len(token_cols)
n_show_sub = min(8, n_subspaces)
x = np.arange(n_show_sub)
width = 0.8 / d_vocab
colors = plt.cm.Set2(np.linspace(0, 1, d_vocab))

for j, col in enumerate(token_cols):
    ax.bar(x + j * width, overlap_df[col].values[:n_show_sub], width,
           color=colors[j], label=f"Token {j}", alpha=0.85)

ax.set_xlabel("Subspace index")
ax.set_ylabel(r"$\|V_k w_j\|^2 / \|w_j\|^2$")
ax.set_title("Per-token overlap breakdown")
ax.set_xticks(x + width * (d_vocab - 1) / 2)
ax.set_xticklabels(range(n_show_sub))
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

fig.tight_layout()
plt.show()

# %% Plot 2: Effective emission matrix — side-by-side heatmaps
M_eff = emission_data["subspace0"]
log_O_T = emission_data["log_O_T"]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left: recovered M_eff
ax = axes[0]
im = ax.imshow(M_eff, cmap="RdBu_r", aspect="auto")
ax.set_xlabel("Token j")
ax.set_ylabel("State i")
ax.set_title(r"Recovered $M_{\mathrm{eff}}$ (subspace 0)")
for i in range(M_eff.shape[0]):
    for j in range(M_eff.shape[1]):
        ax.text(j, i, f"{M_eff[i, j]:.2f}", ha="center", va="center", fontsize=9)
fig.colorbar(im, ax=ax, shrink=0.8)

# Right: theoretical log(O^T)
ax = axes[1]
im = ax.imshow(log_O_T, cmap="RdBu_r", aspect="auto")
ax.set_xlabel("Token j")
ax.set_ylabel("State i")
ax.set_title(r"Theoretical $\log(O^T)$")
for i in range(log_O_T.shape[0]):
    for j in range(log_O_T.shape[1]):
        ax.text(j, i, f"{log_O_T[i, j]:.2f}", ha="center", va="center", fontsize=9)
fig.colorbar(im, ax=ax, shrink=0.8)

fig.suptitle(r"Effective emission matrix vs. Bayes-optimal", y=1.02)
fig.tight_layout()
plt.show()

# %% Plot 3: W_U ablation — per-subspace KL + cumulative
per_sub = ablation_df[ablation_df["ablation_type"] == "per_subspace"].copy()
cumul = ablation_df[ablation_df["ablation_type"] == "cumulative"].copy()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: per-subspace KL (bar chart)
ax = axes[0]
n_show = min(20, len(per_sub))
ax.bar(per_sub["subspace"].values[:n_show], per_sub["mean_kl"].values[:n_show],
       color="firebrick", alpha=0.8,
       yerr=per_sub["std_kl"].values[:n_show], capsize=2, error_kw={"linewidth": 0.8})
ax.set_xlabel("Subspace index")
ax.set_ylabel("KL(clean || ablated)")
ax.set_title(r"Per-subspace $W_U$ ablation")
ax.set_yscale("log")
ax.grid(True, alpha=0.3, axis="y")

# Right: cumulative KL (line chart)
ax = axes[1]
# Compute cumulative dims
cum_dims = np.arange(1, len(cumul) + 1) * 2  # each subspace is rank 2
# Actually, read from probe metrics if available
try:
    probe_df = pd.read_csv(f"{results_dir}/postln_iterative_metrics.csv")
    cum_dims = probe_df["cumulative_dims"].values[:len(cumul)]
except Exception:
    cum_dims = np.arange(1, len(cumul) + 1) * 2

ax.plot(cum_dims, cumul["mean_kl"].values, "o-",
        color="firebrick", markersize=4)
ax.fill_between(cum_dims,
                cumul["mean_kl"].values - cumul["std_kl"].values,
                cumul["mean_kl"].values + cumul["std_kl"].values,
                color="firebrick", alpha=0.15)
ax.set_xlabel("Cumulative dimensions removed from $W_U$")
ax.set_ylabel("KL(clean || ablated)")
ax.set_title(r"Cumulative $W_U$ ablation")
ax.grid(True, alpha=0.3)

fig.tight_layout()
plt.show()

# %% Plot 4: Logit contribution variance + Bayes correlation
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

n_show = min(20, len(logit_df))

# Left: variance
ax = axes[0]
ax.bar(logit_df["subspace"].values[:n_show], logit_df["mean_variance"].values[:n_show],
       color="teal", alpha=0.8)
ax.set_xlabel("Subspace index")
ax.set_ylabel("Mean logit variance")
ax.set_title("Logit contribution variance per subspace")
ax.grid(True, alpha=0.3, axis="y")

# Right: Bayes-optimal correlation
ax = axes[1]
ax.bar(logit_df["subspace"].values[:n_show], logit_df["bayes_correlation"].values[:n_show],
       color="darkorange", alpha=0.8)
ax.set_xlabel("Subspace index")
ax.set_ylabel("Correlation with Bayes-optimal logits")
ax.set_title("Logit contribution: correlation with Bayes-optimal")
ax.set_ylim(-1, 1)
ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
ax.grid(True, alpha=0.3, axis="y")

fig.tight_layout()
plt.show()
