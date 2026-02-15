"""
Scatter plot comparing final validation loss vs number of parameters
for all models in models/cylinderhmm/.
"""
#%%
#%% Empirical entropy rate from data (no HMM seed needed)
from src.hmm import CylinderGraphHMM
proc = CylinderGraphHMM(n=6, depth=3, tokens_per_cluster=16, dirichlet_alpha=0.3, p=0.1, seed=42)
entropy_rate = proc.entropy_rate_empirical_estimate(10000, burn_in=1000)
print(f"Empirical entropy rate: {entropy_rate:.6f}")
SAVE_DIR = "notes/figures"


#%% Computing the optimal prediction loss given the fixed context window
import torch as t
import numpy as np
from jaxtyping import Int
from src.hmm import HMM, CylinderGraphHMM
from scipy.special import logsumexp
from tqdm import tqdm
proc = CylinderGraphHMM(n=6, depth=3, tokens_per_cluster=16, dirichlet_alpha=0.3, p=0.1, seed=42)
example_shard = t.load("data/datasets/cylinder_graph_hmm/train/observations.pt")
print(example_shard.shape)
trunc = 2000
example_shard = example_shard[:trunc]

def true_next_token_prediction(proc: HMM, sequences: Int[t.Tensor, "batch seq_len"]):
    """Compute Bayes-optimal log P(x_t = v | x_0, ..., x_{t-1}) at every position.

    Returns:
        np.ndarray of shape (batch, seq_len, vocab) where entry (b, t, v)
        is the log predictive probability of token v at position t given all
        preceding tokens, under the true HMM.  At t=0 the conditioning
        set is empty so the marginal from the stationary distribution is used.
    """
    sequences_np = sequences.numpy() if isinstance(sequences, t.Tensor) else sequences
    batch_size, seq_len = sequences_np.shape
    E = proc.emission_matrices          # (vocab, n_states, n_states)
    log_E = np.log(E)
    init_state = proc.get_stationary_distribution()
    log_init_state = np.log(init_state)

    all_log_pred_dists = []
    for batch_idx in tqdm(range(batch_size), desc="Computing true next token prediction"):
        seq = sequences_np[batch_idx]
        log_pred_dists = []

        # --- Position 0: predict x_0 from stationary prior only ---
        # P(x_0 = v) = sum_{s,s'} pi(s) E[v, s, s']
        log_joint_0 = log_init_state[np.newaxis, :, np.newaxis] + log_E
        log_pred_0 = logsumexp(log_joint_0, axis=(1, 2))       # (vocab,)
        log_pred_dists.append(log_pred_0)

        # Update belief after observing x_0
        log_alpha = logsumexp(log_init_state[:, np.newaxis] + log_E[seq[0]], axis=0)
        log_alpha -= logsumexp(log_alpha)                       # normalize

        # --- Positions 1 .. seq_len-1 ---
        for ti in range(1, seq_len):
            # Predictive distribution: P(x_t = v | x_{<t})
            log_joint = log_alpha[np.newaxis, :, np.newaxis] + log_E
            log_pred = logsumexp(log_joint, axis=(1, 2))        # (vocab,)
            log_pred_dists.append(log_pred)

            # Update belief after observing x_t
            log_alpha = logsumexp(log_alpha[:, np.newaxis] + log_E[seq[ti]], axis=0)
            log_alpha -= logsumexp(log_alpha)                   # normalize

        all_log_pred_dists.append(np.stack(log_pred_dists))

    return np.array(all_log_pred_dists)[:, 1:]  # (batch, seq_len, vocab)

def compute_optimal_predictor_loss(proc: HMM, sequences: Int[t.Tensor, "batch seq_len"]):
    log_pred_dists = t.from_numpy(true_next_token_prediction(proc, sequences))
    target_tokens = sequences[:, 1:]
    loss = -t.log_softmax(log_pred_dists, dim=-1).gather(dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1).mean()
    return loss

loss = compute_optimal_predictor_loss(proc, example_shard)
#%%
print(loss.item())

#%% 


#%%
import json
import os
import re
import matplotlib.pyplot as plt
import numpy as np

MODEL_DIR = "models/cylinderhmm"

records = []
for run_name in sorted(os.listdir(MODEL_DIR)):
    meta_path = os.path.join(MODEL_DIR, run_name, "metadata.json")
    if not os.path.isfile(meta_path):
        continue
    with open(meta_path) as f:
        meta = json.load(f)

    # Parse model config from run_name:
    #   "20260206_135142_L1_d16_full_LN"          (old, no H field)
    #   "20260207_134849_L2_d16_H1_attn_only_LN"  (new, with H field)
    match = re.search(r"L(\d+)_d(\d+)(?:_H(\d+))?_(attn_only|full)_(LN|noLN)$", run_name)
    if not match:
        continue

    n_layer = int(match.group(1))
    d_model = int(match.group(2))
    n_head  = int(match.group(3)) if match.group(3) else 1
    arch = match.group(4)        # "attn_only" or "full"
    norm = match.group(5)        # "LN" or "noLN"
    try:
        records.append({
            "run_name": meta.get("run_name", run_name),
            "num_params": meta["num_params"],
            "best_val_loss": meta["best_val_loss"],
            "n_layer": n_layer,
            "d_model": d_model,
            "n_head": n_head,
            "arch": arch,
            "norm": norm,
        })
    except Exception as e:
        print(f"Error parsing run_name: {run_name}")
        print(f"Error: {e}")
        continue

print(f"Found {len(records)} models")

# --- Scatter plot ---
fig, ax = plt.subplots(figsize=(10, 8))

# Style mapping
marker_map = {"attn_only": "o", "full": "s"}
color_map = {"LN": "tab:blue", "noLN": "tab:red"}

for r in records:
    ax.scatter(
        r["num_params"],
        r["best_val_loss"],
        marker=marker_map[r["arch"]],
        color=color_map[r["norm"]],
        s=80,
        edgecolors="k",
        linewidths=0.5,
        zorder=3,
    )
    ax.annotate(
        f"L{r['n_layer']}d{r['d_model']}H{r['n_head']}",
        (r["num_params"], r["best_val_loss"]),
        fontsize=7,
        textcoords="offset points",
        xytext=(5, 3),
    )

ax.set_xscale("log")
ax.set_xlabel("Number of Parameters", fontsize=13)
ax.set_ylabel("Best Validation Loss", fontsize=13)
ax.set_title("Model Size vs Validation Loss (Cylinder Graph HMM)", fontsize=14)

# Build legend manually, including axhline legend
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=8, label="attn_only"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", markersize=8, label="full"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue", markersize=8, label="LN"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:red", markersize=8, label="noLN"),
]
GT_ENTROPY_RATE = 2.439599
GT_OPTIMAL_LOSS = loss.item()
hline = ax.axhline(GT_ENTROPY_RATE, color="k", linestyle="--", label="Theoretical Entropy Rate")
legend_handles.append(Line2D([0], [0], color="k", linestyle="--", label="Theoretical Entropy Rate"))
hline2 = ax.axhline(GT_OPTIMAL_LOSS, color="r", linestyle="--", label="Optimal Predictor Loss")
legend_handles.append(Line2D([0], [0], color="r", linestyle="--", label="Optimal Predictor Loss"))

# N-gram baseline horizontal lines with text annotations
ngram_results_path = "out/ngram_baseline_results.json"
if os.path.isfile(ngram_results_path):
    with open(ngram_results_path) as f:
        ngram_data = json.load(f)
    ngram_cmap = plt.cm.viridis
    ngram_orders = ngram_data["results"]
    for i, entry in enumerate(ngram_orders):
        n = entry["n"]
        test_ce = entry["test_ce"]
        color = ngram_cmap(i / max(len(ngram_orders) - 1, 1))
        ax.axhline(test_ce, color=color, linestyle=":", linewidth=1.2, alpha=0.8)
        # Text label at right edge of plot
        ax.text(
            1.01, test_ce, f"{n}-gram",
            transform=ax.get_yaxis_transform(),
            color=color, fontsize=8, fontweight="bold",
            va="center", ha="left",
        )

ax.legend(handles=legend_handles, fontsize=11)
ax.grid(True, alpha=0.3)

ax.set_ylim(GT_ENTROPY_RATE, 2.9)

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
print("Saved model_comparison.png")
plt.show()

#%% --- Log-log scatter: excess loss over Bayes-optimal ---
fig2, ax2 = plt.subplots(figsize=(10, 7))

excess = [r["best_val_loss"] - GT_OPTIMAL_LOSS for r in records]
params = [r["num_params"] for r in records]

ax2.scatter(params, excess, s=60, color="tab:blue", edgecolors="k", linewidths=0.4, zorder=3)

ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("Number of Parameters", fontsize=13)
ax2.set_ylabel(r"$\hat{L} - L^{*}(k{=}16)$  (nats/token)", fontsize=13)
ax2.set_title("Excess Loss over Bayes-Optimal Predictor", fontsize=14)
ax2.grid(True, alpha=0.3, which="both")

plt.tight_layout()
plt.savefig("excess_loss_loglog.png", dpi=150)
print("Saved excess_loss_loglog.png")
plt.show()

#%% Embedding cluster similarity analysis
import pandas as pd
import torch.nn.functional as F

cluster_records = []
for run_dir in sorted(os.listdir(MODEL_DIR)):
    ckpt_path = os.path.join(MODEL_DIR, run_dir, "best_model.pt")
    meta_path = os.path.join(MODEL_DIR, run_dir, "metadata.json")
    if not os.path.isfile(ckpt_path) or not os.path.isfile(meta_path):
        continue
    match = re.search(r"L(\d+)_d(\d+)(?:_H(\d+))?_(attn_only|full)_(LN|noLN)$", run_dir)
    if not match:
        continue
    with open(meta_path) as f:
        meta = json.load(f)

    n_layer = int(match.group(1))
    d_model = int(match.group(2))
    n_head  = int(match.group(3)) if match.group(3) else 1
    arch = match.group(4)
    norm = match.group(5)

    ckpt = t.load(ckpt_path, map_location="cpu")
    W_E = ckpt["model_state_dict"]["embed.W_E"]  # (48, d_model)

    # Cluster centroids
    c0 = W_E[0:16].mean(dim=0)
    c1 = W_E[16:32].mean(dim=0)
    c2 = W_E[32:48].mean(dim=0)

    sim_01 = F.cosine_similarity(c0.unsqueeze(0), c1.unsqueeze(0)).item()
    sim_02 = F.cosine_similarity(c0.unsqueeze(0), c2.unsqueeze(0)).item()
    sim_12 = F.cosine_similarity(c1.unsqueeze(0), c2.unsqueeze(0)).item()

    cluster_records.append({
        "run_name": run_dir,
        "num_params": meta["num_params"],
        "best_val_loss": meta["best_val_loss"],
        "n_layer": n_layer,
        "d_model": d_model,
        "n_head": n_head,
        "arch": arch,
        "norm": norm,
        "sim_01": sim_01,
        "sim_02": sim_02,
        "sim_12": sim_12,
    })

df_cluster = pd.DataFrame(cluster_records)
print(f"Cluster similarity dataframe: {len(df_cluster)} rows")
print(df_cluster[["run_name", "sim_01", "sim_02", "sim_12"]].head(10))

#%% Scatter plot: mean cluster cosine similarity vs validation loss
df_cluster["sim_mean"] = df_cluster[["sim_01", "sim_02", "sim_12"]].mean(axis=1)
df_noLN = df_cluster[df_cluster["norm"] == "noLN"]

fig3, ax3 = plt.subplots(figsize=(10, 7))

ax3.scatter(df_noLN["best_val_loss"], df_noLN["sim_mean"],
            s=50, color="tab:blue", edgecolors="k", linewidths=0.4,
            alpha=0.7, zorder=3)

ax3.set_xlabel("Best Validation Loss (nats/token)", fontsize=13)
ax3.set_ylabel("Mean Cosine Similarity (cluster centroids)", fontsize=13)
ax3.set_title("Embedding Cluster Similarity vs Validation Loss", fontsize=14)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "embedding_cluster_similarity.png"), dpi=150)
print(f"Saved {SAVE_DIR}/embedding_cluster_similarity.png")
plt.show()

#%% Compare the model's performance for fixed embedding modes

import json
from pathlib import Path

fixed_emb_dir = Path("models/cylinderhmm_fixed_emb_new")
records = []
for meta_path in sorted(fixed_emb_dir.glob("*/metadata.json")):
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        run_name = meta["run_name"]
        # Parse embedding mode and architecture from run_name
        # Format: fixed_{mode}_L{layers}_d{dim}_H{heads}_full_noLN
        parts = run_name.split("_")
        # embedding mode is between "fixed" and "L{n}"
        l_idx = next(i for i, p in enumerate(parts) if p.startswith("L"))
        emb_mode = "_".join(parts[1:l_idx])
        arch_str = "_".join(parts[l_idx:])  # e.g. L4_d8_H1_full_noLN
        # Extract d and H
        d_model = int([p for p in parts if p.startswith("d") and p[1:].isdigit()][0][1:])
        n_heads = int([p for p in parts if p.startswith("H") and p[1:].isdigit()][0][1:])
        records.append({
            "emb_mode": emb_mode,
            "d_model": d_model,
            "n_heads": n_heads,
            "arch": f"d{d_model}_H{n_heads}",
            "best_val_loss": meta["best_val_loss"],
            "final_train_loss": meta["final_train_loss"],
            "trainable_params": meta["trainable_params"],
            "total_params": meta["total_params"],
        })
    except Exception as e:
        print(f"Error: {e}")
        continue

# ── Grouped bar chart: embedding mode × architecture ──
SHOW_NGRAM = True  # Toggle to overlay n-gram baseline on the plot

emb_modes = ["trained", "pmi", "random", "onehot"]
archs = sorted(set(r["arch"] for r in records if r["d_model"] == 48))
emb_labels = ["Trained", "PMI", "Random", "One-Hot"]

fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(len(emb_modes))
n_archs = len(archs)
width = 0.8 / max(n_archs, 1)
offsets = [(i - (n_archs - 1) / 2) * width for i in range(n_archs)]

for i, arch in enumerate(archs):
    vals = []
    for mode in emb_modes:
        match = [r for r in records if r["emb_mode"] == mode and r["arch"] == arch]
        vals.append(match[0]["best_val_loss"] if match else np.nan)
    bars = ax.bar(x + offsets[i], vals, width, label=arch)
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

ax.axhline(y=loss.item(), color="black", linestyle="--", linewidth=1, label=f"Best predictor loss ({loss.item():.3f})")

if SHOW_NGRAM:
    ngram_path = Path("out/ngram_baseline_results.json")
    if ngram_path.exists():
        with open(ngram_path) as f:
            ngram_data = json.load(f)
        # Show best n-gram (lowest test CE) as a horizontal line
        best_ngram = min(ngram_data["results"], key=lambda r: r["test_ce"])
        ax.axhline(y=best_ngram["test_ce"], color="gray", linestyle=":",
                   linewidth=1.5, label=f"Best {best_ngram['n']}-gram ({best_ngram['test_ce']:.3f})")
        # Also show bigram as reference (strong single-token context baseline)
        bigram = next(r for r in ngram_data["results"] if r["n"] == 2)
        ax.axhline(y=bigram["test_ce"], color="gray", linestyle="-.",
                   linewidth=1, alpha=0.6, label=f"Bigram ({bigram['test_ce']:.3f})")
    else:
        print("Warning: out/ngram_baseline_results.json not found. Run src/ngram_baseline.py first.")

ax.set_ylabel("Best Validation Loss (nats)")
ax.set_title("Fixed Embedding: Validation Loss by Embedding Mode and Architecture")
ax.set_xticks(x)
ax.set_xticklabels(emb_labels)
ax.legend()
ax.set_ylim(bottom=2.45, top=2.60)

plt.tight_layout()
FIXED_EMB_DIR = "notes/fixed_embedding"
plt.savefig(os.path.join(FIXED_EMB_DIR, "fixed_embedding_comparison.png"), dpi=150)
print(f"Saved {FIXED_EMB_DIR}/fixed_embedding_comparison.png")
plt.show()