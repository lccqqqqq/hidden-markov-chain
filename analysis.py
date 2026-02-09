#%% Take the best checkpoint and load the embedding matrix
import torch as t
import sys
sys.path.append('src')
from src.utils import initialize_transformer_from_yaml

CONFIG_PATH = "base_config.yaml"
device = "cuda" if t.cuda.is_available() else "cpu"

# Initialize model architecture
model = initialize_transformer_from_yaml(CONFIG_PATH)

# Load checkpoint
checkpoint = t.load("checkpoints/best_model.pt", map_location=device)

# Load the model weights
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()
print(f"Model loaded successfully from checkpoint (step {checkpoint['global_step']}, val_loss: {checkpoint['val_loss']:.6f})")

#%% Get the embedding matrix
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Access embedding matrix from HookedTransformer
embedding_matrix = model.embed.W_E.detach().cpu().numpy()  # Shape: (vocab_size, d_model)
print(f"Embedding matrix shape: {embedding_matrix.shape}")
print(f"Vocab size: {embedding_matrix.shape[0]}, Embedding dim: {embedding_matrix.shape[1]}")

#%% Calculate norms of embedding vectors
embedding_norms = np.linalg.norm(embedding_matrix, axis=1)
print(f"\nEmbedding norms - Min: {embedding_norms.min():.4f}, Max: {embedding_norms.max():.4f}, Mean: {embedding_norms.mean():.4f}")

#%% Calculate cosine similarity matrix
# Normalize each embedding vector to unit length
embedding_normalized = embedding_matrix / np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
# Cosine similarity is the dot product of normalized vectors
cosine_similarity_matrix = embedding_normalized @ embedding_normalized.T
print(f"Cosine similarity matrix shape: {cosine_similarity_matrix.shape}")

#%% Create interactive visualization with plotly
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Cosine Similarity Matrix', 'Embedding Vector Norms'),
    specs=[[{"type": "heatmap"}, {"type": "bar"}]],
    horizontal_spacing=0.15
)

# Plot 1: Cosine similarity matrix
fig.add_trace(
    go.Heatmap(
        z=cosine_similarity_matrix,
        colorscale='RdBu_r',
        zmin=-1, zmax=1,
        colorbar=dict(x=0.45, len=0.8, title="Cosine Sim"),
        hovertemplate='Token %{y} vs %{x}<br>Similarity: %{z:.3f}<extra></extra>'
    ),
    row=1, col=1
)

# Plot 2: Embedding norms
fig.add_trace(
    go.Bar(
        x=list(range(len(embedding_norms))),
        y=embedding_norms,
        marker_color='steelblue',
        hovertemplate='Token: %{x}<br>Norm: %{y:.3f}<extra></extra>'
    ),
    row=1, col=2
)

# Update axes labels
fig.update_xaxes(title_text="Token ID", row=1, col=1)
fig.update_yaxes(title_text="Token ID", row=1, col=1)
fig.update_xaxes(title_text="Token ID", row=1, col=2)
fig.update_yaxes(title_text="L2 Norm", row=1, col=2)

fig.update_layout(height=600, width=1200, title_text="Embedding Matrix Analysis", showlegend=False)
fig.show()

#%% Analyze structure by depth (if using cylinder graph HMM)
# Based on config: depth=3, tokens_per_cluster=16, so vocab_size=48
depth = 3
tokens_per_cluster = 16

print(f"\n=== Analysis by Depth Level ===")
for d in range(depth):
    start_idx = d * tokens_per_cluster
    end_idx = (d + 1) * tokens_per_cluster
    depth_embeddings = embedding_matrix[start_idx:end_idx]
    depth_norms = embedding_norms[start_idx:end_idx]

    print(f"\nDepth {d} (tokens {start_idx}-{end_idx-1}):")
    print(f"  Mean norm: {depth_norms.mean():.4f} ± {depth_norms.std():.4f}")
    print(f"  Norm range: [{depth_norms.min():.4f}, {depth_norms.max():.4f}]")

    # Within-depth cosine similarity
    depth_normalized = depth_embeddings / np.linalg.norm(depth_embeddings, axis=1, keepdims=True)
    within_depth_sim = depth_normalized @ depth_normalized.T
    # Exclude diagonal when computing mean
    mask = ~np.eye(within_depth_sim.shape[0], dtype=bool)
    mean_within_sim = within_depth_sim[mask].mean()
    print(f"  Mean within-depth cosine similarity: {mean_within_sim:.4f}")


#%% Vector norm and word frequencies
# No obvious relationship between the embedding norm and the token frequency
from collections import Counter
import matplotlib.pyplot as plt

# Load data & compute frequencies
data = np.load("data/datasets/cylinder_graph_hmm/shards/obs_00001.npy")
token_counts = Counter(data)
total_tokens = len(data)
token_freqs = {token.item(): count / total_tokens for token, count in token_counts.items()}

# Build arrays for plot
xs = [token_freqs.get(i, 0.0) for i in range(len(embedding_norms))]
ys = embedding_norms

# Beautify the scatter plot
plt.figure(figsize=(8, 6))
sc = plt.scatter(xs, ys, s=50, alpha=0.85, edgecolors='k', linewidths=0.3)

plt.title("Embedding Norm vs Token Frequency", fontsize=16, pad=12)
plt.xlabel("Token Frequency", fontsize=13)
plt.ylabel("Embedding L2 Norm", fontsize=13)
plt.grid(True, linestyle='--', alpha=0.35)
plt.tight_layout()
plt.show()

#%% Compute empirical co-occurrence matrix with sliding window: WARNING: Costly...
print("\n=== Computing Co-occurrence Matrix ===")

# Parameters
window_size = 16
vocab_size = embedding_matrix.shape[0]

# Initialize co-occurrence matrix
cooccurrence_matrix = np.zeros((vocab_size, vocab_size), dtype=np.float64)

# Load all observation data (or use the data already loaded)
# Use multiple shards if available for better statistics
print(f"Loading observation data...")
all_data = []
for shard_idx in range(10):  # Try loading up to 10 shards
    try:
        shard_data = np.load(f"data/datasets/cylinder_graph_hmm/shards/obs_{shard_idx:05d}.npy")
        all_data.append(shard_data)
        print(f"  Loaded shard {shard_idx}: {len(shard_data)} tokens")
    except FileNotFoundError:
        break

if not all_data:
    # Fallback to single shard if multiple shards not found
    all_data = [data]

observations = np.concatenate(all_data)
print(f"Total tokens: {len(observations):,}")

# Sliding window to count co-occurrences
print(f"Computing co-occurrences with window size {window_size}...")
num_windows = len(observations) - window_size + 1

for i in range(0, num_windows, 1000):  # Process in batches for progress reporting
    end_idx = min(i + 1000, num_windows)

    for window_start in range(i, end_idx):
        window = observations[window_start:window_start + window_size]

        # Count all pairs in this window (including self-pairs)
        unique_tokens = np.unique(window)
        for token_i in unique_tokens:
            for token_j in unique_tokens:
                cooccurrence_matrix[token_i, token_j] += 1

    if (i + 1000) % 100000 == 0:
        print(f"  Processed {i + 1000:,} / {num_windows:,} windows")

# Normalize to get probabilities P(i, j)
# Total number of token pairs observed
total_pairs = cooccurrence_matrix.sum()
cooccurrence_prob = cooccurrence_matrix / total_pairs

print(f"Total co-occurrence pairs counted: {total_pairs:,.0f}")
print(f"Co-occurrence matrix shape: {cooccurrence_prob.shape}")
print(f"Non-zero entries: {np.count_nonzero(cooccurrence_prob)} / {vocab_size * vocab_size}")

# Save co-occurrence matrices to disk
import os
os.makedirs("out", exist_ok=True)
np.save("out/cooccurrence_counts.npy", cooccurrence_matrix)
np.save("out/cooccurrence_prob.npy", cooccurrence_prob)
print(f"\nSaved co-occurrence matrices:")
print(f"  - out/cooccurrence_counts.npy (raw counts)")
print(f"  - out/cooccurrence_prob.npy (probabilities)")

#%% Normalize by token frequencies to compute PMI matrix
print("\n=== Computing Pointwise Mutual Information (PMI) ===")

# Compute marginal probabilities P(i) and P(j)
# P(i) = sum over j of P(i, j)
p_i = cooccurrence_prob.sum(axis=1)  # Shape: (vocab_size,)
p_j = cooccurrence_prob.sum(axis=0)  # Shape: (vocab_size,)

print(f"Marginal probability sum (should be ~1.0): {p_i.sum():.6f}")

# Compute expected co-occurrence under independence: P(i) * P(j)
# Using outer product to get all combinations
expected_cooccurrence = np.outer(p_i, p_j)  # Shape: (vocab_size, vocab_size)

# Compute PMI = log(P(i,j) / (P(i) * P(j)))
# Add small epsilon to avoid log(0) and division by zero
epsilon = 1e-10
pmi_matrix = np.log((cooccurrence_prob + epsilon) / (expected_cooccurrence + epsilon))

print(f"PMI matrix shape: {pmi_matrix.shape}")
print(f"PMI range: [{pmi_matrix.min():.2f}, {pmi_matrix.max():.2f}]")
print(f"PMI mean: {pmi_matrix.mean():.4f}, std: {pmi_matrix.std():.4f}")

# Positive PMI (PPMI) - commonly used in NLP, zeros out negative values
ppmi_matrix = np.maximum(pmi_matrix, 0)
print(f"PPMI non-zero entries: {np.count_nonzero(ppmi_matrix)} / {vocab_size * vocab_size}")

# Compute ratio without log: P(i,j) / (P(i) * P(j))
# This is just exp(PMI), but computed directly for clarity
ratio_matrix = (cooccurrence_prob + epsilon) / (expected_cooccurrence + epsilon)
print(f"\nRatio matrix (without log):")
print(f"  Range: [{ratio_matrix.min():.2f}, {ratio_matrix.max():.2f}]")
print(f"  Values > 1 (co-occur more than expected): {np.sum(ratio_matrix > 1)} / {vocab_size * vocab_size}")
print(f"  Values < 1 (co-occur less than expected): {np.sum(ratio_matrix < 1)} / {vocab_size * vocab_size}")


#%% Visualize ratio matrix (PMI without log)
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Ratio: P(i,j) / (P(i)×P(j))', 'Log₁₀ Ratio'),
    specs=[[{"type": "heatmap"}, {"type": "heatmap"}]],
    horizontal_spacing=0.15
)

# Plot 1: Linear scale ratio
fig.add_trace(
    go.Heatmap(
        z=ratio_matrix,
        colorscale='RdBu_r',
        zmid=1.0,  # Center at 1 (independent case)
        colorbar=dict(x=0.45, len=0.8, title="Ratio"),
        hovertemplate='Token %{y} & %{x}<br>Ratio: %{z:.3f}<extra></extra>'
    ),
    row=1, col=1
)

# Plot 2: Log scale for better visualization
log_ratio = np.log10(ratio_matrix)
fig.add_trace(
    go.Heatmap(
        z=log_ratio,
        colorscale='RdBu_r',
        zmid=0,  # log10(1) = 0
        colorbar=dict(x=1.0, len=0.8, title="log₁₀ Ratio"),
        hovertemplate='Token %{y} & %{x}<br>log₁₀ Ratio: %{z:.3f}<extra></extra>'
    ),
    row=1, col=2
)

# Update axes
fig.update_xaxes(title_text="Token j", row=1, col=1)
fig.update_yaxes(title_text="Token i", row=1, col=1)
fig.update_xaxes(title_text="Token j", row=1, col=2)
fig.update_yaxes(title_text="Token i", row=1, col=2)

fig.update_layout(
    height=600,
    width=1200,
    title_text="Co-occurrence Ratio: Values > 1 mean tokens co-occur more than expected",
    showlegend=False
)
fig.show()

#%% Visualize log(1 + ratio): log(1 + P(i,j)/(P(i)×P(j)))
# This transformation is like PMI but shifted: log(1 + ratio) instead of log(ratio)
log1p_ratio = np.log1p(ratio_matrix)  # log(1 + P(i,j)/(P(i)×P(j)))

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('log(1 + P(i,j)/(P(i)×P(j)))', 'Ratio for comparison'),
    specs=[[{"type": "heatmap"}, {"type": "heatmap"}]],
    horizontal_spacing=0.15
)

# Plot 1: log(1 + ratio)
fig.add_trace(
    go.Heatmap(
        z=log1p_ratio,
        colorscale='Viridis',
        colorbar=dict(x=0.45, len=0.8, title="log(1+ratio)"),
        hovertemplate='Token %{y} & %{x}<br>log(1+ratio): %{z:.4f}<extra></extra>'
    ),
    row=1, col=1
)

# Plot 2: Raw ratio for comparison
fig.add_trace(
    go.Heatmap(
        z=ratio_matrix,
        colorscale='RdBu_r',
        zmid=1.0,
        colorbar=dict(x=1.0, len=0.8, title="ratio"),
        hovertemplate='Token %{y} & %{x}<br>ratio: %{z:.3f}<extra></extra>'
    ),
    row=1, col=2
)

# Update axes
fig.update_xaxes(title_text="Token j", row=1, col=1)
fig.update_yaxes(title_text="Token i", row=1, col=1)
fig.update_xaxes(title_text="Token j", row=1, col=2)
fig.update_yaxes(title_text="Token i", row=1, col=2)

fig.update_layout(
    height=600,
    width=1200,
    title_text="log(1 + ratio): Smoothed version of PMI, handles zeros better",
    showlegend=False
)
fig.show()

#%% Visualize PMI and PPMI matrices
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Pointwise Mutual Information (PMI)', 'Positive PMI (PPMI)'),
    specs=[[{"type": "heatmap"}, {"type": "heatmap"}]],
    horizontal_spacing=0.15
)

# Plot 1: Full PMI (can be negative)
fig.add_trace(
    go.Heatmap(
        z=pmi_matrix,
        colorscale='RdBu_r',
        zmid=0,
        colorbar=dict(x=0.45, len=0.8, title="PMI"),
        hovertemplate='Token %{y} & %{x}<br>PMI: %{z:.3f}<extra></extra>'
    ),
    row=1, col=1
)

# Plot 2: Positive PMI (PPMI)
fig.add_trace(
    go.Heatmap(
        z=ppmi_matrix,
        colorscale='Viridis',
        colorbar=dict(x=1.0, len=0.8, title="PPMI"),
        hovertemplate='Token %{y} & %{x}<br>PPMI: %{z:.3f}<extra></extra>'
    ),
    row=1, col=2
)

# Update axes
fig.update_xaxes(title_text="Token j", row=1, col=1)
fig.update_yaxes(title_text="Token i", row=1, col=1)
fig.update_xaxes(title_text="Token j", row=1, col=2)
fig.update_yaxes(title_text="Token i", row=1, col=2)

fig.update_layout(
    height=500,
    width=1200,
    title_text="PMI Analysis: Tokens co-occurring more/less than expected by chance",
    showlegend=False
)
fig.show()

#%% Analyze co-occurrence structure by depth
print(f"\n=== Co-occurrence Analysis by Depth ===")
depth = 3
tokens_per_cluster = 16

for d in range(depth):
    start_idx = d * tokens_per_cluster
    end_idx = (d + 1) * tokens_per_cluster

    # Within-depth co-occurrences
    within_depth_cooc = cooccurrence_prob[start_idx:end_idx, start_idx:end_idx]
    within_depth_mean = within_depth_cooc.mean()

    # Cross-depth co-occurrences (with other depths)
    cross_depth_cooc = np.concatenate([
        cooccurrence_prob[start_idx:end_idx, :start_idx],
        cooccurrence_prob[start_idx:end_idx, end_idx:]
    ], axis=1)
    cross_depth_mean = cross_depth_cooc.mean()

    print(f"\nDepth {d} (tokens {start_idx}-{end_idx-1}):")
    print(f"  Mean within-depth co-occurrence: {within_depth_mean:.6f}")
    print(f"  Mean cross-depth co-occurrence: {cross_depth_mean:.6f}")
    print(f"  Ratio (within/cross): {within_depth_mean / cross_depth_mean:.2f}x")

    # PMI analysis
    within_depth_pmi = pmi_matrix[start_idx:end_idx, start_idx:end_idx]
    within_depth_pmi_mean = within_depth_pmi.mean()

    cross_depth_pmi = np.concatenate([
        pmi_matrix[start_idx:end_idx, :start_idx],
        pmi_matrix[start_idx:end_idx, end_idx:]
    ], axis=1)
    cross_depth_pmi_mean = cross_depth_pmi.mean()

    print(f"  Mean within-depth PMI: {within_depth_pmi_mean:.4f}")
    print(f"  Mean cross-depth PMI: {cross_depth_pmi_mean:.4f}")

#%% Visualize depth-structured co-occurrence
fig = go.Figure()

fig.add_trace(
    go.Heatmap(
        z=log_cooccurrence,
        colorscale='Viridis',
        colorbar=dict(title="log₁₀ P(i,j)"),
        hovertemplate='Token %{y} & %{x}<br>log P(i,j): %{z:.2f}<extra></extra>'
    )
)

# Add depth boundary lines
shapes = []
for d in range(1, depth):
    boundary = d * tokens_per_cluster - 0.5
    shapes.append(dict(
        type="line",
        x0=-0.5, x1=vocab_size - 0.5,
        y0=boundary, y1=boundary,
        line=dict(color="red", width=2, dash="dash")
    ))
    shapes.append(dict(
        type="line",
        x0=boundary, x1=boundary,
        y0=-0.5, y1=vocab_size - 0.5,
        line=dict(color="red", width=2, dash="dash")
    ))

# Add depth labels
annotations = []
for d in range(depth):
    mid_point = (d + 0.5) * tokens_per_cluster
    annotations.append(dict(
        x=-2.5, y=mid_point,
        text=f'<b>D{d}</b>',
        showarrow=False,
        xanchor='right',
        font=dict(size=12, color='red')
    ))
    annotations.append(dict(
        x=mid_point, y=-2.5,
        text=f'<b>D{d}</b>',
        showarrow=False,
        yanchor='top',
        font=dict(size=12, color='red')
    ))

fig.update_layout(
    title=f'Co-occurrence Matrix with Depth Boundaries (window={window_size})',
    xaxis_title='Token j',
    yaxis_title='Token i',
    width=900, height=900,
    shapes=shapes,
    annotations=annotations,
    xaxis=dict(scaleanchor="y", constrain="domain"),
    yaxis=dict(constrain="domain")
)

fig.show()

#%% Visualize depth-structured PMI matrix
fig = go.Figure()

fig.add_trace(
    go.Heatmap(
        z=pmi_matrix,
        colorscale='RdBu_r',
        zmid=0,
        colorbar=dict(title="PMI"),
        hovertemplate='Token %{y} & %{x}<br>PMI: %{z:.3f}<extra></extra>'
    )
)

# Add depth boundary lines (reuse shapes from before)
shapes = []
for d in range(1, depth):
    boundary = d * tokens_per_cluster - 0.5
    shapes.append(dict(
        type="line",
        x0=-0.5, x1=vocab_size - 0.5,
        y0=boundary, y1=boundary,
        line=dict(color="black", width=2, dash="dash")
    ))
    shapes.append(dict(
        type="line",
        x0=boundary, x1=boundary,
        y0=-0.5, y1=vocab_size - 0.5,
        line=dict(color="black", width=2, dash="dash")
    ))

# Add depth labels
annotations = []
for d in range(depth):
    mid_point = (d + 0.5) * tokens_per_cluster
    annotations.append(dict(
        x=-2.5, y=mid_point,
        text=f'<b>D{d}</b>',
        showarrow=False,
        xanchor='right',
        font=dict(size=12)
    ))
    annotations.append(dict(
        x=mid_point, y=-2.5,
        text=f'<b>D{d}</b>',
        showarrow=False,
        yanchor='top',
        font=dict(size=12)
    ))

fig.update_layout(
    title=f'PMI Matrix with Depth Boundaries (window={window_size})',
    xaxis_title='Token j',
    yaxis_title='Token i',
    width=900, height=900,
    shapes=shapes,
    annotations=annotations,
    xaxis=dict(scaleanchor="y", constrain="domain"),
    yaxis=dict(constrain="domain")
)

fig.show()

#%%