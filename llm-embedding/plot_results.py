"""
Visualizations for the bilinear Deep Sets experiment.

Reads JSON result files from a sweep directory and produces figures:
1. R² distribution across random seeds (histogram)
2. Learned vs ground-truth predicted log P (scatter, best seed)
3. Learned embedding separation |e_0 - e_1| vs R² (scatter)
"""

import argparse
import json
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_results(results_dir):
    """Load all result_*.json files from directory."""
    files = sorted(glob.glob(os.path.join(results_dir, "result_*.json")))
    results = []
    for f in files:
        with open(f) as fh:
            results.append(json.load(fh))
    return results


def plot_r2_histogram(results, save_path):
    """Histogram of R² across random seeds."""
    r2_vals = [r['r_squared'] for r in results]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(r2_vals, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(np.median(r2_vals), color='red', ls='--', label=f'median={np.median(r2_vals):.4f}')
    ax.set_xlabel('$R^2$')
    ax.set_ylabel('Count')
    ax.set_title(f'$R^2$ across {len(results)} random initializations')
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_embedding_vs_r2(results, save_path):
    """Scatter: embedding separation |e_0 - e_1| vs R²."""
    r2_vals = []
    separations = []
    for r in results:
        e = np.array(r['learned']['embeddings'])
        sep = np.linalg.norm(e[0] - e[1])
        separations.append(sep)
        r2_vals.append(r['r_squared'])

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.scatter(separations, r2_vals, s=20, alpha=0.7)
    ax.set_xlabel('$|e_0 - e_1|$')
    ax.set_ylabel('$R^2$')
    ax.set_title('Embedding separation vs fit quality')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_predicted_vs_actual(results, data_path, save_path):
    """Scatter plot of model predictions vs true log P for best seed."""
    import torch
    from model import BilinearDeepSets

    # Find best seed
    best = max(results, key=lambda r: r['r_squared'])
    args = best['args']

    # Load data
    data = np.load(data_path, allow_pickle=True)
    sequences = torch.from_numpy(data['sequences']).long()
    log_probs = data['log_probs']

    # Reconstruct model
    V = len(best['learned']['embeddings'])
    r = len(best['learned']['embeddings'][0])
    model = BilinearDeepSets(V=V, r=r)
    with torch.no_grad():
        model.embedding.weight.copy_(torch.tensor(best['learned']['embeddings']))
        model.M.copy_(torch.tensor(best['learned']['M']))
        model.d.copy_(torch.tensor(best['learned']['d']))
        model.c.fill_(best['learned']['c'])
        predictions = model(sequences).numpy()

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(log_probs, predictions, s=1, alpha=0.3)
    lims = [min(log_probs.min(), predictions.min()), max(log_probs.max(), predictions.max())]
    ax.plot(lims, lims, 'r--', lw=1, label='$y=x$')
    ax.set_xlabel('True $\\log P(w_{1:L})$')
    ax.set_ylabel('Model $\\log p_\\theta(w_{1:L})$')
    ax.set_title(f'Best seed (R²={best["r_squared"]:.4f})')
    ax.legend()
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", help="Directory with result_*.json files")
    parser.add_argument("--data_path", default=None, help="Path to .npz data for scatter plot")
    parser.add_argument("--out_dir", default=None, help="Output directory for figures (defaults to results_dir)")
    args = parser.parse_args()

    out_dir = args.out_dir or args.results_dir
    os.makedirs(out_dir, exist_ok=True)

    results = load_results(args.results_dir)
    if not results:
        print(f"No result files found in {args.results_dir}")
        return

    print(f"Loaded {len(results)} results")
    r2_vals = [r['r_squared'] for r in results]
    print(f"R² stats: min={min(r2_vals):.4f}, median={np.median(r2_vals):.4f}, "
          f"max={max(r2_vals):.4f}, mean={np.mean(r2_vals):.4f}")

    plot_r2_histogram(results, os.path.join(out_dir, "r2_histogram.pdf"))
    plot_embedding_vs_r2(results, os.path.join(out_dir, "embedding_vs_r2.pdf"))

    if args.data_path and os.path.exists(args.data_path):
        plot_predicted_vs_actual(results, args.data_path, os.path.join(out_dir, "pred_vs_actual.pdf"))


if __name__ == "__main__":
    main()
