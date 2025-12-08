from hmm import RRXOR, Z1R, Mess3Proc
from model import HookedTransformerModel, MaskedHeadTransformerModel
import yaml
import torch as t
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pickle
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import math
from jaxtyping import Int, Float
import numpy as np
import einops
from utils import print_shape
import pandas as pd
import pandas as pd

from nnsight import NNsight
import transformer_lens


def get_device() -> str:
    """Get the best available device in order: cuda -> mps -> cpu"""
    if t.cuda.is_available():
        return "cuda"
    elif t.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def load_model_from_weights(model_dir: str, file_name: str = 'final_model_mess3.pt'):
    """Load a model from a weights file, load with nnsight"""
    with open(os.path.join(model_dir, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["name"]
    try:
        model = globals()[model_name](
            config_path=os.path.join(model_dir, "config.yaml")
        )
    except KeyError:
        raise ValueError(f"Model {model_name} not found in globals, possibly due to an absent import")

    # Load the state dict
    checkpoint = t.load(os.path.join(model_dir, file_name))

    # Check if this is a checkpoint file (with model_state_dict) or direct weights
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # This is a checkpoint file from training
        state_dict = checkpoint['model_state_dict']
        print(f"Loading from checkpoint (step/epoch: {checkpoint.get('step', checkpoint.get('epoch', 'unknown'))})")
    else:
        # This is a direct state dict
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    print(f"Loaded model {model_name} from {os.path.join(model_dir, file_name)} on {device}")
    return model


def extract_residual_stream(
    model: nn.Module,
    sequences: Int[t.Tensor, 'batch seq_len'],
    device: str | None = None,
    device: str | None = None,
) -> Float[t.Tensor, 'layer batch seq_len d_model']:
    if device is None:
        device = get_device()

    if device is None:
        device = get_device()

    model.eval()
    model.to(device)
    sequences = sequences.to(device)

    sequences = sequences.to(device)

    with t.no_grad():
        # at the moment only have two variants of model architectures, they behave similarly loss-wise.
        # Follow the grammar of NNsight
        if model.__class__.__name__ == "MaskedHeadTransformerModel":
            actv_list = []
            with model.trace(sequences) as tracer:
                for layer in model.model.layers:
                    actv = layer.output[0].save()
                    actv_list.append(actv.cpu())
        elif model.__class__.__name__ == "HookedTransformerModel":
            # Follow the grammar of TransformerLens
            actv_list = []
            logits, cache = model.model.run_with_cache(sequences)
            for layer_idx in range(model.model.cfg.n_layers):
                actv = cache["resid_post", layer_idx]
                actv_list.append(actv.cpu())
        
    actvs = t.stack(actv_list)
    return actvs

def learn_affine_mapping(
    actvs: Float[t.Tensor, 'batch d_actv'],
    belief_states: Float[t.Tensor, 'batch d_vocab'],
):
    # Fit a linear regression model to map actvs to belief states
    actvs = actvs.cpu().numpy()
    belief_states = belief_states.cpu().numpy()
    regressor = LinearRegression()
    regressor.fit(actvs, belief_states)


    predictions = regressor.predict(actvs)
    mse = np.mean((predictions - belief_states)**2)

    # Compute R² score (proportion of variance explained)
    r2_score = regressor.score(actvs, belief_states)

    # Compute baseline MSE (predicting the mean)
    baseline_mse = np.mean((belief_states - belief_states.mean(axis=0))**2)

    return regressor, predictions, mse, r2_score, baseline_mse

    # Compute R² score (proportion of variance explained)
    r2_score = regressor.score(actvs, belief_states)

    # Compute baseline MSE (predicting the mean)
    baseline_mse = np.mean((belief_states - belief_states.mean(axis=0))**2)

    return regressor, predictions, mse, r2_score, baseline_mse

def pca_concat_actvs(
    raw_actvs: Float[t.Tensor, 'layer batch seq_len d_model'],
    belief_states: Float[t.Tensor, 'batch seq_len d_vocab'],
    use_last_pos: bool = True, # for some reason it's better to use the last position of the sequences for the PCA?
    save_dir: str = "pca_results",
    run_id: str = "",
    save_data: bool = True,  # Whether to save activations, targets, and predictions
    save_data: bool = True,  # Whether to save activations, targets, and predictions
):
    actvs = einops.rearrange(
        raw_actvs,
        'layer batch seq_len d_model -> batch seq_len (layer d_model)',
    )
    if use_last_pos:
        actvs = actvs[:, -1, :]
        belief_states = belief_states[:, -1, :]
    else:
        actvs = einops.rearrange(
            actvs,
            'batch seq_len concat_dim -> (batch seq_len) concat_dim',
        )
        belief_states = einops.rearrange(
            belief_states,
            'batch seq_len d_vocab -> (batch seq_len) d_vocab',
        )
    
    regressor, predictions, mse, r2_score, baseline_mse = learn_affine_mapping(actvs, belief_states)
    regressor, predictions, mse, r2_score, baseline_mse = learn_affine_mapping(actvs, belief_states)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save regressor (always save this as it's small)

    # Save regressor (always save this as it's small)
    with open(os.path.join(save_dir, f"{run_id}_regressor.pkl"), "wb") as f:
        pickle.dump(regressor, f)

    # Save data if requested
    if save_data:
        np.savez_compressed(
            os.path.join(save_dir, f"{run_id}_data.npz"),
            activations=actvs.cpu().numpy(),
            targets=belief_states.cpu().numpy(),
            predictions=predictions,
        )
        print(f"Data saved to {os.path.join(save_dir, f'{run_id}_data.npz')}")

    # Save metrics as a dataframe
    results_df = pd.DataFrame({
        'layer': ['concat'],
        'mse': [mse],
        'r2_score': [r2_score],
        'baseline_mse': [baseline_mse],
    })
    results_df.to_csv(os.path.join(save_dir, f"{run_id}_metrics.csv"), index=False)

    print(f"MSE: {mse:.6f}, R²: {r2_score:.6f}, Baseline MSE: {baseline_mse:.6f}")
    print(f"Metrics saved to {os.path.join(save_dir, f'{run_id}_metrics.csv')}")

    return regressor, predictions, mse, r2_score, baseline_mse

    # Save data if requested
    if save_data:
        np.savez_compressed(
            os.path.join(save_dir, f"{run_id}_data.npz"),
            activations=actvs.cpu().numpy(),
            targets=belief_states.cpu().numpy(),
            predictions=predictions,
        )
        print(f"Data saved to {os.path.join(save_dir, f'{run_id}_data.npz')}")

    # Save metrics as a dataframe
    results_df = pd.DataFrame({
        'layer': ['concat'],
        'mse': [mse],
        'r2_score': [r2_score],
        'baseline_mse': [baseline_mse],
    })
    results_df.to_csv(os.path.join(save_dir, f"{run_id}_metrics.csv"), index=False)

    print(f"MSE: {mse:.6f}, R²: {r2_score:.6f}, Baseline MSE: {baseline_mse:.6f}")
    print(f"Metrics saved to {os.path.join(save_dir, f'{run_id}_metrics.csv')}")

    return regressor, predictions, mse, r2_score, baseline_mse

def pca_layer_wise_actvs(
    raw_actvs: Float[t.Tensor, 'layer batch seq_len d_model'],
    belief_states: Float[t.Tensor, 'batch seq_len d_vocab'],
    use_last_pos: bool = True,
    save_dir: str = "pca_results",
    run_id: str = "",
    save_data: bool = True,  # Whether to save activations, targets, and predictions
    save_data: bool = True,  # Whether to save activations, targets, and predictions
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    regressors = []
    mses = []
    r2_scores = []
    baseline_mses = []
    r2_scores = []
    baseline_mses = []
    predictions = []


    if use_last_pos:
        belief_states = belief_states[:, -1, :]
    else:
        belief_states = einops.rearrange(
            belief_states,
            'batch seq_len d_vocab -> (batch seq_len) d_vocab',
        )


    print_shape(belief_states)


    for i in range(raw_actvs.shape[0]):
        actvs = raw_actvs[i, :, :, :]
        if use_last_pos:
            actvs = actvs[:, -1, :]
        else:
            actvs = einops.rearrange(
                actvs,
                'batch seq_len d_model -> (batch seq_len) d_model',
            )
        print_shape(actvs)

        regressor, preds, mse, r2_score, baseline_mse = learn_affine_mapping(actvs, belief_states)


        regressor, preds, mse, r2_score, baseline_mse = learn_affine_mapping(actvs, belief_states)

        regressors.append(regressor)
        predictions.append(preds)
        mses.append(mse)
        r2_scores.append(r2_score)
        baseline_mses.append(baseline_mse)
        print(f"Layer {i} - MSE: {mse:.6f}, R²: {r2_score:.6f}, Baseline MSE: {baseline_mse:.6f}")
        r2_scores.append(r2_score)
        baseline_mses.append(baseline_mse)
        print(f"Layer {i} - MSE: {mse:.6f}, R²: {r2_score:.6f}, Baseline MSE: {baseline_mse:.6f}")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save regressors (always save, they're small)

    # Save regressors (always save, they're small)
    with open(os.path.join(save_dir, f"{run_id}_regressors.pkl"), "wb") as f:
        pickle.dump(regressors, f)

    # Save data if requested
    if save_data:
        # Save layer-wise activations and predictions
        data_dict = {
            'targets': belief_states.cpu().numpy(),
        }
        # Add layer-wise activations and predictions
        for i in range(raw_actvs.shape[0]):
            layer_actvs = raw_actvs[i, :, :, :]
            if use_last_pos:
                layer_actvs = layer_actvs[:, -1, :]
            else:
                layer_actvs = einops.rearrange(
                    layer_actvs,
                    'batch seq_len d_model -> (batch seq_len) d_model',
                )
            data_dict[f'layer_{i}_activations'] = layer_actvs.cpu().numpy()
            data_dict[f'layer_{i}_predictions'] = predictions[i]

        np.savez_compressed(
            os.path.join(save_dir, f"{run_id}_data.npz"),
            **data_dict
        )
        print(f"Data saved to {os.path.join(save_dir, f'{run_id}_data.npz')}")

    # Save metrics as a dataframe
    results_df = pd.DataFrame({
        'layer': range(len(mses)),
        'mse': mses,
        'r2_score': r2_scores,
        'baseline_mse': baseline_mses,
    })
    results_df.to_csv(os.path.join(save_dir, f"{run_id}_metrics.csv"), index=False)
    print(f"Metrics saved to {os.path.join(save_dir, f'{run_id}_metrics.csv')}")

    return regressors, predictions, mses, r2_scores, baseline_mses


def load_pca_results(save_dir: str, run_id: str):
    """
    Load saved PCA results from disk.

    Args:
        save_dir: Directory where results are saved
        run_id: Run identifier used when saving

    Returns:
        dict containing:
            - 'metrics': DataFrame with MSE, R², and baseline MSE per layer
            - 'regressors': List of fitted LinearRegression models
            - 'data': Dict with activations, targets, and predictions (if saved)
    """
    results = {}

    # Load metrics
    metrics_path = os.path.join(save_dir, f"{run_id}_metrics.csv")
    if os.path.exists(metrics_path):
        results['metrics'] = pd.read_csv(metrics_path)
        print(f"Loaded metrics from {metrics_path}")

    # Load regressors
    regressors_path = os.path.join(save_dir, f"{run_id}_regressors.pkl")
    if os.path.exists(regressors_path):
        with open(regressors_path, "rb") as f:
            results['regressors'] = pickle.load(f)
        print(f"Loaded regressors from {regressors_path}")

    # Load data if available
    data_path = os.path.join(save_dir, f"{run_id}_data.npz")
    if os.path.exists(data_path):
        results['data'] = dict(np.load(data_path))
        print(f"Loaded data from {data_path}")
        print(f"  Available keys: {list(results['data'].keys())}")

    return results


def barycentric_to_cartesian_2d(simplex_coords: np.ndarray) -> np.ndarray:
    """
    Convert 3D barycentric (simplex) coordinates to 2D Cartesian coordinates.

    Maps points on a 3-simplex (where coordinates sum to 1) to an equilateral triangle.

    Args:
        simplex_coords: (batch, 3) array where each row sums to 1

    Returns:
        cartesian_coords: (batch, 2) array of 2D positions
    """
    # Vertices of equilateral triangle
    # v0 = (0, 0) for simplex coordinate [1, 0, 0]
    # v1 = (1, 0) for simplex coordinate [0, 1, 0]
    # v2 = (0.5, sqrt(3)/2) for simplex coordinate [0, 0, 1]
    v0 = np.array([0.0, 0.0])
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3) / 2])

    # Linear combination: position = p0*v0 + p1*v1 + p2*v2
    p0 = simplex_coords[:, 0:1]
    p1 = simplex_coords[:, 1:2]
    p2 = simplex_coords[:, 2:3]

    cartesian = p0 * v0 + p1 * v1 + p2 * v2

    return cartesian


def plot_in_simplex(
    points: np.ndarray,
    labels: np.ndarray | None = None,
    title: str = "3-Simplex Visualization",
    alpha: float = 0.5,
    s: float = 1,
    figsize: tuple = (10, 10),
    show_grid: bool = True,
    save_path: str | None = None,
):
    """
    Plot points on a 3-simplex (equilateral triangle).

    Args:
        points: (batch, 3) array of barycentric coordinates (each row sums to 1)
        labels: Optional (batch,) array of labels for coloring points
        title: Plot title
        alpha: Point transparency
        s: Point size
        figsize: Figure size
        show_grid: Whether to show simplex grid lines
        save_path: Optional path to save the figure
    """
    # Convert to 2D
    coords_2d = barycentric_to_cartesian_2d(points)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the simplex triangle
    triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
    ax.plot(triangle[:, 0], triangle[:, 1], 'k-', linewidth=2)

    # Add vertex labels
    ax.text(-0.05, -0.05, 'State 0', fontsize=12, ha='right')
    ax.text(1.05, -0.05, 'State 1', fontsize=12, ha='left')
    ax.text(0.5, np.sqrt(3) / 2 + 0.05, 'State 2', fontsize=12, ha='center')

    # Add grid lines
    if show_grid:
        n_lines = 10
        for i in range(1, n_lines):
            t = i / n_lines
            # Lines parallel to each edge
            # Parallel to base (constant p2)
            p1_start = np.array([t, 0, 1 - t])
            p1_end = np.array([0, t, 1 - t])
            c1_start = barycentric_to_cartesian_2d(p1_start[np.newaxis, :])[0]
            c1_end = barycentric_to_cartesian_2d(p1_end[np.newaxis, :])[0]
            ax.plot([c1_start[0], c1_end[0]], [c1_start[1], c1_end[1]], 'k-', alpha=0.1, linewidth=0.5)

            # Parallel to left edge (constant p1)
            p2_start = np.array([t, 1 - t, 0])
            p2_end = np.array([0, 1 - t, t])
            c2_start = barycentric_to_cartesian_2d(p2_start[np.newaxis, :])[0]
            c2_end = barycentric_to_cartesian_2d(p2_end[np.newaxis, :])[0]
            ax.plot([c2_start[0], c2_end[0]], [c2_start[1], c2_end[1]], 'k-', alpha=0.1, linewidth=0.5)

            # Parallel to right edge (constant p0)
            p3_start = np.array([1 - t, t, 0])
            p3_end = np.array([1 - t, 0, t])
            c3_start = barycentric_to_cartesian_2d(p3_start[np.newaxis, :])[0]
            c3_end = barycentric_to_cartesian_2d(p3_end[np.newaxis, :])[0]
            ax.plot([c3_start[0], c3_end[0]], [c3_start[1], c3_end[1]], 'k-', alpha=0.1, linewidth=0.5)

    # Plot points
    if labels is not None:
        scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c=labels, s=s, alpha=alpha, cmap='viridis')
        plt.colorbar(scatter, ax=ax, label='Label')
    else:
        ax.scatter(coords_2d[:, 0], coords_2d[:, 1], s=s, alpha=alpha, color='blue')

    ax.set_aspect('equal')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, np.sqrt(3) / 2 + 0.1)
    ax.set_title(title, fontsize=14)
    ax.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    return fig, ax

    # Save data if requested
    if save_data:
        # Save layer-wise activations and predictions
        data_dict = {
            'targets': belief_states.cpu().numpy(),
        }
        # Add layer-wise activations and predictions
        for i in range(raw_actvs.shape[0]):
            layer_actvs = raw_actvs[i, :, :, :]
            if use_last_pos:
                layer_actvs = layer_actvs[:, -1, :]
            else:
                layer_actvs = einops.rearrange(
                    layer_actvs,
                    'batch seq_len d_model -> (batch seq_len) d_model',
                )
            data_dict[f'layer_{i}_activations'] = layer_actvs.cpu().numpy()
            data_dict[f'layer_{i}_predictions'] = predictions[i]

        np.savez_compressed(
            os.path.join(save_dir, f"{run_id}_data.npz"),
            **data_dict
        )
        print(f"Data saved to {os.path.join(save_dir, f'{run_id}_data.npz')}")

    # Save metrics as a dataframe
    results_df = pd.DataFrame({
        'layer': range(len(mses)),
        'mse': mses,
        'r2_score': r2_scores,
        'baseline_mse': baseline_mses,
    })
    results_df.to_csv(os.path.join(save_dir, f"{run_id}_metrics.csv"), index=False)
    print(f"Metrics saved to {os.path.join(save_dir, f'{run_id}_metrics.csv')}")

    return regressors, predictions, mses, r2_scores, baseline_mses


def load_pca_results(save_dir: str, run_id: str):
    """
    Load saved PCA results from disk.

    Args:
        save_dir: Directory where results are saved
        run_id: Run identifier used when saving

    Returns:
        dict containing:
            - 'metrics': DataFrame with MSE, R², and baseline MSE per layer
            - 'regressors': List of fitted LinearRegression models
            - 'data': Dict with activations, targets, and predictions (if saved)
    """
    results = {}

    # Load metrics
    metrics_path = os.path.join(save_dir, f"{run_id}_metrics.csv")
    if os.path.exists(metrics_path):
        results['metrics'] = pd.read_csv(metrics_path)
        print(f"Loaded metrics from {metrics_path}")

    # Load regressors
    regressors_path = os.path.join(save_dir, f"{run_id}_regressors.pkl")
    if os.path.exists(regressors_path):
        with open(regressors_path, "rb") as f:
            results['regressors'] = pickle.load(f)
        print(f"Loaded regressors from {regressors_path}")

    # Load data if available
    data_path = os.path.join(save_dir, f"{run_id}_data.npz")
    if os.path.exists(data_path):
        results['data'] = dict(np.load(data_path))
        print(f"Loaded data from {data_path}")
        print(f"  Available keys: {list(results['data'].keys())}")

    return results


def barycentric_to_cartesian_2d(simplex_coords: np.ndarray) -> np.ndarray:
    """
    Convert 3D barycentric (simplex) coordinates to 2D Cartesian coordinates.

    Maps points on a 3-simplex (where coordinates sum to 1) to an equilateral triangle.

    Args:
        simplex_coords: (batch, 3) array where each row sums to 1

    Returns:
        cartesian_coords: (batch, 2) array of 2D positions
    """
    # Vertices of equilateral triangle
    # v0 = (0, 0) for simplex coordinate [1, 0, 0]
    # v1 = (1, 0) for simplex coordinate [0, 1, 0]
    # v2 = (0.5, sqrt(3)/2) for simplex coordinate [0, 0, 1]
    v0 = np.array([0.0, 0.0])
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3) / 2])

    # Linear combination: position = p0*v0 + p1*v1 + p2*v2
    p0 = simplex_coords[:, 0:1]
    p1 = simplex_coords[:, 1:2]
    p2 = simplex_coords[:, 2:3]

    cartesian = p0 * v0 + p1 * v1 + p2 * v2

    return cartesian


def plot_in_simplex(
    points: np.ndarray,
    labels: np.ndarray | None = None,
    title: str = "3-Simplex Visualization",
    alpha: float = 0.5,
    s: float = 1,
    figsize: tuple = (10, 10),
    show_grid: bool = True,
    save_path: str | None = None,
):
    """
    Plot points on a 3-simplex (equilateral triangle).

    Args:
        points: (batch, 3) array of barycentric coordinates (each row sums to 1)
        labels: Optional (batch,) array of labels for coloring points
        title: Plot title
        alpha: Point transparency
        s: Point size
        figsize: Figure size
        show_grid: Whether to show simplex grid lines
        save_path: Optional path to save the figure
    """
    # Convert to 2D
    coords_2d = barycentric_to_cartesian_2d(points)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the simplex triangle
    triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
    ax.plot(triangle[:, 0], triangle[:, 1], 'k-', linewidth=2)

    # Add vertex labels
    ax.text(-0.05, -0.05, 'State 0', fontsize=12, ha='right')
    ax.text(1.05, -0.05, 'State 1', fontsize=12, ha='left')
    ax.text(0.5, np.sqrt(3) / 2 + 0.05, 'State 2', fontsize=12, ha='center')

    # Add grid lines
    if show_grid:
        n_lines = 10
        for i in range(1, n_lines):
            t = i / n_lines
            # Lines parallel to each edge
            # Parallel to base (constant p2)
            p1_start = np.array([t, 0, 1 - t])
            p1_end = np.array([0, t, 1 - t])
            c1_start = barycentric_to_cartesian_2d(p1_start[np.newaxis, :])[0]
            c1_end = barycentric_to_cartesian_2d(p1_end[np.newaxis, :])[0]
            ax.plot([c1_start[0], c1_end[0]], [c1_start[1], c1_end[1]], 'k-', alpha=0.1, linewidth=0.5)

            # Parallel to left edge (constant p1)
            p2_start = np.array([t, 1 - t, 0])
            p2_end = np.array([0, 1 - t, t])
            c2_start = barycentric_to_cartesian_2d(p2_start[np.newaxis, :])[0]
            c2_end = barycentric_to_cartesian_2d(p2_end[np.newaxis, :])[0]
            ax.plot([c2_start[0], c2_end[0]], [c2_start[1], c2_end[1]], 'k-', alpha=0.1, linewidth=0.5)

            # Parallel to right edge (constant p0)
            p3_start = np.array([1 - t, t, 0])
            p3_end = np.array([1 - t, 0, t])
            c3_start = barycentric_to_cartesian_2d(p3_start[np.newaxis, :])[0]
            c3_end = barycentric_to_cartesian_2d(p3_end[np.newaxis, :])[0]
            ax.plot([c3_start[0], c3_end[0]], [c3_start[1], c3_end[1]], 'k-', alpha=0.1, linewidth=0.5)

    # Plot points
    if labels is not None:
        scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c=labels, s=s, alpha=alpha, cmap='viridis')
        plt.colorbar(scatter, ax=ax, label='Label')
    else:
        ax.scatter(coords_2d[:, 0], coords_2d[:, 1], s=s, alpha=alpha, color='blue')

    ax.set_aspect('equal')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, np.sqrt(3) / 2 + 0.1)
    ax.set_title(title, fontsize=14)
    ax.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    return fig, ax


def visualize_simplex_3D(
    predictions: Float[t.Tensor, 'batch d_vocab'],
):
    """Deprecated: Use plot_in_simplex instead."""
    """Deprecated: Use plot_in_simplex instead."""
    pass

if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")

    device = get_device()
    print(f"Using device: {device}")

    model = load_model_from_weights(
        model_dir="records/20251127_134441",
        model_dir="records/20251127_134441",
        file_name="final_model_mess3.pt"
    )
    print(model)


    process = Mess3Proc()
    batch_obs = process.generate_data(
        batch_size=100_000, length=10, use_tqdm=True,
    )
    
    # get the belief states
    belief_states = process.mixed_state_presentation(batch_obs)
    print(belief_states.shape)
    actvs = extract_residual_stream(
        model=model,
        sequences=batch_obs,
    )
    print(actvs.shape)
    
    regressors, predictions, mses, r2_scores, baseline_mses = pca_layer_wise_actvs(
    regressors, predictions, mses, r2_scores, baseline_mses = pca_layer_wise_actvs(
        raw_actvs=actvs,
        belief_states=t.tensor(belief_states, device=device),
        run_id="20251127_134441",
    )

    # Print summary
    print(f"\n{'='*60}")
    print("Summary of Results:")
    print(f"{'='*60}")
    print(f"Predictions shape: {predictions[0].shape}")
    results_df = pd.DataFrame({
        'layer': range(len(mses)),
        'mse': mses,
        'r2_score': r2_scores,
        'baseline_mse': baseline_mses,
    })
    print(results_df.to_string(index=False))

    # Example: To load the saved results later, use:
    # results = load_pca_results(save_dir="pca_results", run_id="20251127_134441")
    # metrics_df = results['metrics']
    # regressors = results['regressors']
    # data = results['data']  # Contains 'targets', 'layer_0_activations', 'layer_0_predictions', etc.


if __name__ == "__main__":
    time_stamp = '20251207_232030'
    device = get_device()
    print(f"Using device: {device}")

    model = load_model_from_weights(
        model_dir=f"records/{time_stamp}",
        file_name=f"checkpoints/checkpoint_step_80000.pt"
    )
    print(model)

    process = Mess3Proc()
    batch_obs = process.generate_data(
        batch_size=100_000, length=10, use_tqdm=True,
    )
    
    # get the belief states
    belief_states = process.mixed_state_presentation(batch_obs)
    print(belief_states.shape)
    actvs = extract_residual_stream(
        model=model,
        sequences=batch_obs,
    )
    print(actvs.shape)
    
    regressors, predictions, mses, r2_scores, baseline_mses = pca_layer_wise_actvs(
        raw_actvs=actvs,
        belief_states=t.tensor(belief_states, device=device),
        run_id=time_stamp,
    )

    # Print summary
    print(f"\n{'='*60}")
    print("Summary of Results:")
    print(f"{'='*60}")
    print(f"Predictions shape: {predictions[0].shape}")
    results_df = pd.DataFrame({
        'layer': range(len(mses)),
        'mse': mses,
        'r2_score': r2_scores,
        'baseline_mse': baseline_mses,
    })
    print(results_df.to_string(index=False))
    