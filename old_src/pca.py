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

from nnsight import NNsight
import transformer_lens

# Try importing MPI (optional dependency)
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False


def get_device() -> str:
    """Get the best available device in order: cuda -> mps -> cpu"""
    if t.cuda.is_available():
        return "cuda"
    elif t.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def load_model_from_weights(model_dir: str, file_name: str = 'final_model_mess3.pt', device: str | None = None):
    """Load a model from a weights file, load with nnsight

    Args:
        model_dir: Directory containing the model
        file_name: Name of the checkpoint file
        device: Device to load model on. If None, auto-detects best device
    """
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
    if device is None:
        device = get_device()

    checkpoint = t.load(os.path.join(model_dir, file_name), map_location=device)

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
) -> Float[t.Tensor, 'layer batch seq_len d_model']:
    if device is None:
        device = get_device()

    model.eval()
    model.to(device)
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

def pca_concat_actvs(
    raw_actvs: Float[t.Tensor, 'layer batch seq_len d_model'],
    belief_states: Float[t.Tensor, 'batch seq_len d_vocab'],
    use_last_pos: bool = True, # for some reason it's better to use the last position of the sequences for the PCA?
    save_dir: str = "pca_results",
    run_id: str = "",
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

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

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

def pca_layer_wise_actvs(
    raw_actvs: Float[t.Tensor, 'layer batch seq_len d_model'],
    belief_states: Float[t.Tensor, 'batch seq_len d_vocab'],
    use_last_pos: bool = True,
    save_dir: str = "pca_results",
    run_id: str = "",
    save_data: bool = True,  # Whether to save activations, targets, and predictions
):
    if save_data and save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    regressors = []
    mses = []
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

        regressors.append(regressor)
        predictions.append(preds)
        mses.append(mse)
        r2_scores.append(r2_score)
        baseline_mses.append(baseline_mse)
        print(f"Layer {i} - MSE: {mse:.6f}, R²: {r2_score:.6f}, Baseline MSE: {baseline_mse:.6f}")

    # Save regressors if requested
    if save_data and save_dir:
        with open(os.path.join(save_dir, f"regressors.pkl"), "wb") as f:
            pickle.dump(regressors, f)

    # Save data if requested
    if save_data and save_dir:
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
            os.path.join(save_dir, f"pca_data.npz"),
            **data_dict
        )
        print(f"Data saved to {os.path.join(save_dir, f'pca_data.npz')}")

        # Save metrics as a dataframe
        results_df = pd.DataFrame({
            'layer': range(len(mses)),
            'mse': mses,
            'r2_score': r2_scores,
            'baseline_mse': baseline_mses,
        })
        results_df.to_csv(os.path.join(save_dir, f"pca_metrics.csv"), index=False)
        print(f"Metrics saved to {os.path.join(save_dir, f'pca_metrics.csv')}")

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
    metrics_path = os.path.join(save_dir, f"pca_metrics.csv")
    if os.path.exists(metrics_path):
        results['metrics'] = pd.read_csv(metrics_path)
        print(f"Loaded metrics from {metrics_path}")

    # Load regressors
    regressors_path = os.path.join(save_dir, f"regressors.pkl")
    if os.path.exists(regressors_path):
        with open(regressors_path, "rb") as f:
            results['regressors'] = pickle.load(f)
        print(f"Loaded regressors from {regressors_path}")

    # Load data if available
    data_path = os.path.join(save_dir, f"pca_data.npz")
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

def run_pca_and_save_results(
    dirs: list[str] | str,
    batch_size: int = 100_000,
    seq_length: int = 16, # Currently the program is only consistent when this input is equal or larger than the actual n_ctx as in the model (if larger then truncate)
    use_mpi: bool = False,
    save_reusable_data_path: str = "pca_results/belief_states.npz",
    device: str = 'auto',
):
    """
    Run PCA analysis on model activations and save results.

    Args:
        dirs: Directory or list of directories containing trained models
        batch_size: Number of samples to generate for analysis
        seq_length: Length of sequences to generate. Should be >= max n_ctx of all models.
                   Sequences will be automatically truncated to match each model's n_ctx.
        use_mpi: Whether to use MPI for parallel processing of multiple directories
        save_reusable_data_path: Path to save belief states for reuse
        device: Device to use ('cuda', 'mps', 'cpu', or 'auto' for auto-detection)

    Context window handling:
        - Sequences are generated once with the specified seq_length
        - For each model, sequences are truncated to match the model's n_ctx from config
        - Belief states are similarly truncated to maintain alignment
        - This ensures compatibility across models with different context windows

    Error handling and robustness:
        - Validates each directory before processing (checks for config.yaml, model file)
        - Skips already-processed directories (checks for pca_data.npz)
        - Catches and logs errors, continues processing remaining directories
        - Provides detailed summary of successes and failures per rank
        - Automatic GPU recovery on CUDA errors

    MPI behavior:
        - Rank 0 generates belief states and broadcasts to all ranks
        - Each rank processes a subset of directories
        - Automatic GPU assignment: ranks distributed across available GPUs (round-robin)
        - Results are saved independently by each rank (no coordination needed)
    """
    if isinstance(dirs, str):
        dirs = [dirs]

    # Initialize MPI if requested
    if use_mpi:
        if not MPI_AVAILABLE:
            raise ImportError("MPI support requested but mpi4py not available. Install with: pip install mpi4py")

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        print(f"[Rank {rank}/{size}] MPI initialized")
    else:
        rank = 0
        size = 1

    # Resolve device with MPI-aware GPU allocation
    if device == 'auto':
        device = get_device()

    # If using CUDA with MPI, distribute ranks across available GPUs
    if use_mpi and device.startswith('cuda'):
        if device == 'cuda':  # No specific GPU specified
            # Distribute ranks across available GPUs
            if t.cuda.is_available():
                n_gpus = t.cuda.device_count()
                gpu_id = rank % n_gpus  # Round-robin GPU assignment
                device = f'cuda:{gpu_id}'
                print(f"[Rank {rank}/{size}] Assigned to GPU {gpu_id} (out of {n_gpus} GPUs)")
            else:
                print(f"[Rank {rank}] Warning: CUDA not available, falling back to CPU")
                device = 'cpu'
        # If specific GPU was requested (e.g., cuda:0), keep it as-is

    print(f"[Rank {rank}] Using device: {device}")

    # Rank 0 generates belief states, then broadcasts to all ranks
    if rank == 0:
        print(f"[Rank 0] Generating belief states with batch_size={batch_size}, seq_length={seq_length}")
        process = Mess3Proc()
        batch_obs = process.generate_data(
            batch_size=batch_size, length=seq_length, use_tqdm=True,
        )

        # get the belief states
        belief_states = process.mixed_state_presentation(batch_obs)
        print(f"[Rank 0] Belief states shape: {belief_states.shape}")

        # Save belief states (only rank 0 saves)
        os.makedirs(os.path.dirname(save_reusable_data_path), exist_ok=True)
        np.savez_compressed(save_reusable_data_path, belief_states=belief_states, batch_obs=batch_obs)
        print(f"[Rank 0] Saved belief states and observations to {save_reusable_data_path}")
    else:
        batch_obs = None
        belief_states = None

    # Broadcast belief states and observations to all ranks
    if use_mpi:
        print(f"[Rank {rank}] Waiting for broadcast...")
        batch_obs = comm.bcast(batch_obs, root=0)
        belief_states = comm.bcast(belief_states, root=0)
        print(f"[Rank {rank}] Received belief states with shape: {belief_states.shape}")

    # Distribute directories among ranks
    dirs_per_rank = np.array_split(dirs, size)
    my_dirs = dirs_per_rank[rank]

    print(f"[Rank {rank}] Processing {len(my_dirs)} directories out of {len(dirs)} total")
    if len(my_dirs) > 0:
        print(f"[Rank {rank}] Directories: {my_dirs}")

    # Track success and failures
    successful_dirs = []
    failed_dirs = []

    # Process assigned directories
    for idx, dir_ in enumerate(my_dirs):
        print(f"\n[Rank {rank}] {'='*70}")
        print(f"[Rank {rank}] Processing directory {idx+1}/{len(my_dirs)}: {dir_}")
        print(f"[Rank {rank}] {'='*70}")

        try:
            # Validate directory structure
            if not os.path.isdir(dir_):
                raise FileNotFoundError(f"Directory does not exist: {dir_}")

            config_path = os.path.join(dir_, "config.yaml")
            model_path = os.path.join(dir_, "final_model_mess3.pt")

            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Missing config.yaml in {dir_}")

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Missing final_model_mess3.pt in {dir_}")

            # Check if already processed
            output_path = os.path.join(dir_, "pca_data.npz")
            if os.path.exists(output_path):
                print(f"[Rank {rank}] Skipping {dir_} - already processed (pca_data.npz exists)")
                successful_dirs.append(dir_)
                continue

            # Load model config to get n_ctx
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            model_n_ctx = config['model']['n_ctx']
            print(f"[Rank {rank}] Model n_ctx: {model_n_ctx}, Data seq_length: {seq_length}")

            # Truncate sequences and belief states to match model's context window
            if model_n_ctx < seq_length:
                print(f"[Rank {rank}] Truncating sequences from {seq_length} to {model_n_ctx}")
                truncated_obs = batch_obs[:, :model_n_ctx]
                truncated_beliefs = belief_states[:, :model_n_ctx, :]
            else:
                truncated_obs = batch_obs
                truncated_beliefs = belief_states

            model = load_model_from_weights(
                model_dir=dir_,
                file_name="final_model_mess3.pt",
                device=device
            )
            actvs = extract_residual_stream(
                model=model,
                sequences=truncated_obs,
                device=device
            )
            print(f"[Rank {rank}] Activations shape: {actvs.shape}")

            regressors, predictions, mses, r2_scores, baseline_mses = pca_layer_wise_actvs(
                raw_actvs=actvs,
                belief_states=t.tensor(truncated_beliefs, device=device),
                save_dir=dir_,
                save_data=True,
            )

            # Print summary
            print(f"\n[Rank {rank}] {'='*60}")
            print(f"[Rank {rank}] Summary of Results for {dir_}:")
            print(f"[Rank {rank}] {'='*60}")
            print(f"[Rank {rank}] Predictions shape: {predictions[0].shape}")
            results_df = pd.DataFrame({
                'layer': range(len(mses)),
                'mse': mses,
                'r2_score': r2_scores,
                'baseline_mse': baseline_mses,
            })
            print(f"[Rank {rank}] " + "\n[Rank {rank}] ".join(results_df.to_string(index=False).split('\n')))

            # Clean up memory
            del model, actvs, regressors, predictions
            if t.cuda.is_available():
                t.cuda.empty_cache()
                # Synchronize GPU to ensure cleanup is complete before next iteration
                if device.startswith('cuda'):
                    t.cuda.synchronize(device)

            # Mark as successful
            successful_dirs.append(dir_)
            print(f"[Rank {rank}] ✓ Successfully processed {dir_}")

        except FileNotFoundError as e:
            print(f"[Rank {rank}] ✗ SKIPPED {dir_}: {e}")
            failed_dirs.append((dir_, f"FileNotFoundError: {e}"))
            continue

        except RuntimeError as e:
            if "CUDA" in str(e):
                error_msg = f"CUDA error: {str(e)[:100]}"
                print(f"[Rank {rank}] ✗ FAILED {dir_}: {error_msg}")
                print(f"[Rank {rank}] Device: {device}, Available GPUs: {t.cuda.device_count() if t.cuda.is_available() else 0}")
                print(f"[Rank {rank}] Suggestion: Reduce number of MPI ranks or ensure enough GPUs are available")
                # Try to recover by clearing cache
                if t.cuda.is_available():
                    t.cuda.empty_cache()
                    if device.startswith('cuda'):
                        try:
                            t.cuda.synchronize(device)
                        except:
                            pass
            else:
                error_msg = f"RuntimeError: {str(e)[:100]}"
                print(f"[Rank {rank}] ✗ FAILED {dir_}: {error_msg}")
                import traceback
                traceback.print_exc()

            failed_dirs.append((dir_, error_msg))
            continue

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)[:100]}"
            print(f"[Rank {rank}] ✗ FAILED {dir_}: {error_msg}")
            import traceback
            traceback.print_exc()
            failed_dirs.append((dir_, error_msg))
            continue

    # Print summary for this rank
    print(f"\n[Rank {rank}] {'='*70}")
    print(f"[Rank {rank}] PROCESSING SUMMARY")
    print(f"[Rank {rank}] {'='*70}")
    print(f"[Rank {rank}] Total directories assigned: {len(my_dirs)}")
    print(f"[Rank {rank}] Successfully processed: {len(successful_dirs)}")
    print(f"[Rank {rank}] Failed/Skipped: {len(failed_dirs)}")

    if successful_dirs:
        print(f"[Rank {rank}] \n✓ Successful directories:")
        for d in successful_dirs:
            print(f"[Rank {rank}]   - {d}")

    if failed_dirs:
        print(f"[Rank {rank}] \n✗ Failed/Skipped directories:")
        for d, error in failed_dirs:
            print(f"[Rank {rank}]   - {d}")
            print(f"[Rank {rank}]     Error: {error}")

    if use_mpi:
        # Synchronize all ranks before finishing
        comm.Barrier()
        if rank == 0:
            print(f"\n[Rank 0] All ranks completed processing")

    print(f"[Rank {rank}] Completed processing all assigned directories")

        
        

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run PCA analysis on model checkpoints")
    parser.add_argument("--use-mpi", action="store_true", help="Use MPI for parallel processing")
    parser.add_argument("--batch-size", type=int, default=100_000, help="Number of samples to generate")
    parser.add_argument("--seq-length", type=int, default=10, help="Length of sequences to generate")
    parser.add_argument("--device", type=str, default='auto', help="Device to use (cuda, mps, cpu, or auto)")
    parser.add_argument("--dirs", nargs="+", help="Specific directories to process (default: auto-discover)")
    args = parser.parse_args()

    if args.dirs:
        valid_records = [os.path.join("records", d) if not d.startswith("records/") else d for d in args.dirs]
    else:
        all_records = os.listdir("records")
        valid_records = []
        for run_id in all_records:
            record_path = os.path.join("records", run_id)
            if os.path.isdir(record_path) and "final_model_mess3.pt" in os.listdir(record_path) and 'pca_data.npz' not in os.listdir(record_path):
                valid_records.append(record_path)

    print(f"Found {len(valid_records)} valid records")

    # Run with or without MPI
    run_pca_and_save_results(
        dirs=valid_records,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        use_mpi=args.use_mpi,
        device=args.device,
    )

    # Example usage:
    # Serial execution:
    #   python pca.py
    #
    # MPI execution with 4 processes:
    #   mpirun -n 4 python pca.py --use-mpi
    #
    # Process specific directories with MPI:
    #   mpirun -n 4 python pca.py --use-mpi --dirs 20251207_232030 20251209_044936
    #
    # Specify device:
    #   python pca.py --device cuda
    #   mpirun -n 4 python pca.py --use-mpi --device cuda
    # time_stamp = '20251207_232030'
    # device = get_device()
    # print(f"Using device: {device}")

    # model = load_model_from_weights(
    #     model_dir=f"records/{time_stamp}",
    #     file_name=f"checkpoints/checkpoint_step_80000.pt"
    # )
    # print(model)

    # process = Mess3Proc()
    # batch_obs = process.generate_data(
    #     batch_size=100_000, length=10, use_tqdm=True,
    # )
    
    # # get the belief states
    # belief_states = process.mixed_state_presentation(batch_obs)
    # print(belief_states.shape)
    # actvs = extract_residual_stream(
    #     model=model,
    #     sequences=batch_obs,
    # )
    # print(actvs.shape)
    
    # regressors, predictions, mses, r2_scores, baseline_mses = pca_layer_wise_actvs(
    #     raw_actvs=actvs,
    #     belief_states=t.tensor(belief_states, device=device),
    #     run_id=time_stamp,
    # )

    # # Print summary
    # print(f"\n{'='*60}")
    # print("Summary of Results:")
    # print(f"{'='*60}")
    # print(f"Predictions shape: {predictions[0].shape}")
    # results_df = pd.DataFrame({
    #     'layer': range(len(mses)),
    #     'mse': mses,
    #     'r2_score': r2_scores,
    #     'baseline_mse': baseline_mses,
    # })
    # print(results_df.to_string(index=False))
    