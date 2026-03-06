"""
Model loading and residual-stream activation extraction.

Handles device management, config reading, and torch operations.
Uses existing utilities from src/utils.py.
"""

import os
import glob
import yaml
import torch as t
import numpy as np
from transformer_lens import HookedTransformer

from utils import initialize_transformer_from_yaml, create_process_from_dict
from hmm import HMM


def get_device() -> str:
    """Get the best available device: cuda -> mps -> cpu."""
    if t.cuda.is_available():
        return "cuda"
    elif t.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def _find_checkpoint(model_dir: str) -> str:
    """
    Auto-detect the best checkpoint file in a model directory.

    Priority: best_model.pt > latest.pt > highest-epoch checkpoint_epoch_*.pt
    """
    for name in ["best_model.pt", "latest.pt"]:
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            return name

    # Fall back to highest-numbered checkpoint
    pattern = os.path.join(model_dir, "checkpoint_epoch_*.pt")
    matches = sorted(glob.glob(pattern))
    if matches:
        return os.path.basename(matches[-1])

    raise FileNotFoundError(
        f"No checkpoint found in {model_dir}. "
        "Expected best_model.pt, latest.pt, or checkpoint_epoch_*.pt"
    )


def load_model_from_dir(
    model_dir: str,
    checkpoint_filename: str | None = None,
    device: str | None = None,
) -> HookedTransformer:
    """
    Load a trained HookedTransformer from a model directory.

    Reads config.yaml to reconstruct the architecture, then loads weights.

    Args:
        model_dir: Directory containing config.yaml and checkpoint file(s)
        checkpoint_filename: Specific checkpoint to load; auto-detected if None
        device: Device to load onto; auto-detected if None

    Returns:
        HookedTransformer in eval mode on the specified device
    """
    config_path = os.path.join(model_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config.yaml in {model_dir}")

    if checkpoint_filename is None:
        checkpoint_filename = _find_checkpoint(model_dir)

    if device is None:
        device = get_device()

    model = initialize_transformer_from_yaml(config_path)

    checkpoint_path = os.path.join(model_dir, checkpoint_filename)
    checkpoint = t.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    return model


def get_process_from_config(config: dict) -> HMM:
    """
    Instantiate an HMM process from a model's saved config.

    Args:
        config: Full config dict (with 'data_generator.process' key)

    Returns:
        Instantiated HMM process
    """
    return create_process_from_dict(config["data_generator"]["process"])


def extract_residual_stream(
    model: HookedTransformer,
    sequences: t.Tensor,
    device: str | None = None,
    batch_size: int | None = None,
) -> t.Tensor:
    """
    Extract post-layer residual stream activations via TransformerLens cache.

    Args:
        model: HookedTransformer in eval mode
        sequences: (batch, seq_len) integer token sequences
        device: Device for inference; auto-detected if None
        batch_size: If set, process in chunks to reduce peak memory

    Returns:
        (n_layers, batch, seq_len, d_model) tensor on CPU
    """
    if device is None:
        device = get_device()

    model.eval()
    model.to(device)

    n_layers = model.cfg.n_layers

    if batch_size is None or batch_size >= sequences.shape[0]:
        return _extract_batch(model, sequences, device, n_layers)

    # Chunked extraction
    chunks = []
    for start in range(0, sequences.shape[0], batch_size):
        end = min(start + batch_size, sequences.shape[0])
        chunk = _extract_batch(model, sequences[start:end], device, n_layers)
        chunks.append(chunk)

    return t.cat(chunks, dim=1)


def _extract_batch(
    model: HookedTransformer,
    sequences: t.Tensor,
    device: str,
    n_layers: int,
) -> t.Tensor:
    """Extract residual stream for a single batch."""
    sequences = sequences.to(device)

    with t.no_grad():
        _, cache = model.run_with_cache(sequences)

    actv_list = []
    for layer_idx in range(n_layers):
        actv = cache["resid_post", layer_idx].cpu()
        actv_list.append(actv)

    return t.stack(actv_list)


def prepare_probe_data(
    raw_activations: t.Tensor,
    belief_states: np.ndarray,
    use_last_pos: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reshape activations and belief states for probing.

    Args:
        raw_activations: (n_layers, batch, seq_len, d_model) from extract_residual_stream
        belief_states: (batch, seq_len, n_hidden_states) numpy array
        use_last_pos: If True, use only the last sequence position.
                      If False, flatten all positions into samples.

    Returns:
        (layer_activations, targets) where:
            layer_activations: (n_layers, n_samples, d_model) numpy
            targets: (n_samples, n_hidden_states) numpy
    """
    # Convert activations to numpy
    actvs = raw_activations.numpy()  # (n_layers, batch, seq_len, d_model)

    if use_last_pos:
        # Take last position only
        layer_actvs = actvs[:, :, -1, :]          # (n_layers, batch, d_model)
        targets = belief_states[:, -1, :]          # (batch, n_hidden_states)
    else:
        # Flatten batch and seq_len dimensions
        n_layers, batch, seq_len, d_model = actvs.shape
        layer_actvs = actvs.reshape(n_layers, batch * seq_len, d_model)
        targets = belief_states.reshape(batch * seq_len, -1)

    return layer_actvs, targets
