"""
Checkpoint validation module for transformer models trained on HMM tasks.

This module provides the CheckpointValidator class for evaluating model checkpoints
with both belief state geometry analysis and log-likelihood ratio testing.
"""

import torch as t
import numpy as np
import pandas as pd
import yaml
import wandb
from pathlib import Path
from typing import Dict, Optional, Tuple
from tqdm import tqdm
from hmm import HMM, RRXOR, Z1R, Mess3Proc, PSL7HMM
from model import HookedTransformerModel, MaskedHeadTransformerModel
from pca import extract_residual_stream, pca_layer_wise_actvs
from hmm_proc_kldiv import (
    compute_log_likelihood_forward_conditional,
    compute_transformer_log_likelihood
)


def get_device() -> str:
    """Get the best available device in order: cuda -> mps -> cpu"""
    if t.cuda.is_available():
        return "cuda"
    elif t.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class CheckpointValidator:
    """
    Validates model checkpoints with geometry and likelihood tests.

    Key features:
    - Caches test dataset once at initialization (1000 sequences by default)
    - Efficient GPU-based validation (< 10 seconds per checkpoint)
    - Supports both HookedTransformer and MaskedHeadTransformer
    """

    def __init__(
        self,
        hmm_process: HMM,
        test_size: int = 1000,
        seq_length: Optional[int] = None,
        device: str = 'auto',
        seed: int = 42
    ):
        """
        Initialize validator with cached test data.

        Args:
            hmm_process: HMM instance (e.g., Mess3Proc())
            test_size: Number of test sequences (default: 1000)
            seq_length: Sequence length (if None, will be inferred from model later)
            device: Device for computation ('auto', 'cuda', 'mps', 'cpu')
            seed: Random seed for reproducible test set
        """
        self.hmm_process = hmm_process
        self.test_size = test_size
        self.seq_length = seq_length
        self.seed = seed

        # Device selection
        if device == 'auto':
            self.device = get_device()
        else:
            self.device = device

        print(f"CheckpointValidator: Using device {self.device}")

        # Generate and cache test dataset
        self._generate_test_dataset()

    def _generate_test_dataset(self):
        """Generate and cache test observations and belief states."""
        print(f"Generating test dataset ({self.test_size} sequences)...")

        # Set seed for reproducibility
        if self.seed is not None:
            t.manual_seed(self.seed)
            if t.cuda.is_available():
                t.cuda.manual_seed(self.seed)

        # Generate observations
        # If seq_length not provided, we'll need to get it from model later
        # For now, generate a reasonable default or wait until first validation
        if self.seq_length is None:
            # Default to a reasonable length; will regenerate if needed
            self.seq_length = 100

        self.test_observations = self.hmm_process.generate_data(
            batch_size=self.test_size,
            length=self.seq_length,
            use_tqdm=False
        ).to(self.device)

        # Compute ground truth belief states
        test_obs_np = self.test_observations.cpu().numpy()
        belief_states_np = self.hmm_process.mixed_state_presentation(test_obs_np)
        self.test_belief_states = t.tensor(
            belief_states_np,
            dtype=t.float32,
            device=self.device
        )

        print(f"Test dataset cached: {self.test_observations.shape}")
        print(f"Belief states shape: {self.test_belief_states.shape}")

    def _ensure_seq_length(self, required_length: int):
        """Regenerate test dataset if required length doesn't match."""
        if self.seq_length != required_length:
            print(f"Regenerating test dataset for seq_length={required_length}")
            self.seq_length = required_length
            self._generate_test_dataset()

    def validate_geometry(
        self,
        model,
        use_last_pos: bool = True,
        num_samples: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Run belief state geometry validation (PCA/linear regression).

        Args:
            model: Transformer model to validate
            use_last_pos: Whether to use only last position for analysis
            num_samples: Number of samples to use (None = all test samples)

        Returns:
            Dictionary with:
                - r2_scores: List[float] - R² per layer
                - mses: List[float] - MSE per layer
                - baseline_mses: List[float] - Baseline MSE per layer
                - mean_r2: float - Average R² across layers
                - best_layer_r2: float - Best layer R²
                - best_layer_idx: int - Index of best layer
        """
        # Determine number of samples to use
        if num_samples is None:
            num_samples = self.test_size
        else:
            num_samples = min(num_samples, self.test_size)

        # Extract test subset
        test_obs_subset = self.test_observations[:num_samples]
        test_beliefs_subset = self.test_belief_states[:num_samples]

        # Get model's context window size
        if hasattr(model, 'model') and hasattr(model.model, 'cfg'):
            n_ctx = model.model.cfg.n_ctx
        else:
            n_ctx = model.cfg.n_ctx

        # Truncate sequences to model's context window if needed
        if test_obs_subset.shape[1] > n_ctx:
            print(f"Truncating sequences from {test_obs_subset.shape[1]} to {n_ctx} (model's n_ctx)")
            test_obs_subset = test_obs_subset[:, :n_ctx]
            test_beliefs_subset = test_beliefs_subset[:, :n_ctx, :]

        print(f"Running geometry validation on {num_samples} samples...")

        # Extract residual stream activations
        with t.no_grad():
            activations = extract_residual_stream(
                model=model,
                sequences=test_obs_subset,
                device=self.device
            )

        # Run layer-wise PCA analysis
        regressors, predictions, mses, r2_scores, baseline_mses = pca_layer_wise_actvs(
            raw_actvs=activations,
            belief_states=test_beliefs_subset,
            use_last_pos=use_last_pos,
            save_dir="",  # Don't save
            run_id="",  # Don't save
            save_data=False  # Critical: don't save to disk during training
        )

        # Compute aggregate statistics
        mean_r2 = float(np.mean(r2_scores))
        best_layer_idx = int(np.argmax(r2_scores))
        best_layer_r2 = float(r2_scores[best_layer_idx])

        return {
            'r2_scores': r2_scores,
            'mses': mses,
            'baseline_mses': baseline_mses,
            'mean_r2': mean_r2,
            'best_layer_r2': best_layer_r2,
            'best_layer_idx': best_layer_idx
        }

    def validate_llr(
        self,
        model,
        offset: Optional[int] = None,
        num_samples: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Run log-likelihood ratio validation.

        Args:
            model: Transformer model to validate
            offset: Offset for conditional likelihood computation (default: n_ctx - 1)
            num_samples: Number of samples to use (None = all test samples)

        Returns:
            Dictionary with:
                - llr_mean: float - Mean log-likelihood ratio
                - llr_std: float - Std of LLR
                - llr_min: float - Minimum LLR
                - llr_max: float - Maximum LLR
                - true_ll_mean: float - Mean true HMM log-likelihood
                - model_ll_mean: float - Mean model log-likelihood
        """
        # Get model's context window size
        if hasattr(model, 'model') and hasattr(model.model, 'cfg'):
            n_ctx = model.model.cfg.n_ctx
        else:
            n_ctx = model.cfg.n_ctx

        # Set default offset to n_ctx - 1 if not provided
        if offset is None:
            offset = n_ctx - 1
            print(f"Using default offset = {offset} (n_ctx - 1)")

        # Determine number of samples to use
        if num_samples is None:
            num_samples = self.test_size
        else:
            num_samples = min(num_samples, self.test_size)

        # Extract test subset
        test_obs_subset = self.test_observations[:num_samples]
        test_obs_np = test_obs_subset.cpu().numpy()

        print(f"Running LLR validation on {num_samples} samples...")

        # Compute true HMM log-likelihood
        true_ll = compute_log_likelihood_forward_conditional(
            hmm=self.hmm_process,
            observations=test_obs_np,
            offset=offset,
            reduction=None,  # Return per-sequence values
            use_tqdm=False
        )

        # Compute model log-likelihood
        with t.no_grad():
            model_ll = compute_transformer_log_likelihood(
                model=model,
                sequences=test_obs_subset,
                reduction=None  # Return per-sequence values
            )

        # Compute log-likelihood ratio
        llr = true_ll - model_ll

        # Compute statistics
        return {
            'llr_mean': float(np.mean(llr)),
            'llr_std': float(np.std(llr)),
            'llr_min': float(np.min(llr)),
            'llr_max': float(np.max(llr)),
            'true_ll_mean': float(np.mean(true_ll)),
            'model_ll_mean': float(np.mean(model_ll))
        }

    def validate_all(
        self,
        model,
        step: int,
        num_geometry_samples: int = 1000,
        num_llr_samples: int = 1000
    ) -> Dict[str, float]:
        """
        Run all validations and return combined results.

        Args:
            model: Transformer model to validate
            step: Training step number (for logging)
            num_geometry_samples: Samples for geometry analysis
            num_llr_samples: Samples for LLR testing

        Returns:
            Combined dictionary with all validation metrics
        """
        print(f"\n{'='*60}")
        print(f"Checkpoint Validation at Step {step}")
        print(f"{'='*60}")

        # Run geometry validation
        geometry_metrics = self.validate_geometry(
            model=model,
            num_samples=num_geometry_samples
        )

        # Run LLR validation
        llr_metrics = self.validate_llr(
            model=model,
            num_samples=num_llr_samples
        )

        # Merge results
        all_metrics = {
            **geometry_metrics,
            **llr_metrics
        }

        print(f"{'='*60}")

        return all_metrics

    def log_to_wandb(
        self,
        metrics: Dict,
        step: int,
        prefix: str = 'validation'
    ):
        """
        Log validation metrics to WandB with appropriate structure.

        Args:
            metrics: Dictionary from validate_all()
            step: Training step number
            prefix: Metric prefix (default: 'validation')
        """
        wandb_metrics = {}

        # Log per-layer geometry metrics
        if 'r2_scores' in metrics:
            for i, r2 in enumerate(metrics['r2_scores']):
                wandb_metrics[f"{prefix}/geometry/r2_layer_{i}"] = float(r2)

        # Log aggregate geometry metrics
        for key in ['mean_r2', 'best_layer_r2', 'best_layer_idx']:
            if key in metrics:
                wandb_metrics[f"{prefix}/geometry/{key}"] = metrics[key]

        # Log LLR metrics
        for key in ['llr_mean', 'llr_std', 'llr_min', 'llr_max',
                    'true_ll_mean', 'model_ll_mean']:
            if key in metrics:
                wandb_metrics[f"{prefix}/{key}"] = metrics[key]

        # Log to WandB
        wandb.log(wandb_metrics, step=step)


def load_model_from_checkpoint(config_path: str, checkpoint_path: Path,
                                is_final_model: bool, device: str):
    """
    Load model from checkpoint file, handling both model types and file formats.

    Args:
        config_path: Path to config.yaml (as string)
        checkpoint_path: Path to checkpoint .pt file
        is_final_model: True if final_model_*.pt, False if checkpoint_step_*.pt
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    # Load config to determine model type
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_name = config['model']['name']

    # Initialize appropriate model class
    if model_name == "HookedTransformerModel":
        model = HookedTransformerModel(str(config_path))
    elif model_name == "MaskedHeadTransformerModel":
        model = MaskedHeadTransformerModel(str(config_path))
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    # Load checkpoint
    checkpoint = t.load(checkpoint_path, map_location=device)

    # Extract state_dict (final model = direct dict, checkpoint = wrapped)
    state_dict = checkpoint if is_final_model else checkpoint['model_state_dict']

    # Load weights
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model


def validate_model_dir(
    model_dir: str,
    validate_all_checkpoints: bool = False,
    num_geometry_samples: int = 1000,
    num_llr_samples: int = 1000,
    test_size: int = 1000,
    device: str = 'auto',
    seed: int = 42
) -> Tuple[pd.DataFrame, str]:
    """
    Validate transformer model checkpoints and save results for metadata tracking.

    This function validates either just the final model (default) or all checkpoints
    in a training run directory. Results are saved in a format compatible with
    metadata_builder.py for automatic extraction of validation metrics.

    Args:
        model_dir: Path to training run directory (e.g., 'records/20251207_232030')
        validate_all_checkpoints: If True, validate all checkpoints + final model.
                                   If False (default), only validate final model.
        num_geometry_samples: Number of samples for PCA/geometry validation (default: 1000)
        num_llr_samples: Number of samples for log-likelihood ratio testing (default: 1000)
        test_size: Size of cached test dataset in validator (default: 1000)
        device: Computation device ('auto', 'cuda', 'mps', 'cpu')
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Tuple of (DataFrame, output_path):
        - DataFrame with columns: step, mean_r2, best_layer_r2, best_layer_idx,
          llr_mean, llr_std, llr_min, llr_max, true_ll_mean, model_ll_mean
        - String path to saved checkpoint_validation_results.parquet file

    Raises:
        FileNotFoundError: If model_dir or config.yaml not found
        ValueError: If config is invalid, unknown process/model type, or no models to validate

    Example:
        # Validate final model only (fast)
        df, path = validate_model_dir("records/20251207_232030")

        # Validate all checkpoints (comprehensive)
        df, path = validate_model_dir("records/20251207_232030", validate_all_checkpoints=True)
    """
    # ========================================================================
    # 1. Input Validation
    # ========================================================================
    model_dir = Path(model_dir)
    config_path = model_dir / "config.yaml"

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate config structure
    required_keys = ['model', 'train']
    missing_keys = [k for k in required_keys if k not in config]
    if missing_keys:
        raise ValueError(f"Config missing required keys: {missing_keys}")

    print(f"Loading config from {config_path}")

    # ========================================================================
    # 2. Initialize HMM Process
    # ========================================================================
    process_name = config['train']['process']

    if process_name == "rrxor":
        hmm_process = RRXOR()
    elif process_name == "z1r":
        hmm_process = Z1R()
    elif process_name == "mess3":
        hmm_process = Mess3Proc()
    elif process_name == "psl7":
        hmm_process = PSL7HMM()
    else:
        raise ValueError(
            f"Unknown process: {process_name}. "
            f"Supported processes: rrxor, z1r, mess3, psl7"
        )

    print(f"Initialized HMM process: {process_name}")

    # ========================================================================
    # 3. Determine Sequence Length
    # ========================================================================
    n_ctx = config['model']['n_ctx']
    # LLR validation requires seq_length > n_ctx for proper conditional likelihood
    # Use 10x n_ctx as reasonable default (matches notebook: 100 for n_ctx=10)
    seq_length = max(100, 10 * n_ctx)
    print(f"Using seq_length={seq_length} for validation (n_ctx={n_ctx})")

    # ========================================================================
    # 4. Initialize CheckpointValidator
    # ========================================================================
    validator = CheckpointValidator(
        hmm_process=hmm_process,
        test_size=test_size,
        seq_length=seq_length,
        device=device,
        seed=seed
    )

    # ========================================================================
    # 5. Discover Checkpoints
    # ========================================================================
    checkpoint_dir = model_dir / "checkpoints"

    if not checkpoint_dir.exists():
        checkpoint_files = []
        checkpoint_steps = []
        print(f"Warning: No checkpoints directory found at {checkpoint_dir}")
    else:
        # Find all checkpoint files (sorted for consistent ordering)
        checkpoint_files = sorted(checkpoint_dir.glob("checkpoint_step_*.pt"))

        # Extract step numbers from filenames
        checkpoint_steps = []
        for f in checkpoint_files:
            # Filename format: checkpoint_step_12345.pt
            step_str = f.stem.split('_')[-1]
            try:
                step = int(step_str)
                checkpoint_steps.append(step)
            except ValueError:
                print(f"Warning: Skipping malformed checkpoint filename: {f.name}")
                checkpoint_files = [cf for cf in checkpoint_files if cf != f]

        print(f"Found {len(checkpoint_files)} checkpoints")
        if len(checkpoint_files) > 0:
            print(f"Step range: {min(checkpoint_steps)} - {max(checkpoint_steps)}")

    # ========================================================================
    # 6. Discover and Handle Final Model
    # ========================================================================
    # Find final model file (pattern: final_model_{process_name}.pt)
    final_model_files = list(model_dir.glob("final_model_*.pt"))

    if len(final_model_files) == 0:
        print("Warning: No final model found")
        final_model_file = None
        final_model_step = None
    elif len(final_model_files) > 1:
        print(f"Warning: Multiple final models found: {[f.name for f in final_model_files]}")
        print(f"Using first match: {final_model_files[0].name}")
        final_model_file = final_model_files[0]
    else:
        final_model_file = final_model_files[0]

    # Determine step number for final model
    if final_model_file is not None:
        metrics_file = model_dir / "training_metrics.parquet"
        if metrics_file.exists():
            try:
                df_metrics = pd.read_parquet(metrics_file)
                final_model_step = int(df_metrics['step'].iloc[-1])
                print(f"Final model: {final_model_file.name} (step={final_model_step} from metrics)")
            except Exception as e:
                print(f"Warning: Could not read final step from metrics: {e}")
                # Fallback to max checkpoint step + 1
                if checkpoint_steps:
                    final_model_step = max(checkpoint_steps) + 1
                    print(f"Using fallback step: {final_model_step}")
                else:
                    final_model_step = -1
                    print(f"Using sentinel step: {final_model_step}")
        else:
            # No metrics file - use fallback
            if checkpoint_steps:
                final_model_step = max(checkpoint_steps) + 1
            else:
                final_model_step = -1
            print(f"No training metrics found, using step={final_model_step} for final model")

    # ========================================================================
    # 7. Determine Validation Targets
    # ========================================================================
    results = []

    if validate_all_checkpoints:
        # Validate all checkpoints + final model
        files_to_validate = list(zip(checkpoint_files, checkpoint_steps, [False] * len(checkpoint_files)))
        if final_model_file is not None:
            files_to_validate.append((final_model_file, final_model_step, True))
        print(f"\nValidating {len(files_to_validate)} checkpoints (including final model)...")
    else:
        # Validate only final model
        if final_model_file is None:
            raise ValueError(
                "No final model found and validate_all_checkpoints=False. "
                "Either set validate_all_checkpoints=True or ensure final model exists."
            )
        files_to_validate = [(final_model_file, final_model_step, True)]
        print(f"\nValidating final model only...")

    # ========================================================================
    # 8. Validation Loop with Progress Reporting
    # ========================================================================
    for checkpoint_file, step, is_final in tqdm(files_to_validate, desc="Validating checkpoints"):
        try:
            # Load model
            model = load_model_from_checkpoint(
                config_path=str(config_path),
                checkpoint_path=checkpoint_file,
                is_final_model=is_final,
                device=validator.device
            )

            # Run validation (both geometry and LLR)
            metrics = validator.validate_all(
                model=model,
                step=step,
                num_geometry_samples=num_geometry_samples,
                num_llr_samples=num_llr_samples
            )

            # Extract scalar metrics only (matches notebook cell 12)
            result = {
                'step': step,
                'mean_r2': metrics['mean_r2'],
                'best_layer_r2': metrics['best_layer_r2'],
                'best_layer_idx': metrics['best_layer_idx'],
                'llr_mean': metrics['llr_mean'],
                'llr_std': metrics['llr_std'],
                'llr_min': metrics['llr_min'],
                'llr_max': metrics['llr_max'],
                'true_ll_mean': metrics['true_ll_mean'],
                'model_ll_mean': metrics['model_ll_mean']
            }
            results.append(result)

            # Memory cleanup (prevent accumulation)
            del model
            del metrics
            if t.cuda.is_available():
                t.cuda.empty_cache()

        except Exception as e:
            print(f"\nError validating checkpoint at step {step}: {e}")
            print(f"Skipping this checkpoint and continuing...")
            continue

    # Check if any checkpoints succeeded
    if len(results) == 0:
        raise ValueError("No checkpoints were successfully validated")

    # ========================================================================
    # 9. Create DataFrame
    # ========================================================================
    df = pd.DataFrame(results)

    # Sort by step (chronological order)
    df = df.sort_values('step').reset_index(drop=True)

    # Print summary
    print(f"\nValidation complete! Processed {len(df)} checkpoints")
    print(f"\nSummary statistics:")
    print(df[['step', 'mean_r2', 'best_layer_r2', 'llr_mean']].describe())

    # ========================================================================
    # 10. Save and Return
    # ========================================================================
    output_path = model_dir / "checkpoint_validation_results.parquet"
    df.to_parquet(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    return df, str(output_path)
