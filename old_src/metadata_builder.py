#!/usr/bin/env python3
"""
Metadata Builder for HMM Transformer Training Runs

This script scans a training runs directory (default: records/) and generates a consolidated
metadata.parquet file that tracks all training runs with their configurations, paths, and
performance metrics.

Usage:
    python metadata_builder.py --rebuild              # Full rebuild from scratch
    python metadata_builder.py --update               # Incremental update (default)
    python metadata_builder.py --add-run RUN_ID       # Add/update specific run
    python metadata_builder.py --validate             # Validate existing metadata
    python metadata_builder.py --rebuild --verbose    # Verbose logging
    python metadata_builder.py --records-dir /path/to/records  # Specify custom records directory
"""

import os
import sys
import time
import logging
import argparse
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('metadata_builder.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# Core Extraction Functions
# ============================================================================

def parse_run_id(run_id: str) -> Dict[str, any]:
    """
    Parse run_id to extract timestamp and optional UUID.

    Args:
        run_id: Directory name (e.g., "20251207_232030" or "20251209_002943_d5155ba6")

    Returns:
        dict with 'timestamp', 'uuid', 'is_sweep' fields
    """
    parts = run_id.split('_')

    if len(parts) >= 2:
        timestamp_str = f"{parts[0]}_{parts[1]}"  # YYYYMMDD_HHMMSS
        try:
            timestamp = pd.to_datetime(timestamp_str, format='%Y%m%d_%H%M%S')
        except:
            timestamp = pd.NaT
    else:
        timestamp = pd.NaT

    # UUID is present if we have 3 parts (timestamp + UUID)
    uuid = parts[2] if len(parts) == 3 else None
    is_sweep = uuid is not None

    return {
        'timestamp': timestamp,
        'uuid': uuid,
        'is_sweep': is_sweep
    }


def compute_param_count(model_cfg: dict) -> int:
    """
    Calculate total trainable parameters from model config.

    Formula:
    - Embeddings: vocab_size * n_embd (W_E)
    - Output: vocab_size * n_embd (W_U) - unless tied
    - Position embedding: n_ctx * n_embd
    - Per layer:
      * Attention: 4 * n_embd * n_embd (Q, K, V, O projections)
      * MLP: 2 * n_embd * n_inner (unless attn_only)
      * LayerNorm: 4 * n_embd (2 LN per layer, unless None)
    """
    try:
        vocab_size = model_cfg.get('vocab_size', 3)
        n_embd = model_cfg.get('n_embd', 64)
        n_layer = model_cfg.get('n_layer', 2)
        n_ctx = model_cfg.get('n_ctx', 10)
        n_inner = model_cfg.get('n_inner', 4 * n_embd)
        attn_only = model_cfg.get('attn_only', False)
        tie_word_embeddings = model_cfg.get('tie_word_embeddings', True)
        norm_type = model_cfg.get('normalization_type', 'LN')

        # Embeddings
        embedding_params = vocab_size * n_embd  # W_E
        if not tie_word_embeddings:
            embedding_params += vocab_size * n_embd  # W_U

        # Position embedding
        pos_emb_params = n_ctx * n_embd

        # Per-layer parameters
        attn_params = 4 * n_embd * n_embd  # Q, K, V, O
        mlp_params = 0 if attn_only else (n_embd * n_inner + n_inner * n_embd)
        ln_params = 0 if norm_type is None else (4 * n_embd)  # 2 LN/layer, 2 params each

        per_layer = attn_params + mlp_params + ln_params

        return embedding_params + pos_emb_params + (n_layer * per_layer)
    except Exception as e:
        logger.warning(f"Failed to compute param count: {e}")
        return 0


def determine_status(run_dir: Path) -> str:
    """
    Determine run status based on file presence and modification times.

    Returns one of: "completed", "running", "failed", "incomplete"
    """
    config_exists = (run_dir / "config.yaml").exists()
    metrics_exists = (run_dir / "training_metrics.parquet").exists()
    final_model_exists = len(list(run_dir.glob("final_model_*.pt"))) > 0

    # Check for emergency checkpoints (failures)
    if (run_dir / "checkpoints").exists():
        emergency_ckpts = list((run_dir / "checkpoints").glob("emergency_*.pt"))
        if emergency_ckpts:
            return "failed"

    # Completed: has all key artifacts
    if config_exists and metrics_exists and final_model_exists:
        return "completed"

    # Running: metrics file modified in last 24 hours
    if metrics_exists:
        mtime = (run_dir / "training_metrics.parquet").stat().st_mtime
        if time.time() - mtime < 86400:  # 24 hours
            return "running"

    # Incomplete: has config but missing artifacts
    if config_exists:
        return "incomplete"

    return "failed"


def extract_training_metrics(run_dir: Path) -> Dict[str, any]:
    """Extract final performance metrics from training_metrics.parquet."""
    metrics_file = run_dir / "training_metrics.parquet"

    if not metrics_file.exists():
        return {
            'final_loss': None,
            'final_val_loss': None,
            'min_train_loss': None,
            'min_val_loss': None,
            'total_steps': None,
            'total_passes': None,
            'final_lr': None,
            'training_time_estimate': None
        }

    try:
        df = pd.read_parquet(metrics_file)

        # Get final values
        final_loss = df['loss'].iloc[-1] if 'loss' in df.columns else None
        final_val_loss = df['val_loss'].dropna().iloc[-1] if 'val_loss' in df.columns and df['val_loss'].notna().any() else None

        # Get minimum values
        min_train_loss = df['loss'].min() if 'loss' in df.columns else None
        min_val_loss = df['val_loss'].min() if 'val_loss' in df.columns else None

        # Get step/pass info
        total_steps = int(df['step'].iloc[-1]) if 'step' in df.columns else None
        total_passes = int(df['pass'].iloc[-1]) if 'pass' in df.columns else None
        final_lr = df['learning_rate'].iloc[-1] if 'learning_rate' in df.columns else None

        # Estimate training time (rough estimate based on steps)
        training_time_estimate = None
        if total_steps:
            # Rough estimate: assume 1000 steps/hour
            training_time_estimate = total_steps / 1000.0

        return {
            'final_loss': final_loss,
            'final_val_loss': final_val_loss,
            'min_train_loss': min_train_loss,
            'min_val_loss': min_val_loss,
            'total_steps': total_steps,
            'total_passes': total_passes,
            'final_lr': final_lr,
            'training_time_estimate': training_time_estimate
        }
    except Exception as e:
        logger.warning(f"Failed to extract training metrics from {run_dir.name}: {e}")
        return {
            'final_loss': None,
            'final_val_loss': None,
            'min_train_loss': None,
            'min_val_loss': None,
            'total_steps': None,
            'total_passes': None,
            'final_lr': None,
            'training_time_estimate': None
        }


def extract_validation_metrics(run_dir: Path) -> Dict[str, any]:
    """Extract checkpoint validation metrics from checkpoint_validation_results.parquet."""
    validation_file = run_dir / "checkpoint_validation_results.parquet"

    if not validation_file.exists():
        return {
            'best_mean_r2': None,
            'best_layer_r2': None,
            'final_llr_mean': None,
            'final_llr_std': None,
            'validation_checkpoints': None
        }

    try:
        df = pd.read_parquet(validation_file)

        best_mean_r2 = df['mean_r2'].max() if 'mean_r2' in df.columns else None
        best_layer_r2 = df['best_layer_r2'].max() if 'best_layer_r2' in df.columns else None
        final_llr_mean = df['llr_mean'].iloc[-1] if 'llr_mean' in df.columns else None
        final_llr_std = df['llr_std'].iloc[-1] if 'llr_std' in df.columns else None
        validation_checkpoints = len(df)

        return {
            'best_mean_r2': best_mean_r2,
            'best_layer_r2': best_layer_r2,
            'final_llr_mean': final_llr_mean,
            'final_llr_std': final_llr_std,
            'validation_checkpoints': validation_checkpoints
        }
    except Exception as e:
        logger.warning(f"Failed to extract validation metrics from {run_dir.name}: {e}")
        return {
            'best_mean_r2': None,
            'best_layer_r2': None,
            'final_llr_mean': None,
            'final_llr_std': None,
            'validation_checkpoints': None
        }


def extract_run_metadata(run_dir: Path) -> Dict[str, any]:
    """
    Extract all metadata for a single run directory.

    Returns a dictionary with all 49 fields defined in the schema.
    """
    run_id = run_dir.name
    run_dir_abs = str(run_dir.absolute())

    # Parse run_id
    id_info = parse_run_id(run_id)

    # Check file existence
    config_exists = (run_dir / "config.yaml").exists()
    metrics_exists = (run_dir / "training_metrics.parquet").exists()
    final_model_exists = len(list(run_dir.glob("final_model_*.pt"))) > 0

    # Count checkpoints
    num_checkpoints = 0
    if (run_dir / "checkpoints").exists():
        num_checkpoints = len(list((run_dir / "checkpoints").glob("checkpoint_*.pt")))

    # Get last modified time
    try:
        last_modified = datetime.fromtimestamp(run_dir.stat().st_mtime)
    except:
        last_modified = None

    # Determine status
    status = determine_status(run_dir)

    # Load config
    config = None
    if config_exists:
        try:
            with open(run_dir / "config.yaml", 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load config for {run_id}: {e}")

    # Extract config fields
    if config:
        model_cfg = config.get('model', {})
        train_cfg = config.get('train', {})
        scheduler_cfg = config.get('scheduler', {})
        dataset_cfg = config.get('dataset', {})

        # Model architecture fields
        model_name = model_cfg.get('name')
        vocab_size = model_cfg.get('vocab_size')
        n_ctx = model_cfg.get('n_ctx')
        n_layer = model_cfg.get('n_layer')
        n_embd = model_cfg.get('n_embd')
        n_head = model_cfg.get('n_head')
        d_head = model_cfg.get('d_head')
        n_inner = model_cfg.get('n_inner')
        attn_only = model_cfg.get('attn_only')
        normalization_type = model_cfg.get('normalization_type')
        activation_function = model_cfg.get('activation_function')
        param_count = compute_param_count(model_cfg)

        # Training configuration fields
        process = train_cfg.get('process')
        batch_size = train_cfg.get('batch_size')
        learning_rate = train_cfg.get('learning_rate')
        num_epochs = train_cfg.get('num_epochs')
        device = train_cfg.get('device')
        wandb_project = train_cfg.get('wandb_project_name')
        wandb_run_name = train_cfg.get('wandb_run_name')

        # Other config fields
        scheduler_type = scheduler_cfg.get('type')
        dataset_mode = dataset_cfg.get('mode')
    else:
        # No config - set all to None
        model_name = vocab_size = n_ctx = n_layer = n_embd = None
        n_head = d_head = n_inner = attn_only = normalization_type = None
        activation_function = param_count = None
        process = batch_size = learning_rate = num_epochs = device = None
        wandb_project = wandb_run_name = None
        scheduler_type = dataset_mode = None

    # Extract performance metrics
    training_metrics = extract_training_metrics(run_dir)
    validation_metrics = extract_validation_metrics(run_dir)

    # Build complete metadata dict
    metadata = {
        # Run identification (7)
        'run_id': run_id,
        'timestamp': id_info['timestamp'],
        'uuid': id_info['uuid'],
        'run_dir': run_dir_abs,
        'wandb_project': wandb_project,
        'wandb_run_name': wandb_run_name,
        'is_sweep': id_info['is_sweep'],

        # Run status (6)
        'status': status,
        'has_config': config_exists,
        'has_metrics': metrics_exists,
        'has_final_model': final_model_exists,
        'num_checkpoints': num_checkpoints,
        'last_modified': last_modified,

        # Model architecture (12)
        'model_name': model_name,
        'vocab_size': vocab_size,
        'n_ctx': n_ctx,
        'n_layer': n_layer,
        'n_embd': n_embd,
        'n_head': n_head,
        'd_head': d_head,
        'n_inner': n_inner,
        'attn_only': attn_only,
        'normalization_type': normalization_type,
        'activation_function': activation_function,
        'param_count': param_count,

        # Training configuration (7)
        'process': process,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'device': device,
        'scheduler_type': scheduler_type,
        'dataset_mode': dataset_mode,

        # Performance metrics (13)
        **training_metrics,
        **validation_metrics,

        # Sweep information (4) - will be computed in post-processing
        'sweep_id': None,
        'sweep_timestamp': id_info['timestamp'] if id_info['is_sweep'] else None,
        'sweep_run_count': None,
        'sweep_rank': None
    }

    return metadata


# ============================================================================
# Build Functions
# ============================================================================

def scan_records_directory(records_dir: Path = None) -> List[Path]:
    """Scan records/ directory and return list of run directories."""
    if records_dir is None:
        records_dir = Path(__file__).parent / "records"

    if not records_dir.exists():
        logger.error(f"Records directory not found: {records_dir}")
        return []

    # Get all subdirectories
    run_dirs = [d for d in records_dir.iterdir() if d.is_dir()]

    logger.info(f"Found {len(run_dirs)} run directories in {records_dir}")
    return run_dirs


def compute_sweep_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute sweep-level statistics and rankings.

    For each sweep (grouped by sweep_timestamp), compute:
    - sweep_run_count: Number of runs in sweep
    - sweep_rank: Rank within sweep by min_val_loss (1 = best)
    """
    if len(df) == 0:
        return df

    # Only process sweep runs
    sweep_df = df[df['is_sweep']].copy()

    if len(sweep_df) == 0:
        return df

    # Group by sweep_timestamp
    for sweep_ts, group in sweep_df.groupby('sweep_timestamp'):
        run_count = len(group)

        # Rank by min_val_loss (lower is better)
        # Runs without val_loss get rank = None
        has_val_loss = group['min_val_loss'].notna()

        if has_val_loss.any():
            ranks = group.loc[has_val_loss, 'min_val_loss'].rank(method='min')

            # Update DataFrame
            for run_id in group.index:
                df.loc[run_id, 'sweep_run_count'] = run_count
                if run_id in ranks.index:
                    df.loc[run_id, 'sweep_rank'] = int(ranks[run_id])
        else:
            # No val_loss available - just set run count
            for run_id in group.index:
                df.loc[run_id, 'sweep_run_count'] = run_count

    return df


def build_metadata_full(records_dir: Path = None, parallel: int = 8) -> pd.DataFrame:
    """
    Build metadata from scratch by scanning all runs.

    Args:
        records_dir: Path to records directory
        parallel: Number of parallel workers (0 = sequential)

    Returns:
        DataFrame with metadata for all runs
    """
    logger.info("Starting full metadata rebuild...")

    # Scan for all runs
    run_dirs = scan_records_directory(records_dir)

    if len(run_dirs) == 0:
        logger.warning("No run directories found")
        return pd.DataFrame()

    # Extract metadata
    if parallel > 1:
        logger.info(f"Extracting metadata using {parallel} workers...")
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {executor.submit(extract_run_metadata, run_dir): run_dir
                      for run_dir in run_dirs}

            metadata_list = []
            for future in as_completed(futures):
                try:
                    metadata = future.result()
                    metadata_list.append(metadata)
                except Exception as e:
                    run_dir = futures[future]
                    logger.error(f"Failed to extract metadata for {run_dir.name}: {e}")
    else:
        logger.info("Extracting metadata sequentially...")
        metadata_list = []
        for run_dir in run_dirs:
            try:
                metadata = extract_run_metadata(run_dir)
                metadata_list.append(metadata)
            except Exception as e:
                logger.error(f"Failed to extract metadata for {run_dir.name}: {e}")

    # Create DataFrame
    df = pd.DataFrame(metadata_list)

    # Sort by timestamp
    df = df.sort_values('timestamp', ascending=True).reset_index(drop=True)

    # Compute sweep ranks
    df = compute_sweep_ranks(df)

    logger.info(f"Successfully built metadata for {len(df)} runs")

    return df


def build_metadata_incremental(existing_df: pd.DataFrame, records_dir: Path = None) -> pd.DataFrame:
    """
    Incrementally update metadata by processing only new/changed runs.

    Args:
        existing_df: Existing metadata DataFrame
        records_dir: Path to records directory

    Returns:
        Updated DataFrame
    """
    logger.info("Starting incremental metadata update...")

    # Scan for all runs
    all_runs = scan_records_directory(records_dir)
    all_run_ids = {r.name for r in all_runs}

    # Find new runs (not in existing metadata)
    existing_run_ids = set(existing_df['run_id'])
    new_run_ids = all_run_ids - existing_run_ids

    logger.info(f"Found {len(new_run_ids)} new runs")

    # Find previously running runs (may have completed)
    running_run_ids = set(existing_df[existing_df['status'] == 'running']['run_id'])

    logger.info(f"Re-scanning {len(running_run_ids)} previously running runs")

    # Combine runs to process
    runs_to_process = new_run_ids | running_run_ids

    if len(runs_to_process) == 0:
        logger.info("No updates needed")
        return existing_df

    # Extract metadata for updated runs
    logger.info(f"Processing {len(runs_to_process)} runs...")

    run_dir_map = {r.name: r for r in all_runs}
    metadata_list = []

    for run_id in runs_to_process:
        if run_id not in run_dir_map:
            logger.warning(f"Run {run_id} not found in records directory")
            continue

        try:
            metadata = extract_run_metadata(run_dir_map[run_id])
            metadata_list.append(metadata)
        except Exception as e:
            logger.error(f"Failed to extract metadata for {run_id}: {e}")

    # Remove old entries for re-scanned runs
    df_filtered = existing_df[~existing_df['run_id'].isin(runs_to_process)]

    # Append new data
    if metadata_list:
        df_new = pd.DataFrame(metadata_list)
        df_combined = pd.concat([df_filtered, df_new], ignore_index=True)
    else:
        df_combined = df_filtered

    # Sort by timestamp
    df_combined = df_combined.sort_values('timestamp', ascending=True).reset_index(drop=True)

    # Recompute sweep ranks
    df_combined = compute_sweep_ranks(df_combined)

    logger.info(f"Successfully updated metadata ({len(metadata_list)} new/updated)")

    return df_combined


# ============================================================================
# I/O Functions
# ============================================================================

def save_metadata(df: pd.DataFrame, path: Path):
    """Save metadata with atomic write (temp file + rename)."""
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file
    temp_path = path.with_suffix('.parquet.tmp')
    df.to_parquet(temp_path, index=False)

    # Atomic rename (overwrites existing)
    temp_path.replace(path)

    logger.info(f"Saved metadata to {path} ({len(df)} runs)")


def load_metadata(path: Path = None) -> pd.DataFrame:
    """Load metadata with default path handling."""
    if path is None:
        path = Path(__file__).parent / "records" / "metadata.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"Metadata not found: {path}\n"
            f"Run: python metadata_builder.py --rebuild"
        )

    df = pd.read_parquet(path)
    logger.info(f"Loaded metadata from {path} ({len(df)} runs)")

    return df


def validate_metadata(df: pd.DataFrame) -> bool:
    """
    Validate metadata integrity.

    Returns True if valid, False with logged warnings otherwise.
    """
    logger.info("Validating metadata...")

    valid = True

    # Check schema - all 49 columns present
    expected_columns = [
        # Run identification (7)
        'run_id', 'timestamp', 'uuid', 'run_dir', 'wandb_project', 'wandb_run_name', 'is_sweep',
        # Run status (6)
        'status', 'has_config', 'has_metrics', 'has_final_model', 'num_checkpoints', 'last_modified',
        # Model architecture (12)
        'model_name', 'vocab_size', 'n_ctx', 'n_layer', 'n_embd', 'n_head', 'd_head', 'n_inner',
        'attn_only', 'normalization_type', 'activation_function', 'param_count',
        # Training configuration (7)
        'process', 'batch_size', 'learning_rate', 'num_epochs', 'device', 'scheduler_type', 'dataset_mode',
        # Performance metrics (13)
        'final_loss', 'final_val_loss', 'min_train_loss', 'min_val_loss', 'total_steps',
        'total_passes', 'final_lr', 'training_time_estimate', 'best_mean_r2', 'best_layer_r2',
        'final_llr_mean', 'final_llr_std', 'validation_checkpoints',
        # Sweep information (4)
        'sweep_id', 'sweep_timestamp', 'sweep_run_count', 'sweep_rank'
    ]

    missing_cols = set(expected_columns) - set(df.columns)
    if missing_cols:
        logger.error(f"Missing columns: {missing_cols}")
        valid = False

    # Check uniqueness of run_id
    if df['run_id'].duplicated().any():
        logger.error("Duplicate run_ids found")
        valid = False

    # Check path existence (sample check on completed runs)
    completed = df[df['status'] == 'completed']
    if len(completed) > 0:
        sample = completed.sample(min(10, len(completed)))
        for _, row in sample.iterrows():
            if not Path(row['run_dir']).exists():
                logger.warning(f"Run directory not found: {row['run_dir']}")
                valid = False

    # Logical constraint: completed → has_final_model
    invalid = df[(df['status'] == 'completed') & (~df['has_final_model'])]
    if len(invalid) > 0:
        logger.warning(f"{len(invalid)} completed runs missing final_model")
        valid = False

    # Check param_count > 0 for runs with config
    invalid = df[(df['has_config']) & ((df['param_count'].isna()) | (df['param_count'] <= 0))]
    if len(invalid) > 0:
        logger.warning(f"{len(invalid)} runs with config have invalid param_count")
        valid = False

    if valid:
        logger.info("Metadata validation passed ✓")
    else:
        logger.warning("Metadata validation failed - see warnings above")

    return valid


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Build and manage metadata for HMM transformer training runs'
    )

    parser.add_argument(
        '--rebuild',
        action='store_true',
        help='Full rebuild from scratch'
    )

    parser.add_argument(
        '--update',
        action='store_true',
        help='Incremental update (default)'
    )

    parser.add_argument(
        '--add-run',
        type=str,
        metavar='RUN_ID',
        help='Add/update specific run by ID'
    )

    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate existing metadata only'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Custom output path (default: records/metadata.parquet)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose logging'
    )

    parser.add_argument(
        '--parallel',
        type=int,
        default=8,
        help='Number of parallel workers for full rebuild (default: 8)'
    )

    parser.add_argument(
        '--records-dir',
        type=Path,
        default=None,
        help='Path to records directory (default: ./records)'
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Default records directory
    if args.records_dir is None:
        args.records_dir = Path(__file__).parent / "records"

    # Default output path
    if args.output is None:
        args.output = args.records_dir / "metadata.parquet"

    # Validate only
    if args.validate:
        try:
            df = load_metadata(args.output)
            validate_metadata(df)
        except FileNotFoundError as e:
            logger.error(str(e))
            sys.exit(1)
        return

    # Add specific run
    if args.add_run:
        run_id = args.add_run
        run_dir = args.records_dir / run_id

        if not run_dir.exists():
            logger.error(f"Run directory not found: {run_dir}")
            sys.exit(1)

        # Load existing or create empty
        try:
            df = load_metadata(args.output)
        except FileNotFoundError:
            df = pd.DataFrame()

        # Remove old entry if exists
        df = df[df['run_id'] != run_id]

        # Extract new metadata
        logger.info(f"Extracting metadata for run: {run_id}")
        metadata = extract_run_metadata(run_dir)

        # Append
        df = pd.concat([df, pd.DataFrame([metadata])], ignore_index=True)
        df = df.sort_values('timestamp', ascending=True).reset_index(drop=True)
        df = compute_sweep_ranks(df)

        # Save
        save_metadata(df, args.output)
        logger.info(f"Added/updated run: {run_id}")
        return

    # Full rebuild
    if args.rebuild:
        df = build_metadata_full(records_dir=args.records_dir, parallel=args.parallel)
        save_metadata(df, args.output)
        validate_metadata(df)
        return

    # Incremental update (default)
    try:
        existing_df = load_metadata(args.output)
        df = build_metadata_incremental(existing_df, records_dir=args.records_dir)
        save_metadata(df, args.output)
        validate_metadata(df)
    except FileNotFoundError:
        logger.warning("No existing metadata found - performing full rebuild")
        df = build_metadata_full(records_dir=args.records_dir, parallel=args.parallel)
        save_metadata(df, args.output)
        validate_metadata(df)


if __name__ == "__main__":
    main()
