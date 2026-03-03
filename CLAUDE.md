# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase for training small transformers on synthetic data generated from Hidden Markov Models (HMMs). The goal is to study how transformers learn HMM structure. Uses TransformerLens for interpretable transformer models and Weights & Biases for experiment tracking.

## Key Commands

### Data Generation (MPI-parallelized)
```bash
mpirun -n <num_workers> python src/data_generator.py
```
Then consolidate shards into train/test splits by calling `consolidate_and_split()` in `data_generator.py`.

### Training
```bash
# Single run (local or via cluster)
python src/train.py

# Cluster submission (uses addqueue)
bash submit_training.sh

# Sweep (hyperparameter search)
wandb sweep sweep_config.yaml
bash submit_sweep.sh <sweep-id> [num_agents]
```

## Architecture (src/)

### `hmm.py` — HMM Process Definitions
- **`HMM` (abstract base class)**: Defines the HMM interface. Subclasses must implement `emission_matrices` property returning an ndarray of shape `(d_vocab, num_hidden_states, num_hidden_states)` where `E[j, i, k] = P(observe token j AND transition to state k | currently in state i)`.
- Provides: sequence generation, belief state computation (`mixed_state_presentation`), theoretical and empirical entropy rate estimation.
- **Concrete HMMs**: `Z1R` (3 states, 2 tokens), `RRXOR` (5 states, 2 tokens), `Mess3Proc` (3 states, 3 tokens), `PSL7HMM` (loads from .npy file), `CylinderGraphHMM` (parameterized graph-structured HMM with configurable width, depth, and Dirichlet token distributions).

### `utils.py` — Factory and Config Utilities
- `PROCESS_REGISTRY`: Maps string names to HMM classes (`"z1r"`, `"rrxor"`, `"mess3"`, `"psl7"`, `"cylinder_graph"`).
- `create_process_from_dict(config)`: Factory that instantiates HMM from config dict with automatic parameter filtering.
- `initialize_transformer_from_yaml(config_path)`: Creates a `HookedTransformer` from YAML config.

### `train.py` — Training Loop
- Reads `base_config.yaml` for model architecture, optimizer, and training hyperparameters.
- Trains a HookedTransformer on next-token prediction (cross-entropy loss) with cosine LR schedule.
- Supports W&B sweep overrides for hyperparameter search.
- Saves checkpoints (best model + per-epoch) to `models/cylinderhmm/<timestamp>_<run_name>/`.

### `data_generator.py` — MPI Data Generation
- Uses MPI (`mpi4py`) to parallelize long sequence generation across workers.
- Each rank generates a shard saved to `data/datasets/.../shards/`.
- `consolidate_and_split()`: Merges shards into non-overlapping fixed-length sequences, shuffles, and splits into train/test `.pt` files.

## Configuration

All config lives in `base_config.yaml` with three sections:
- `data_generator`: process name + params, num_tokens, save_dir
- `model`: transformer architecture (n_layer, n_embd, n_head, vocab_size, n_ctx, etc.)
- `train`: optimizer params, num_epochs, batch_size, seq_length, logging intervals

**Important constraint**: `vocab_size` must equal `depth * tokens_per_cluster` and `n_ctx` must equal `seq_length`.

## Data Flow

1. Define HMM process in config → 2. Generate long sequence via MPI → 3. Consolidate into train/test `.pt` files → 4. Train transformer on next-token prediction → 5. Analyze via notebooks

## User Notes
- The user is a physicist, not an ML practitioner. Flag non-standard ML practices when they arise.
- When the user says "explain", respond with explanations only—do not write code unless explicitly asked.
- Ask clarifying questions where appropriate.

## Environment

This user works on a Linux cluster via SSH from a Mac. Provide Mac keyboard shortcuts for local actions and Linux/terminal commands for remote actions. Never assume a local desktop GUI environment.

## Python / PyTorch Conventions

When writing Python code that involves PyTorch tensors, always ensure device consistency (CPU vs CUDA) in comparisons, assertions, and operations. Use `.cpu()` or `.to(device)` explicitly when mixing tensor sources.

## LaTeX

When editing LaTeX files, always check that any newly introduced symbols or environments have their required packages (e.g., `amssymb` for `\gtrsim`, `amsmath`, `tikz`). Run a test compilation after edits if possible.

## Cluster / SLURM / Sweeps

When generating SLURM submission scripts or wandb sweep configs, double-check: (1) YAML numeric types are correct (quote strings that look like numbers), (2) variable substitutions like `${args}` are present, (3) environment/module loading commands are included, (4) file paths resolve correctly on the cluster filesystem.

## General Workflow Preferences

- When asked to implement something, implement and run it rather than just planning — don't stop at creating a plan document unless explicitly asked to only plan. If a sweep or job submission is part of the task, actually submit it.
- When searching the codebase for user-referenced code, ask for the exact file path if the first search attempt fails. Don't guess at directory structures.
