from hmm import RRXOR, Z1R, Mess3Proc, PSL7HMM
from model import HookedTransformerModel, MaskedHeadTransformerModel
from checkpoint_validator import CheckpointValidator
import yaml
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler
import os
import wandb
from dotenv import load_dotenv
from contextlib import contextmanager
import pandas as pd
import shutil
from datetime import datetime
import uuid
import math

@contextmanager
def checkpoint_on_error(model, optimizer, config, data_dir, scheduler=None, save_metrics=True):
    """Context manager that saves emergency checkpoint on any exception"""
    epoch = 0
    loss_value = None
    
    class CheckpointState:
        def __init__(self):
            self.epoch = 0
            self.loss = None
            self.metrics_data = []
            self.data_dir = data_dir
        
        def update(self, epoch, loss):
            self.epoch = epoch
            self.loss = loss
        
        def log_metrics(self, metrics):
            self.metrics_data.append(metrics)
        
        def save_metrics_to_file(self):
            if save_metrics and self.metrics_data:
                df = pd.DataFrame(self.metrics_data)
                # df.to_csv("training_metrics.csv", index=False)
                df.to_parquet(os.path.join(self.data_dir, "training_metrics.parquet"), index=False)
                print(f"Training metrics saved to training_metrics.parquet")
    
    state = CheckpointState()
    
    try:
        yield state
    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA Out of Memory at epoch {state.epoch}: {e}")
        print("Try reducing batch_size in config or using a smaller model")
        checkpoint_dir = os.path.join(data_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_data = {
            'epoch': state.epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': state.loss,
            'config': config,
            'error': f"CUDA OOM: {str(e)}"
        }
        if scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
        torch.save(checkpoint_data, f"{checkpoint_dir}/emergency_checkpoint_epoch_{state.epoch}_cuda_oom.pt")
        print(f"Emergency checkpoint saved due to CUDA OOM at epoch {state.epoch}")
        state.save_metrics_to_file()
        raise
    except RuntimeError as e:
        # Check if it's an MPS OOM error
        if "MPS" in str(e) and "memory" in str(e).lower():
            print(f"MPS Out of Memory at epoch {state.epoch}: {e}")
            print("Try reducing batch_size in config or using a smaller model")
            checkpoint_dir = os.path.join(data_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_data = {
                'epoch': state.epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': state.loss,
                'config': config,
                'error': f"MPS OOM: {str(e)}"
            }
            if scheduler is not None:
                checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
            torch.save(checkpoint_data, f"{checkpoint_dir}/emergency_checkpoint_epoch_{state.epoch}_mps_oom.pt")
            print(f"Emergency checkpoint saved due to MPS OOM at epoch {state.epoch}")
            state.save_metrics_to_file()
            raise
        print(f"Runtime error at epoch {state.epoch}: {e}")
        checkpoint_dir = os.path.join(data_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_data = {
            'epoch': state.epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': state.loss,
            'config': config,
            'error': f"Runtime error: {str(e)}"
        }
        if scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
        torch.save(checkpoint_data, f"{checkpoint_dir}/emergency_checkpoint_epoch_{state.epoch}_runtime_error.pt")
        print(f"Emergency checkpoint saved due to runtime error at epoch {state.epoch}")
        state.save_metrics_to_file()
        raise
    except Exception as e:
        print(f"Unexpected error at epoch {state.epoch}: {e}")
        checkpoint_dir = os.path.join(data_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_data = {
            'epoch': state.epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': state.loss,
            'config': config,
            'error': str(e)
        }
        if scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
        torch.save(checkpoint_data, f"{checkpoint_dir}/emergency_checkpoint_epoch_{state.epoch}.pt")
        print(f"Emergency checkpoint saved at epoch {state.epoch}")
        state.save_metrics_to_file()
        raise
    finally:
        # Save metrics on normal completion too
        state.save_metrics_to_file()

def create_scheduler(optimizer, scheduler_config, num_epochs):
    """Create learning rate scheduler based on config"""
    scheduler_type = scheduler_config.get("type", "none")
    
    if scheduler_type == "none":
        return None
    elif scheduler_type == "cosine":
        min_lr = scheduler_config.get("min_lr", 1e-6)
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_lr)
    elif scheduler_type == "step":
        step_size = scheduler_config.get("step_size", 5000)
        gamma = scheduler_config.get("gamma", 0.5)
        return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == "exponential":
        gamma = scheduler_config.get("gamma", 0.95)
        return lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif scheduler_type == "plateau":
        patience = scheduler_config.get("patience", 1000)
        factor = scheduler_config.get("factor", 0.5)
        return lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=factor)
    elif scheduler_type == "warmup_cosine":
        warmup_epochs = scheduler_config.get("warmup_epochs", 1000)
        min_lr = scheduler_config.get("min_lr", 1e-6)
        # Custom warmup + cosine scheduler
        return WarmupCosineScheduler(optimizer, warmup_epochs=warmup_epochs, 
                                   total_epochs=num_epochs, min_lr=min_lr)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

class WarmupCosineScheduler:
    """Custom scheduler with linear warmup followed by cosine annealing"""
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0

    def step(self):
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            cosine_epochs = self.total_epochs - self.warmup_epochs
            cosine_progress = (self.current_epoch - self.warmup_epochs) / cosine_epochs
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * cosine_progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.current_epoch += 1

    def state_dict(self):
        return {
            'current_epoch': self.current_epoch,
            'warmup_epochs': self.warmup_epochs,
            'total_epochs': self.total_epochs,
            'min_lr': self.min_lr,
            'base_lr': self.base_lr
        }

    def load_state_dict(self, state_dict):
        self.current_epoch = state_dict['current_epoch']
        self.warmup_epochs = state_dict['warmup_epochs']
        self.total_epochs = state_dict['total_epochs']
        self.min_lr = state_dict['min_lr']
        self.base_lr = state_dict['base_lr']


class EarlyStopping:
    """Early stopping based on validation loss."""
    def __init__(self, patience=5000, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False

    def __call__(self, val_loss):
        """
        Check if training should stop based on validation loss.

        Args:
            val_loss: Current validation loss

        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            # Improvement
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return self.should_stop


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """
    Evaluate model on validation or test set.

    Args:
        model: The model to evaluate
        dataloader: DataLoader for validation/test data
        criterion: Loss function
        device: Device to run evaluation on

    Returns:
        Average loss over the dataset
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device).long()
        logits = model(inputs)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        total_loss += loss.item()
        num_batches += 1

    model.train()
    return total_loss / num_batches


def train(config_file: str):
    # Load environment variables from .env file
    load_dotenv()

    # Set up wandb API key from environment
    if os.getenv("WANDB_API_KEY"):
        os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Initialize WandB - if called by wandb.agent(), this will automatically
    # pick up sweep parameters. Otherwise, uses our config.
    wandb.init(
        project=config["train"]["wandb_project_name"],
        name=config["train"]["wandb_run_name"],
        config=config
    )

    # After wandb.init(), wandb.config contains either:
    # 1. Sweep parameters (if running in sweep mode)
    # 2. Our base config (if running standalone)
    # We need to sync these back to our config dict

    # Get all parameters from wandb.config
    print("\nSyncing config from WandB...")
    print(f"WandB config keys: {list(wandb.config.keys())}")

    # Override model parameters from wandb.config
    model_overrides = ['n_layer', 'n_embd', 'n_head', 'd_head', 'n_inner',
                     'n_ctx', 'attn_only', 'normalization_type']
    for param in model_overrides:
        if param in wandb.config:
            old_val = config['model'].get(param)
            new_val = wandb.config[param]
            if old_val != new_val:
                config['model'][param] = new_val
                print(f"  model.{param}: {old_val} → {new_val}")

    # Override training parameters from wandb.config
    train_overrides = ['learning_rate', 'batch_size']
    for param in train_overrides:
        if param in wandb.config:
            old_val = config['train'].get(param)
            new_val = wandb.config[param]
            if old_val != new_val:
                config['train'][param] = new_val
                print(f"  train.{param}: {old_val} → {new_val}")

    # Override scheduler type
    if 'scheduler_type' in wandb.config:
        old_val = config['scheduler'].get('type')
        new_val = wandb.config['scheduler_type']
        if old_val != new_val:
            config['scheduler']['type'] = new_val
            print(f"  scheduler.type: {old_val} → {new_val}")

    # Debug: Print all final model parameters after overrides
    print(f"\n{'='*70}")
    print("FINAL MODEL CONFIGURATION (after all overrides):")
    print(f"{'='*70}")
    print("Model Parameters:")
    print(f"  vocab_size: {config['model'].get('vocab_size')}")
    print(f"  n_ctx: {config['model'].get('n_ctx')}")
    print(f"  n_positions: {config['model'].get('n_positions')}")
    print(f"  n_layer: {config['model'].get('n_layer')}")
    print(f"  n_embd (d_model): {config['model'].get('n_embd')}")
    print(f"  n_head: {config['model'].get('n_head')}")
    print(f"  d_head: {config['model'].get('d_head')}")
    print(f"  n_inner (d_mlp): {config['model'].get('n_inner')}")
    print(f"  activation_function: {config['model'].get('activation_function')}")
    print(f"  normalization_type: {config['model'].get('normalization_type')}")
    print(f"  attn_only: {config['model'].get('attn_only')}")
    print(f"  tie_word_embeddings: {config['model'].get('tie_word_embeddings')}")
    print(f"\nTraining Parameters:")
    print(f"  learning_rate: {config['train'].get('learning_rate')}")
    print(f"  batch_size: {config['train'].get('batch_size')}")
    print(f"  num_epochs: {config['train'].get('num_epochs')}")
    print(f"  device: {config['train'].get('device')}")
    print(f"  process: {config['train'].get('process')}")
    print(f"\nScheduler:")
    print(f"  type: {config['scheduler'].get('type')}")
    print(f"\nDataset:")
    print(f"  mode: {config.get('dataset', {}).get('mode')}")
    print(f"  path: {config.get('dataset', {}).get('path')}")
    print(f"{'='*70}\n")

    # Create data directory using datetime and UUID (for multi-agent safety)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]  # Short UUID for readability
    data_dir = os.path.join("records", f"{timestamp}_{unique_id}")
    os.makedirs(data_dir, exist_ok=False)

    # Note: n_embd doesn't have to equal n_head * d_head
    # The model will handle this internally via projection layers

    # Update dataset path based on n_ctx
    n_ctx = config['model']['n_ctx']
    process_name = config['train']['process']
    expected_dataset_path = f"data/datasets/{process_name}_ctx{n_ctx}"
    current_path = config.get('dataset', {}).get('path')

    print(f"\n{'='*70}")
    print(f"DATASET PATH CHECK:")
    print(f"  n_ctx from config: {n_ctx}")
    print(f"  n_positions from config: {config['model']['n_positions']}")
    print(f"  Current dataset path: {current_path}")
    print(f"  Expected dataset path: {expected_dataset_path}")
    print(f"  Paths match: {current_path == expected_dataset_path}")

    if current_path != expected_dataset_path:
        print(f"  → UPDATING dataset path to: {expected_dataset_path}")
        if 'dataset' not in config:
            config['dataset'] = {}
        config['dataset']['path'] = expected_dataset_path
        print(f"  ✓ Updated config['dataset']['path'] = {config['dataset']['path']}")
    else:
        print(f"  → Dataset path already correct, no update needed")
    print(f"{'='*70}\n")
    
    # Extract training config
    train_config = config["train"]
    batch_size = int(train_config["batch_size"])
    learning_rate = float(train_config["learning_rate"])
    num_epochs = int(train_config["num_epochs"])

    # Device selection with auto-detection (cuda > mps > cpu)
    requested_device = train_config.get("device", "auto")

    if requested_device == "auto":
        # Auto-detect best available device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using device: cuda (auto-detected)")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print(f"Using device: mps (auto-detected)")
        else:
            device = torch.device("cpu")
            print(f"Using device: cpu (auto-detected)")
    else:
        # Respect user's specific device request
        if requested_device == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using device: cuda")
        elif requested_device == "mps" and torch.backends.mps.is_available():
            device = torch.device("mps")
            print(f"Using device: mps")
        elif requested_device in ["cuda", "mps"]:
            print(f"Warning: {requested_device} requested but not available, falling back to CPU")
            device = torch.device("cpu")
            print(f"Using device: cpu")
        else:
            device = torch.device("cpu")
            print(f"Using device: cpu")

    # Dataset configuration
    dataset_config = config.get("dataset", {})
    dataset_mode = dataset_config.get("mode", "precomputed")
    dataset_path = dataset_config.get("path", None)

    print(f"Dataset mode: {dataset_mode}")

    # Initialize HMM process
    process_name = train_config["process"]
    if process_name == "rrxor":
        process = RRXOR()
    elif process_name == "z1r":
        process = Z1R()
    elif process_name == "mess3":
        process = Mess3Proc()
    elif process_name == "psl7":
        process = PSL7HMM()
    else:
        raise ValueError(f"Unknown process: {process_name}")

    # WandB already initialized earlier (to pick up sweep params)

    # Get sequence length for data generation
    seq_length = config["model"]["n_ctx"]

    # Setup data loading based on mode
    train_loader = None
    val_loader = None
    test_loader = None

    if dataset_mode == "precomputed":
        from data_loader import load_all_splits

        # Use provided path or default to data/datasets/{process_name}
        if dataset_path is None:
            dataset_path = os.path.join("data", "datasets", process_name)

        print(f"\n{'='*70}")
        print(f"LOADING DATASET:")
        print(f"  Path: {dataset_path}")
        print(f"{'='*70}")
        train_loader, val_loader, test_loader = load_all_splits(dataset_path, batch_size, device=None)

        # Check actual sequence length from loaded data
        sample_inputs, sample_targets = next(iter(train_loader))
        print(f"\n{'='*70}")
        print(f"LOADED DATA SHAPES:")
        print(f"  Input shape: {sample_inputs.shape} (batch_size, seq_len)")
        print(f"  Target shape: {sample_targets.shape}")
        print(f"  Actual sequence length from data: {sample_inputs.shape[1]}")
        print(f"  Expected from n_ctx: {config['model']['n_ctx']}")
        print(f"{'='*70}\n")

        print(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches, Test: {len(test_loader)} batches")
    else:
        print(f"Using on-the-fly data generation (process: {process_name})")
    
    # Save the UPDATED config (after all overrides) to the data directory
    # This includes both sweep parameter overrides AND dataset path updates
    updated_config_path = os.path.join(data_dir, "config.yaml")
    with open(updated_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"\nSaved updated config to: {updated_config_path}")

    # Initialize model
    # according to the setup, we have two options:
    # 1. HookedTransformerModel
    # 2. MaskedHeadTransformerModel
    # we need to get the model class from the config
    # default to MaskedHeadTransformerModel
    model_class = globals()[config["model"]["name"]]
    if model_class is None:
        print(f"Using MaskedHeadTransformerModel as default")
        model_class = MaskedHeadTransformerModel

    # IMPORTANT: Use the updated config, not the original base config!
    # The updated config has sweep parameters (n_ctx, etc.) applied
    print(f"\nInitializing model with updated config (n_ctx={config['model']['n_ctx']})")
    model = model_class(updated_config_path)
    model = model.to(device)
    

    # Initialize checkpoint validator
    validator = None
    validator_config = config.get("validation", {})
    if validator_config.get("enabled", True):  # Default: enabled
        print("\n" + "="*60)
        print("Initializing checkpoint validator...")
        print("="*60)
        try:
            validator = CheckpointValidator(
                hmm_process=process,
                test_size=validator_config.get("test_size", 1000),
                seq_length=seq_length,
                device=device,
                seed=validator_config.get("seed", 42)
            )
            print(f"Validator initialized with {validator.test_size} test sequences")
            print("="*60 + "\n")
        except Exception as e:
            print(f"Warning: Could not initialize validator: {e}")
            print("Continuing without checkpoint validation")
            print("="*60 + "\n")
            validator = None

    # Setup optimizer and loss function
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Setup learning rate scheduler
    scheduler_config = config.get("scheduler", {"type": "none"})
    scheduler = create_scheduler(optimizer, scheduler_config, num_epochs)
    scheduler_type = scheduler_config.get("type", "none")
    print(f"Using scheduler: {scheduler_type}")

    # Setup validation and early stopping
    eval_interval = dataset_config.get("eval_interval", 1000)
    early_stopping = None
    early_stopping_config = dataset_config.get("early_stopping", {})

    if dataset_mode == "precomputed" and early_stopping_config.get("enabled", False):
        early_stopping = EarlyStopping(
            patience=early_stopping_config.get("patience", 5000),
            min_delta=early_stopping_config.get("min_delta", 0.0001)
        )
        print(f"Early stopping enabled: patience={early_stopping.patience}, min_delta={early_stopping.min_delta}")

    # Training loop
    model.train()

    with checkpoint_on_error(model, optimizer, config, data_dir=data_dir, scheduler=scheduler) as checkpoint_state:
        if dataset_mode == "precomputed":
            # Precomputed dataset mode: iterate through DataLoader until num_epochs steps
            batches_per_epoch = len(train_loader)
            total_steps = num_epochs  # num_epochs is actually total steps in this config
            num_passes = (total_steps + batches_per_epoch - 1) // batches_per_epoch  # Ceiling division

            print(f"Training with pre-computed dataset ({batches_per_epoch} batches per pass)")
            print(f"Training for {total_steps} total steps (~{num_passes} passes through dataset)")
            print(f"Validation every {eval_interval} steps")
            step = 0
            early_stop = False

            # Keep cycling through the dataset until we reach num_epochs steps
            pass_num = 0
            while step < total_steps:
                pass_num += 1
                print(f"\n{'='*70}")
                print(f"Dataset pass {pass_num}/{num_passes} (Steps {step}/{total_steps})")
                print(f"{'='*70}")

                for inputs, targets in train_loader:
                    # Check if we've reached the target number of steps
                    if step >= total_steps:
                        break

                    # Move data to device
                    inputs = inputs.to(device)
                    targets = targets.to(device).long()

                    # Forward pass
                    optimizer.zero_grad()
                    logits = model(inputs)

                    # Reshape for loss computation
                    loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

                    # Backward pass
                    loss.backward()
                    optimizer.step()

                    # Update learning rate scheduler (skip plateau here, it's updated on validation)
                    if scheduler is not None and scheduler_type != "plateau":
                        scheduler.step()

                    # Update checkpoint state
                    checkpoint_state.update(step, loss.item())

                    # Get current learning rate
                    current_lr = optimizer.param_groups[0]['lr']

                    # Prepare metrics
                    metrics = {
                        "step": step,
                        "pass": pass_num,
                        "loss": loss.item(),
                        "learning_rate": current_lr
                    }

                    # Periodic validation
                    if step % eval_interval == 0 and step > 0:
                        val_loss = evaluate(model, val_loader, criterion, device)
                        print(f"Step {step}/{total_steps} (Pass {pass_num}): Train Loss = {loss.item():.6f}, Val Loss = {val_loss:.6f}, LR = {current_lr:.2e}")

                        # Add validation loss to metrics
                        metrics["val_loss"] = val_loss

                        # Update plateau scheduler with validation loss
                        if scheduler is not None and scheduler_type == "plateau":
                            scheduler.step(val_loss)

                        # Early stopping check
                        if early_stopping and early_stopping(val_loss):
                            print(f"\nEarly stopping triggered at step {step} (Pass {pass_num})")
                            print(f"Best validation loss: {early_stopping.best_loss:.6f}")
                            # Log final metrics before breaking
                            wandb.log(metrics)
                            checkpoint_state.log_metrics(metrics)
                            early_stop = True
                            break
                    else:
                        # Console logging (without validation)
                        if step % 1000 == 0:
                            print(f"Step {step}/{total_steps} (Pass {pass_num}), Loss: {loss.item():.6f}, LR: {current_lr:.2e}")

                    # Log metrics to wandb
                    wandb.log(metrics)

                    # Log metrics to context manager for local saving
                    checkpoint_state.log_metrics(metrics)

                    # Save checkpoint periodically
                    if step % 10000 == 0 and step > 0:
                        checkpoint_dir = os.path.join(data_dir, "checkpoints")
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        checkpoint_data = {
                            'step': step,
                            'pass': pass_num,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss.item(),
                            'config': config
                        }
                        if scheduler is not None:
                            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
                        torch.save(checkpoint_data, f"{checkpoint_dir}/checkpoint_step_{step}.pt")

                        # Run checkpoint validation
                        if validator is not None:
                            try:
                                val_metrics = validator.validate_all(
                                    model=model,
                                    step=step,
                                    num_geometry_samples=validator_config.get("geometry_samples", 1000),
                                    num_llr_samples=validator_config.get("llr_samples", 1000)
                                )

                                # Log to WandB
                                validator.log_to_wandb(val_metrics, step=step)

                                # Print summary
                                print(f"  Geometry: Mean R² = {val_metrics.get('mean_r2', 0):.4f}")
                                print(f"  LLR: Mean = {val_metrics.get('llr_mean', 0):.6f}, "
                                      f"Std = {val_metrics.get('llr_std', 0):.6f}\n")
                            except Exception as e:
                                print(f"Warning: Checkpoint validation failed: {e}")
                                # Don't crash training - continue even if validation fails

                    step += 1

                # Break outer loop if early stopping triggered
                if early_stop:
                    break

            # Final test evaluation
            if test_loader is not None:
                print("\n" + "=" * 70)
                print("FINAL EVALUATION ON TEST SET")
                print("=" * 70)
                test_loss = evaluate(model, test_loader, criterion, device)
                print(f"Test Loss: {test_loss:.6f}")
                wandb.log({"test_loss": test_loss})
                print("=" * 70)

        else:
            # On-the-fly generation mode: original behavior
            print(f"Training with on-the-fly data generation ({num_epochs} epochs)")

            for epoch in range(num_epochs):
                # Generate fresh training data for this epoch
                train_data = process.generate_data(batch_size, seq_length + 1)  # +1 for next token prediction

                # Create input/target pairs
                inputs = train_data[:, :-1].to(device)  # All tokens except last
                targets = train_data[:, 1:].to(device).long()  # All tokens except first (shifted by 1), ensure LongTensor

                # Forward pass
                optimizer.zero_grad()
                logits = model(inputs)

                # Reshape for loss computation
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

                # Backward pass
                loss.backward()
                optimizer.step()

                # Update learning rate scheduler
                # Note: plateau scheduler needs validation loss, but on-the-fly mode has no validation
                # So we use training loss as a proxy (not ideal but works)
                if scheduler is not None:
                    if scheduler_type == "plateau":
                        scheduler.step(loss.item())
                    else:
                        scheduler.step()

                # Update checkpoint state
                checkpoint_state.update(epoch, loss.item())

                # Get current learning rate
                current_lr = optimizer.param_groups[0]['lr']

                # Prepare metrics
                metrics = {
                    "epoch": epoch,
                    "loss": loss.item(),
                    "learning_rate": current_lr
                }

                # Log metrics to wandb
                wandb.log(metrics)

                # Log metrics to context manager for local saving
                checkpoint_state.log_metrics(metrics)

                # Console logging
                if epoch % 1000 == 0:
                    print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item():.6f}, LR: {current_lr:.2e}")

                # Save checkpoint periodically
                if epoch % 10000 == 0 and epoch > 0:
                    checkpoint_dir = os.path.join(data_dir, "checkpoints")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    checkpoint_data = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.item(),
                        'config': config
                    }
                    if scheduler is not None:
                        checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
                    torch.save(checkpoint_data, f"{checkpoint_dir}/checkpoint_epoch_{epoch}.pt")
    
    # Save final model
    torch.save(model.state_dict(), f"{data_dir}/final_model_{process_name}.pt")
    print(f"Training completed. Final model saved as {data_dir}/final_model_{process_name}.pt")
    
    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    # train("config/mess3_ctx4.yaml")
    # train("config/mess3_ctx6.yaml")
    # train("config/mess3_ctx8.yaml")
    # train("config/mess3_ctx10.yaml")
    # train("config/mess3_ctx15.yaml")
    train("config/reproduction_config.yaml")