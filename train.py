from hmm import RRXOR, Z1R, Mess3Proc, PSL7HMM
from model import HookedTransformerModel, MaskedHeadTransformerModel
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
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + torch.cos(torch.pi * cosine_progress))
        
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

def train(config_file: str):
    # Load environment variables from .env file
    # initalize the data directory using datatime
    data_dir = os.path.join("records", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(data_dir, exist_ok=False)
    # copy the config file to the data directory
    shutil.copy(config_file, os.path.join(data_dir, "config.yaml"))
    load_dotenv()
    
    # Set up wandb API key from environment
    if os.getenv("WANDB_API_KEY"):
        os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
    
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    # Extract training config
    train_config = config["train"]
    batch_size = int(train_config["batch_size"])
    learning_rate = float(train_config["learning_rate"])
    num_epochs = int(train_config["num_epochs"])

    # Device selection with MPS support
    requested_device = train_config["device"]
    if requested_device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif requested_device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif requested_device in ["cuda", "mps"]:
        print(f"Warning: {requested_device} requested but not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    
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
    
    # Initialize wandb
    wandb.init(
        project=config["train"]["wandb_project_name"],
        name=config["train"]["wandb_run_name"],
    )
    
    # Get sequence length for data generation
    seq_length = config["model"]["n_ctx"]
    
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
        
    model = model_class(config_file)
    model = model.to(device)
    
    # Setup optimizer and loss function
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Setup learning rate scheduler
    scheduler_config = config.get("scheduler", {"type": "none"})
    scheduler = create_scheduler(optimizer, scheduler_config, num_epochs)
    scheduler_type = scheduler_config.get("type", "none")
    print(f"Using scheduler: {scheduler_type}")
    
    # Training loop
    model.train()
    
    with checkpoint_on_error(model, optimizer, config, data_dir=data_dir, scheduler=scheduler) as checkpoint_state:
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
    train("config/psl7.yaml")
    