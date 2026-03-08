import argparse
import os
import torch as t
from utils import initialize_transformer_from_yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
import json
from datetime import datetime
from torch.utils.data import DataLoader
import wandb
import time
from pathlib import Path

device = "cuda" if t.cuda.is_available() else "cpu"


def train(config_path: str = "base_config.yaml"):
    # Load base config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Initialize wandb (sweep agent injects swept params as flat keys into wandb.config)
    wandb_tags = cfg.get("wandb", {}).get("tags", None)
    wandb.init(
        project=cfg["wandb"]["project_name"],
        config=cfg,
        tags=wandb_tags,
    )

    # Apply sweep overrides to model config
    mcfg = cfg["model"]
    sweep_keys = ["n_layer", "n_embd", "n_head", "attn_only", "normalization_type"]
    for key in sweep_keys:
        if key in wandb.config:
            val = wandb.config[key]
            mcfg[key] = None if val == "none" else val

    mcfg["d_head"] = mcfg["n_embd"] // mcfg["n_head"]
    mcfg["d_mlp"] = 4 * mcfg["n_embd"]

    # Set descriptive run name
    norm_str = mcfg["normalization_type"] or "noLN"
    attn_str = "attn_only" if mcfg["attn_only"] else "full"
    run_name = f"L{mcfg['n_layer']}_d{mcfg['n_embd']}_H{mcfg['n_head']}_{attn_str}_{norm_str}"
    wandb.run.name = run_name

    # Set up run output directory (derived from process name)
    process_name = cfg["data_generator"]["process"]["name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("models") / process_name / f"{timestamp}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save the full resolved config for this run
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # Initialize the model
    model = initialize_transformer_from_yaml(None, model_cfg=mcfg)
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters())

    tcfg = cfg["train"]
    ocfg = tcfg["optimizer"]

    optimizer = t.optim.AdamW(
        model.parameters(),
        lr=float(ocfg['learning_rate']),
        betas=(float(ocfg['adam_beta1']), float(ocfg['adam_beta2'])),
        eps=float(ocfg["adam_epsilon"]),
        weight_decay=float(ocfg["adam_weight_decay"]),
    )

    # Load data (default to data_generator.save_dir from config)
    data_dir = cfg.get("data", {}).get("data_dir", cfg["data_generator"]["save_dir"])
    train_data = t.load(os.path.join(data_dir, "train", "observations.pt"))
    test_data = t.load(os.path.join(data_dir, "test", "observations.pt"))

    num_epochs = tcfg["num_epochs"]
    batch_size = tcfg["batch_size"]
    seq_length = tcfg["seq_length"]
    val_interval = tcfg["val_interval_in_opt_steps"]
    log_interval = tcfg["log_interval_in_opt_steps"]

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=(device=='cuda'))
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=(device=='cuda'))

    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch
    print(f"Dataset: {len(train_data)} train sequences, {len(test_data)} test sequences")
    print(f"Steps per epoch: {steps_per_epoch}, Total steps: {total_steps}")

    # Scheduler setup (configurable; defaults to cosine for backward compatibility)
    sched_cfg = tcfg.get("scheduler", {"type": "cosine"})
    sched_type = sched_cfg.get("type", "cosine") if isinstance(sched_cfg, dict) else sched_cfg
    if sched_type == "cosine":
        eta_min = float(sched_cfg.get("eta_min", float(ocfg["learning_rate"]) * 0.001))
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=eta_min)
    elif sched_type == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler type: {sched_type}")

    best_val_loss = float('inf')

    # Save metadata at the start
    metadata = {
        "run_name": run_name,
        "wandb_run_id": wandb.run.id,
        "timestamp": timestamp,
        "device": device,
        "num_params": num_params,
        "train_sequences": len(train_data),
        "test_sequences": len(test_data),
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Training loop
    model.train()
    global_step = 0
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            optimizer.zero_grad()
            logits = model(inputs)
            loss = t.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            # Validation
            if global_step % val_interval == 0:
                model.eval()
                val_loss = 0.0
                num_batches = 0

                with t.no_grad():
                    for val_batch in test_loader:
                        val_batch = val_batch.to(device)
                        val_inputs = val_batch[:, :-1]
                        val_targets = val_batch[:, 1:]

                        val_logits = model(val_inputs)
                        val_batch_loss = t.nn.functional.cross_entropy(
                            val_logits.reshape(-1, val_logits.size(-1)),
                            val_targets.reshape(-1)
                        )
                        val_loss += val_batch_loss.item()
                        num_batches += 1

                val_loss /= num_batches
                wandb.log({"val_loss": val_loss, "global_step": global_step})
                print(f"Step {global_step}: Val Loss = {val_loss:.6f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_data = {
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                    }
                    if scheduler is not None:
                        checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
                    t.save(checkpoint_data, run_dir / "best_model.pt")
                    print(f"  Saved best model (val_loss: {val_loss:.6f})")

                model.train()

            # Training metrics logging
            if global_step % log_interval == 0:
                elapsed_time = time.time() - start_time
                tokens_processed = global_step * batch_size * seq_length
                tokens_per_sec = tokens_processed / elapsed_time if elapsed_time > 0 else 0
                current_lr = optimizer.param_groups[0]['lr']

                wandb.log({
                    "train_loss": loss.item(),
                    "learning_rate": current_lr,
                    "tokens_per_sec": tokens_per_sec,
                    "global_step": global_step,
                    "epoch": epoch,
                })

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} complete: Avg Train Loss = {avg_epoch_loss:.6f}")
        wandb.log({"epoch_train_loss": avg_epoch_loss, "epoch": epoch})

        checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_epoch_loss,
        }
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        t.save(checkpoint, run_dir / f"checkpoint_epoch_{epoch+1}.pt")
        t.save(checkpoint, run_dir / "latest.pt")
        print(f"  Saved checkpoint: checkpoint_epoch_{epoch+1}.pt")

    # Update metadata with final results
    metadata["best_val_loss"] = best_val_loss
    metadata["final_train_loss"] = avg_epoch_loss
    metadata["total_time_seconds"] = time.time() - start_time
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Training complete! Output saved to {run_dir}")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train transformer on HMM data")
    parser.add_argument("--config", type=str, default="base_config.yaml",
                        help="Path to config YAML file")
    args = parser.parse_args()
    train(config_path=args.config)
