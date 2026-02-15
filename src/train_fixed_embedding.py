"""
Training script with frozen embeddings.
Mirrors train.py but freezes W_E, W_U, b_U at initialization.
"""

import argparse
import torch as t
from utils import initialize_transformer_from_yaml
from embedding_init import create_embedding
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
import json
from datetime import datetime
from torch.utils.data import DataLoader
import wandb
import time
from pathlib import Path

device = "cuda" if t.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Train transformer with frozen embeddings")
    parser.add_argument("--embedding_mode", required=True,
                        choices=["trained", "random", "coulomb", "sparse_binary", "onehot", "pmi"],
                        help="Embedding initialization strategy")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to checkpoint (required for 'trained' mode)")
    parser.add_argument("--config", type=str, default="base_config.yaml",
                        help="Path to config YAML")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for embedding initialization")
    parser.add_argument("--coulomb_steps", type=int, default=5000,
                        help="Gradient descent steps for Coulomb mode")
    parser.add_argument("--nnz_per_row", type=int, default=3,
                        help="Non-zeros per row for sparse_binary mode")
    return parser.parse_args()


def print_embedding_stats(W_E, mode):
    """Print pairwise cosine similarity statistics."""
    W_norm = W_E / W_E.norm(dim=1, keepdim=True).clamp(min=1e-8)
    cos_sim = W_norm @ W_norm.T
    # Extract upper triangle (excluding diagonal)
    mask = t.triu(t.ones_like(cos_sim, dtype=t.bool), diagonal=1)
    pairwise = cos_sim[mask]
    print(f"\n  Embedding stats ({mode}):")
    print(f"    Shape: {list(W_E.shape)}")
    print(f"    Row norms: mean={W_E.norm(dim=1).mean():.4f}, "
          f"min={W_E.norm(dim=1).min():.4f}, max={W_E.norm(dim=1).max():.4f}")
    print(f"    Pairwise cosine sim: mean={pairwise.mean():.4f}, "
          f"min={pairwise.min():.4f}, max={pairwise.max():.4f}")


def train():
    args = parse_args()

    if args.embedding_mode == "trained" and args.checkpoint_path is None:
        raise ValueError("--checkpoint_path is required for 'trained' embedding mode")

    # Load base config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    mcfg = cfg["model"]
    mcfg["d_head"] = mcfg["n_embd"] // mcfg["n_head"]
    mcfg["d_mlp"] = 4 * mcfg["n_embd"]

    d_vocab = mcfg["vocab_size"]
    d_model = mcfg["n_embd"]

    # Initialize wandb
    wandb.init(
        project=cfg["wandb"]["project_name"],
        config={**cfg, "embedding_mode": args.embedding_mode, "embedding_seed": args.seed,
                "source_checkpoint_dir": str(Path(args.config).parent)},
    )

    # Run name
    norm_str = mcfg["normalization_type"] or "noLN"
    attn_str = "attn_only" if mcfg["attn_only"] else "full"
    run_name = f"fixed_{args.embedding_mode}_L{mcfg['n_layer']}_d{mcfg['n_embd']}_H{mcfg['n_head']}_{attn_str}_{norm_str}"
    wandb.run.name = run_name

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("models/cylinderhmm_fixed_emb_new") / f"{timestamp}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # Initialize model
    model = initialize_transformer_from_yaml(None, model_cfg=mcfg)

    # ── Create and freeze embedding ────────────────────────────────────
    print(f"\nInitializing embedding: mode={args.embedding_mode}")
    embedding_kwargs = {
        "seed": args.seed,
        "n_steps": args.coulomb_steps,
        "nnz_per_row": args.nnz_per_row,
    }
    if args.embedding_mode == "trained":
        embedding_kwargs["checkpoint_path"] = args.checkpoint_path

    W_E = create_embedding(args.embedding_mode, d_vocab, d_model, **embedding_kwargs)
    print_embedding_stats(W_E, args.embedding_mode)

    # Set embedding and unembed weights
    model.embed.W_E.data = W_E
    model.unembed.W_U.data = W_E.T
    model.unembed.b_U.data.zero_()

    # Freeze
    model.embed.W_E.requires_grad = False
    model.unembed.W_U.requires_grad = False
    model.unembed.b_U.requires_grad = False

    model.to(device)

    # ── Param counts ───────────────────────────────────────────────────
    total_params = sum(p.numel() for p in model.parameters())
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_params = total_params - frozen_params
    print(f"\n  Total params:     {total_params}")
    print(f"  Frozen params:    {frozen_params}  (W_E: {d_vocab*d_model}, W_U: {d_vocab*d_model}, b_U: {d_vocab})")
    print(f"  Trainable params: {trainable_params}")

    # Save snapshot of frozen embedding for verification
    W_E_snapshot = model.embed.W_E.data.clone()

    # ── Optimizer (trainable params only) ──────────────────────────────
    tcfg = cfg["train"]
    ocfg = tcfg["optimizer"]

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = t.optim.AdamW(
        trainable,
        lr=float(ocfg["learning_rate"]),
        betas=(float(ocfg["adam_beta1"]), float(ocfg["adam_beta2"])),
        eps=float(ocfg["adam_epsilon"]),
        weight_decay=float(ocfg["adam_weight_decay"]),
    )

    # Load data
    train_data = t.load("data/datasets/cylinder_graph_hmm/train/observations.pt")
    test_data = t.load("data/datasets/cylinder_graph_hmm/test/observations.pt")

    num_epochs = tcfg["num_epochs"]
    batch_size = tcfg["batch_size"]
    seq_length = tcfg["seq_length"]
    val_interval = tcfg["val_interval_in_opt_steps"]
    log_interval = tcfg["log_interval_in_opt_steps"]

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=(device == "cuda"))
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=(device == "cuda"))

    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch
    print(f"\n  Dataset: {len(train_data)} train, {len(test_data)} test sequences")
    print(f"  Steps per epoch: {steps_per_epoch}, Total steps: {total_steps}")

    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=float(ocfg["learning_rate"]) * 0.001)

    best_val_loss = float("inf")

    metadata = {
        "run_name": run_name,
        "embedding_mode": args.embedding_mode,
        "source_checkpoint_dir": str(Path(args.config).parent),
        "wandb_run_id": wandb.run.id,
        "timestamp": timestamp,
        "device": device,
        "total_params": total_params,
        "frozen_params": frozen_params,
        "trainable_params": trainable_params,
        "train_sequences": len(train_data),
        "test_sequences": len(test_data),
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # ── Training loop ──────────────────────────────────────────────────
    model.train()
    global_step = 0
    start_time = time.time()
    verified_frozen = False

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
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            # Verify embeddings are frozen after first step
            if not verified_frozen:
                assert model.embed.W_E.grad is None, "W_E should not have gradients!"
                assert t.allclose(model.embed.W_E.data, W_E_snapshot), "W_E changed after optimizer step!"
                print("  [OK] Verified: embeddings are frozen.")
                verified_frozen = True

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
                            val_targets.reshape(-1),
                        )
                        val_loss += val_batch_loss.item()
                        num_batches += 1

                val_loss /= num_batches
                wandb.log({"val_loss": val_loss, "global_step": global_step})
                print(f"  Step {global_step}: Val Loss = {val_loss:.6f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    t.save(
                        {
                            "epoch": epoch,
                            "global_step": global_step,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "val_loss": val_loss,
                            "embedding_mode": args.embedding_mode,
                        },
                        run_dir / "best_model.pt",
                    )
                    print(f"    Saved best model (val_loss: {val_loss:.6f})")

                model.train()

            # Training metrics logging
            if global_step % log_interval == 0:
                elapsed_time = time.time() - start_time
                tokens_processed = global_step * batch_size * seq_length
                tokens_per_sec = tokens_processed / elapsed_time if elapsed_time > 0 else 0
                current_lr = optimizer.param_groups[0]["lr"]

                wandb.log(
                    {
                        "train_loss": loss.item(),
                        "learning_rate": current_lr,
                        "tokens_per_sec": tokens_per_sec,
                        "global_step": global_step,
                        "epoch": epoch,
                    }
                )

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"  Epoch {epoch+1} complete: Avg Train Loss = {avg_epoch_loss:.6f}")
        wandb.log({"epoch_train_loss": avg_epoch_loss, "epoch": epoch})

        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": avg_epoch_loss,
            "embedding_mode": args.embedding_mode,
        }
        t.save(checkpoint, run_dir / f"checkpoint_epoch_{epoch+1}.pt")
        t.save(checkpoint, run_dir / "latest.pt")
        print(f"  Saved checkpoint: checkpoint_epoch_{epoch+1}.pt")

    # Final metadata
    metadata["best_val_loss"] = best_val_loss
    metadata["final_train_loss"] = avg_epoch_loss
    metadata["total_time_seconds"] = time.time() - start_time
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nTraining complete! Output saved to {run_dir}")
    wandb.finish()


if __name__ == "__main__":
    train()
