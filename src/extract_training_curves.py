"""
Extract training loss curves from local wandb run directories.

Scans wandb/run-*/files/ for cylinder graph HMM runs, parses config.yaml
for architecture params and direction (forward vs reversed), and extracts
per-step val_loss and per-epoch train_loss from output.log.

Output → data/training_dynamics/:
  - loss_curves.csv: [run_id, n_layer, n_embd, attn_only, norm, direction, step, val_loss]
  - epoch_train_loss.csv: [run_id, ..., epoch, train_loss]
  - run_metadata.csv: [run_id, ..., final_val_loss, best_val_loss, total_steps]

Usage:
    python src/extract_training_curves.py
    python src/extract_training_curves.py --wandb_dir wandb --filter_embd 4 8 16
"""

import argparse
import glob
import os
import re

import pandas as pd
import yaml


WANDB_DIR = "wandb"
OUT_DIR = "data/training_dynamics"


def parse_config(config_path):
    """Parse wandb config.yaml to extract architecture params and direction.

    wandb configs have a nested structure where sweep overrides appear as
    top-level keys with {value: X}, while base config is nested under
    sections like model.value.n_layer.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        return None

    # Check if this is a cylinder graph run
    dg = cfg.get("data_generator", {})
    if isinstance(dg, dict) and "value" in dg:
        dg = dg["value"]
    proc = dg.get("process", {})
    proc_name = proc.get("name", "")
    if proc_name != "cylinder_graph":
        return None

    # Extract model params — sweep overrides take precedence over model section
    model = cfg.get("model", {})
    if isinstance(model, dict) and "value" in model:
        model = model["value"]

    def get_param(key):
        # Check top-level sweep override first
        top = cfg.get(key, {})
        if isinstance(top, dict) and "value" in top:
            return top["value"]
        # Fall back to model section
        return model.get(key, None)

    n_layer = get_param("n_layer")
    n_embd = get_param("n_embd")
    attn_only = get_param("attn_only")
    norm = get_param("normalization_type")

    if any(x is None for x in [n_layer, n_embd, attn_only]):
        return None

    # Normalize normalization_type
    if norm is None or norm == "null" or norm == "none":
        norm = "none"
    else:
        norm = str(norm)

    # Determine direction: check data path and tags
    direction = "forward"

    # Check data.value.data_dir
    data_sec = cfg.get("data", {})
    if isinstance(data_sec, dict) and "value" in data_sec:
        data_sec = data_sec["value"]
    data_dir = data_sec.get("data_dir", "")
    if "reversed" in str(data_dir):
        direction = "reversed"

    # Check wandb tags
    wandb_sec = cfg.get("wandb", {})
    if isinstance(wandb_sec, dict) and "value" in wandb_sec:
        wandb_sec = wandb_sec["value"]
    tags = wandb_sec.get("tags", [])
    if isinstance(tags, list) and "reversed" in tags:
        direction = "reversed"

    # Also check experiment name
    experiment = wandb_sec.get("experiment", "")
    if "reversed" in str(experiment):
        direction = "reversed"

    return {
        "n_layer": int(n_layer),
        "n_embd": int(n_embd),
        "attn_only": bool(attn_only),
        "norm": norm,
        "direction": direction,
    }


def parse_output_log(log_path):
    """Parse output.log for per-step val_loss and per-epoch train_loss.

    Expected formats:
        Step 100: Val Loss = 3.405112
        Epoch 1 complete: Avg Train Loss = 2.709618
    """
    step_pattern = re.compile(r"^Step (\d+): Val Loss = ([\d.]+)")
    epoch_pattern = re.compile(r"^Epoch (\d+) complete: Avg Train Loss = ([\d.]+)")

    val_steps = []
    val_losses = []
    epoch_nums = []
    train_losses = []
    complete = False

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()

            m = step_pattern.match(line)
            if m:
                val_steps.append(int(m.group(1)))
                val_losses.append(float(m.group(2)))
                continue

            m = epoch_pattern.match(line)
            if m:
                epoch_nums.append(int(m.group(1)))
                train_losses.append(float(m.group(2)))
                continue

            if "Training complete" in line:
                complete = True

    return {
        "val_steps": val_steps,
        "val_losses": val_losses,
        "epoch_nums": epoch_nums,
        "train_losses": train_losses,
        "complete": complete,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract training curves from wandb logs")
    parser.add_argument("--wandb_dir", default=WANDB_DIR, help="Path to wandb directory")
    parser.add_argument("--out_dir", default=OUT_DIR, help="Output directory for CSVs")
    parser.add_argument(
        "--filter_embd", nargs="+", type=int, default=[4, 8, 16],
        help="Filter to these n_embd values (default: 4 8 16)"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    run_dirs = sorted(glob.glob(os.path.join(args.wandb_dir, "run-*/files")))
    print(f"Found {len(run_dirs)} wandb run directories")

    loss_rows = []
    epoch_rows = []
    meta_rows = []
    skipped = {"no_config": 0, "not_cylinder": 0, "incomplete": 0, "filtered_embd": 0}

    for run_dir in run_dirs:
        run_id = os.path.basename(os.path.dirname(run_dir))  # e.g. run-20260204_220232-hh1ymf9q

        config_path = os.path.join(run_dir, "config.yaml")
        log_path = os.path.join(run_dir, "output.log")

        if not os.path.isfile(config_path):
            skipped["no_config"] += 1
            continue

        params = parse_config(config_path)
        if params is None:
            skipped["not_cylinder"] += 1
            continue

        if params["n_embd"] not in args.filter_embd:
            skipped["filtered_embd"] += 1
            continue

        if not os.path.isfile(log_path):
            skipped["incomplete"] += 1
            continue

        log_data = parse_output_log(log_path)
        if not log_data["complete"]:
            skipped["incomplete"] += 1
            continue

        if len(log_data["val_steps"]) == 0:
            skipped["incomplete"] += 1
            continue

        # Build loss curve rows
        for step, val_loss in zip(log_data["val_steps"], log_data["val_losses"]):
            loss_rows.append({
                "run_id": run_id,
                "n_layer": params["n_layer"],
                "n_embd": params["n_embd"],
                "attn_only": params["attn_only"],
                "norm": params["norm"],
                "direction": params["direction"],
                "step": step,
                "val_loss": val_loss,
            })

        # Build epoch train loss rows
        for epoch, train_loss in zip(log_data["epoch_nums"], log_data["train_losses"]):
            epoch_rows.append({
                "run_id": run_id,
                "n_layer": params["n_layer"],
                "n_embd": params["n_embd"],
                "attn_only": params["attn_only"],
                "norm": params["norm"],
                "direction": params["direction"],
                "epoch": epoch,
                "train_loss": train_loss,
            })

        # Build metadata row
        meta_rows.append({
            "run_id": run_id,
            "n_layer": params["n_layer"],
            "n_embd": params["n_embd"],
            "attn_only": params["attn_only"],
            "norm": params["norm"],
            "direction": params["direction"],
            "final_val_loss": log_data["val_losses"][-1],
            "best_val_loss": min(log_data["val_losses"]),
            "total_steps": log_data["val_steps"][-1],
        })

    # Save CSVs
    df_loss = pd.DataFrame(loss_rows)
    df_epoch = pd.DataFrame(epoch_rows)
    df_meta = pd.DataFrame(meta_rows)

    loss_path = os.path.join(args.out_dir, "loss_curves.csv")
    epoch_path = os.path.join(args.out_dir, "epoch_train_loss.csv")
    meta_path = os.path.join(args.out_dir, "run_metadata.csv")

    df_loss.to_csv(loss_path, index=False)
    df_epoch.to_csv(epoch_path, index=False)
    df_meta.to_csv(meta_path, index=False)

    # Summary
    n_fwd = df_meta[df_meta["direction"] == "forward"].shape[0] if len(df_meta) > 0 else 0
    n_rev = df_meta[df_meta["direction"] == "reversed"].shape[0] if len(df_meta) > 0 else 0
    print(f"\n=== Extraction Summary ===")
    print(f"  Runs extracted: {len(meta_rows)} ({n_fwd} forward, {n_rev} reversed)")
    print(f"  Skipped: {skipped}")
    print(f"  Val loss data points: {len(loss_rows)}")
    print(f"  Epoch train loss data points: {len(epoch_rows)}")
    print(f"\n  Saved:")
    print(f"    {loss_path} ({df_loss.shape[0]} rows)")
    print(f"    {epoch_path} ({df_epoch.shape[0]} rows)")
    print(f"    {meta_path} ({df_meta.shape[0]} rows)")

    # Show unique configs
    if len(df_meta) > 0:
        key_cols = ["n_layer", "n_embd", "attn_only", "norm", "direction"]
        configs = df_meta.groupby(key_cols).size().reset_index(name="n_runs")
        print(f"\n=== Unique Configs ({len(configs)}) ===")
        print(configs.to_string(index=False))


if __name__ == "__main__":
    main()
