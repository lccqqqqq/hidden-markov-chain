"""
Multi-seed learning dynamics experiment: forward vs reversed CylinderGraph HMM.

Trains transformers on forward and reversed data with multiple random seeds,
recording full training curves. The goal is to detect asymmetry in learning
dynamics (convergence rate, variance across seeds) even though the final
entropy rate is the same.

This script manages the full experiment:
1. Trains N_SEEDS runs on forward data
2. Trains N_SEEDS runs on reversed data
3. Collects all loss curves into a single output file

Can be run locally (sequential) or used to generate cluster submission scripts.

Usage:
    # Run locally (sequential, for testing)
    python src/learning_dynamics_experiment.py --mode local --n_seeds 3

    # Generate SLURM job array script
    python src/learning_dynamics_experiment.py --mode slurm --n_seeds 20

    # Collect results after all jobs finish
    python src/learning_dynamics_experiment.py --mode collect
"""

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))


N_SEEDS = 20
OUT_DIR = "out/learning_dynamics"
CONFIG_FWD = "config/base_config.yaml"
CONFIG_REV = "config/base_config_reversed.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-seed learning dynamics experiment")
    parser.add_argument("--mode", choices=["local", "slurm", "collect"],
                        default="local", help="Run mode")
    parser.add_argument("--n_seeds", type=int, default=N_SEEDS,
                        help="Number of random seeds")
    parser.add_argument("--config_fwd", type=str, default=CONFIG_FWD)
    parser.add_argument("--config_rev", type=str, default=CONFIG_REV)
    parser.add_argument("--out_dir", type=str, default=OUT_DIR)
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing (local mode)")
    return parser.parse_args()


def run_local(args):
    """Run all training jobs sequentially on the local machine."""
    os.makedirs(args.out_dir, exist_ok=True)

    for direction, config in [("forward", args.config_fwd), ("reversed", args.config_rev)]:
        for seed in range(args.n_seeds):
            curve_path = os.path.join(
                args.out_dir, f"curve_{direction}_seed{seed}.json")

            if os.path.exists(curve_path):
                print(f"  Skipping {direction} seed={seed} (already exists)")
                continue

            cmd = [
                sys.executable, "src/train.py",
                "--config", config,
                "--model_seed", str(seed),
                "--loss_curve_path", curve_path,
            ]

            print(f"\n{'='*60}")
            print(f"Running: {direction} seed={seed}")
            print(f"  Config: {config}")
            print(f"  Output: {curve_path}")
            print(f"  Command: {' '.join(cmd)}")
            print(f"{'='*60}")

            if args.dry_run:
                continue

            result = subprocess.run(cmd, capture_output=False)
            if result.returncode != 0:
                print(f"WARNING: {direction} seed={seed} failed with code {result.returncode}")


def generate_slurm(args):
    """Generate a SLURM job array script for cluster execution."""
    os.makedirs(args.out_dir, exist_ok=True)

    total_jobs = 2 * args.n_seeds  # forward + reversed
    script = f"""#!/bin/bash
#SBATCH --job-name=learn_dyn
#SBATCH --array=0-{total_jobs - 1}
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --output={args.out_dir}/slurm_%A_%a.out
#SBATCH --error={args.out_dir}/slurm_%A_%a.err

# Load environment
module load python
source activate hmm  # adjust to your conda env

cd ${{SLURM_SUBMIT_DIR:-$(pwd)}}

# Determine direction and seed from array index
TASK_ID=$SLURM_ARRAY_TASK_ID
N_SEEDS={args.n_seeds}

if [ $TASK_ID -lt $N_SEEDS ]; then
    DIRECTION="forward"
    CONFIG="{args.config_fwd}"
    SEED=$TASK_ID
else
    DIRECTION="reversed"
    CONFIG="{args.config_rev}"
    SEED=$((TASK_ID - N_SEEDS))
fi

CURVE_PATH="{args.out_dir}/curve_${{DIRECTION}}_seed${{SEED}}.json"

echo "Direction: $DIRECTION, Seed: $SEED, Config: $CONFIG"
echo "Output: $CURVE_PATH"

python src/train.py \\
    --config "$CONFIG" \\
    --model_seed "$SEED" \\
    --loss_curve_path "$CURVE_PATH"
"""

    script_path = os.path.join(args.out_dir, "submit_learning_dynamics.sh")
    with open(script_path, "w") as f:
        f.write(script)
    os.chmod(script_path, 0o755)

    print(f"SLURM script written to {script_path}")
    print(f"  Total jobs: {total_jobs} ({args.n_seeds} forward + {args.n_seeds} reversed)")
    print(f"  Submit with: sbatch {script_path}")


def collect_results(args):
    """Collect all per-seed loss curves into a single summary file."""
    all_curves = {"forward": {}, "reversed": {}}
    missing = []

    for direction in ["forward", "reversed"]:
        for seed in range(args.n_seeds):
            curve_path = os.path.join(
                args.out_dir, f"curve_{direction}_seed{seed}.json")

            if not os.path.exists(curve_path):
                missing.append(f"{direction}_seed{seed}")
                continue

            with open(curve_path) as f:
                data = json.load(f)

            all_curves[direction][str(seed)] = {
                "run_name": data.get("run_name", ""),
                "model_seed": data.get("model_seed", seed),
                "best_val_loss": data.get("best_val_loss"),
                "final_train_loss": data.get("final_train_loss"),
                "steps": [p["step"] for p in data["loss_curve"]],
                "val_losses": [p["val_loss"] for p in data["loss_curve"]],
            }

    if missing:
        print(f"WARNING: Missing {len(missing)} curves: {missing[:10]}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    n_fwd = len(all_curves["forward"])
    n_rev = len(all_curves["reversed"])
    print(f"Collected: {n_fwd} forward, {n_rev} reversed curves")

    # Save consolidated file
    output_path = os.path.join(args.out_dir, "all_curves.json")
    with open(output_path, "w") as f:
        json.dump(all_curves, f, indent=2)
    print(f"Saved consolidated curves to {output_path}")

    # Print summary statistics
    for direction in ["forward", "reversed"]:
        curves = all_curves[direction]
        if not curves:
            continue
        best_losses = [c["best_val_loss"] for c in curves.values()
                       if c["best_val_loss"] is not None]
        if best_losses:
            import numpy as np
            print(f"\n  {direction}: {len(best_losses)} runs")
            print(f"    Best val_loss: {np.mean(best_losses):.6f} "
                  f"+/- {np.std(best_losses):.6f}")
            print(f"    Range: [{np.min(best_losses):.6f}, {np.max(best_losses):.6f}]")


def main():
    args = parse_args()

    if args.mode == "local":
        run_local(args)
    elif args.mode == "slurm":
        generate_slurm(args)
    elif args.mode == "collect":
        collect_results(args)


if __name__ == "__main__":
    main()
