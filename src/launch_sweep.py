#!/usr/bin/env python3
"""
Launch WandB hyperparameter sweep for HMM transformer training.

Usage:
    # Create new sweep and launch agent
    python launch_sweep.py --config sweep_config.yaml --num_agents 1

    # Join existing sweep with multiple agents
    python launch_sweep.py --sweep_id abc123 --num_agents 4

    # Just create sweep and print command
    python launch_sweep.py --config sweep_config.yaml --print_command
"""

# sweep_id: txhmikge

import wandb
import yaml
import argparse
import subprocess
import os
from dotenv import load_dotenv


def create_sweep(config_file):
    """Create a new WandB sweep and return sweep ID."""
    load_dotenv()

    with open(config_file, 'r') as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(
        sweep_config,
        project="hidden-markov-model"
    )

    return sweep_id


def run_training():
    """Wrapper function called by WandB agent."""
    from train import train
    train("config/sweep_base.yaml")


def main():
    parser = argparse.ArgumentParser(description='Launch WandB sweep')
    parser.add_argument('--config', type=str, default='sweep_config.yaml',
                       help='Sweep configuration file')
    parser.add_argument('--sweep_id', type=str, default=None,
                       help='Existing sweep ID to join')
    parser.add_argument('--num_agents', type=int, default=1,
                       help='Number of parallel agents')
    parser.add_argument('--count', type=int, default=None,
                       help='Number of runs per agent (None = infinite)')
    parser.add_argument('--print_command', action='store_true',
                       help='Print command instead of launching')

    args = parser.parse_args()

    # Load environment for WandB
    load_dotenv()

    # Create or use existing sweep
    if args.sweep_id:
        sweep_id = args.sweep_id
        print(f"Joining existing sweep: {sweep_id}")
    else:
        print(f"Creating new sweep from: {args.config}")
        sweep_id = create_sweep(args.config)
        print(f"Created sweep: {sweep_id}")

    # Get WandB entity (username/org)
    api = wandb.Api()
    entity = api.default_entity or "chuqiao-lin-university-of-oxford"

    print(f"\nSweep URL: https://wandb.ai/{entity}/hidden-markov-model/sweeps/{sweep_id}")
    print(f"\nTo manually launch agents:")
    print(f"  wandb agent {entity}/hidden-markov-model/{sweep_id}")

    if args.print_command:
        return

    # Launch agent(s)
    print(f"\nLaunching {args.num_agents} agent(s)...")

    if args.num_agents == 1:
        # Run in current process
        wandb.agent(sweep_id, function=run_training, count=args.count, project="hidden-markov-model")
    else:
        # Launch multiple agents in parallel processes
        processes = []
        for i in range(args.num_agents):
            cmd = [
                'python', '-c',
                f"import wandb; from launch_sweep import run_training; "
                f"wandb.agent('{entity}/hidden-markov-model/{sweep_id}', function=run_training, count={args.count})"
            ]
            p = subprocess.Popen(cmd)
            processes.append(p)
            print(f"  Launched agent {i+1} (PID: {p.pid})")

        # Wait for all to complete
        for p in processes:
            p.wait()


if __name__ == "__main__":
    main()
