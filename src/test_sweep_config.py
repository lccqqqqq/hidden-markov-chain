#!/usr/bin/env python3
"""
Test script to verify sweep configuration is being applied correctly.
Checks recent run configs to see if parameters vary.
"""

import os
import yaml
from pathlib import Path

def analyze_sweep_configs():
    """Analyze configs from recent runs to verify sweep is working."""

    records_dir = Path("records")

    # Find all config files
    config_files = sorted(records_dir.glob("*/config.yaml"),
                         key=lambda p: p.parent.name,
                         reverse=True)

    if not config_files:
        print("No config files found in records/")
        return

    print(f"Found {len(config_files)} run configs\n")
    print("="*80)
    print("Checking parameter variation across runs:")
    print("="*80)

    # Collect parameters from all configs
    all_params = []
    for config_file in config_files[:10]:  # Check last 10 runs
        with open(config_file) as f:
            config = yaml.safe_load(f)

        run_name = config_file.parent.name
        params = {
            'run': run_name,
            'n_layer': config['model'].get('n_layer'),
            'n_embd': config['model'].get('n_embd'),
            'n_head': config['model'].get('n_head'),
            'd_head': config['model'].get('d_head'),
            'n_inner': config['model'].get('n_inner'),
            'n_ctx': config['model'].get('n_ctx'),
            'attn_only': config['model'].get('attn_only'),
            'learning_rate': config['train'].get('learning_rate'),
        }
        all_params.append(params)

    # Print results
    header = f"{'Run':<25} {'n_layer':<8} {'n_embd':<8} {'n_head':<8} {'d_head':<8} {'n_inner':<8} {'n_ctx':<8} {'attn':<8} {'LR':<10}"
    print(header)
    print("-" * len(header))

    for params in all_params:
        print(f"{params['run']:<25} "
              f"{params['n_layer']:<8} "
              f"{params['n_embd']:<8} "
              f"{params['n_head']:<8} "
              f"{params['d_head']:<8} "
              f"{params['n_inner']:<8} "
              f"{params['n_ctx']:<8} "
              f"{str(params['attn_only']):<8} "
              f"{params['learning_rate']:<10}")

    print("\n" + "="*80)

    # Check for variation
    unique_values = {}
    for key in ['n_layer', 'n_embd', 'n_head', 'd_head', 'n_inner', 'n_ctx', 'attn_only', 'learning_rate']:
        values = set(p[key] for p in all_params)
        unique_values[key] = len(values)

    print("Parameter variation:")
    for key, count in unique_values.items():
        status = "✓ VARYING" if count > 1 else "✗ FIXED"
        print(f"  {key:<15}: {count} unique value(s) - {status}")

    print("\n" + "="*80)

    if all(count == 1 for count in unique_values.values()):
        print("⚠️  WARNING: All parameters are identical across runs!")
        print("    Sweep configuration may not be working correctly.")
        print("\nDEBUG: Check if wandb.run is initialized when train() is called.")
        print("       Run logs should show 'Detected WandB sweep - applying configuration overrides...'")
    else:
        print("✓ Sweep appears to be working - parameters vary across runs")

if __name__ == "__main__":
    analyze_sweep_configs()
