"""
Enhanced sweep evaluation and analysis.

Usage:
    python sweep_eval.py --sweep_id abc123 --output_dir sweep_results/
    python sweep_eval.py --project hmm-sweep --sweep_name mess3-v1
"""

import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse
import yaml
import os


def load_data(data_dir: str):
    """
    Load the log, statistics, and metadata as well
    from timestamped directory
    """
    metadata = yaml.safe_load(open(os.path.join(data_dir, "config.yaml")))
    data = pd.read_parquet(os.path.join(data_dir, "training_metrics.parquet"))

    return metadata, data


def fetch_sweep_runs(sweep_id=None, project="hmm-sweep", sweep_name=None):
    """
    Fetch all runs from a WandB sweep.

    Args:
        sweep_id: WandB sweep ID
        project: WandB project name
        sweep_name: Alternative to sweep_id, find by name

    Returns:
        List of wandb.Run objects
    """
    api = wandb.Api()

    if sweep_id:
        sweep = api.sweep(f"{project}/{sweep_id}")
    elif sweep_name:
        # Find sweep by name
        sweeps = api.sweeps(project)
        sweep = next((s for s in sweeps if s.name == sweep_name), None)
        if sweep is None:
            raise ValueError(f"Sweep '{sweep_name}' not found in project '{project}'")
    else:
        raise ValueError("Must provide either sweep_id or sweep_name")

    return sweep.runs


def extract_run_data(run):
    """
    Extract relevant data from a single run.

    Args:
        run: wandb.Run object

    Returns:
        Dict with config, summary metrics, and history
    """
    # Config parameters
    config = {
        'run_id': run.id,
        'run_name': run.name,
        'state': run.state,  # finished, failed, running, crashed
    }

    # Add all config parameters
    for k, v in run.config.items():
        config[k] = v

    # Summary metrics (final values)
    summary = {
        'final_loss': run.summary.get('loss'),
        'final_val_loss': run.summary.get('val_loss'),
        'final_geometry_r2': run.summary.get('validation/geometry/mean_r2'),
        'final_llr_mean': run.summary.get('validation/llr_mean'),
        'final_llr_std': run.summary.get('validation/llr_std'),
    }

    # History (time series) - only if needed
    try:
        history = run.history(keys=[
            'step', 'loss', 'val_loss',
            'validation/geometry/mean_r2',
            'validation/llr_mean'
        ], samples=1000)  # Limit samples for efficiency
    except:
        history = pd.DataFrame()

    return {
        'config': config,
        'summary': summary,
        'history': history
    }


def aggregate_sweep_results(runs):
    """
    Aggregate results from all runs into DataFrames.

    Args:
        runs: List of wandb.Run objects

    Returns:
        Tuple of (summary_df, completed_df)
    """
    data = [extract_run_data(run) for run in runs]

    # Create summary DataFrame
    summary_records = []
    for d in data:
        record = {**d['config'], **d['summary']}
        summary_records.append(record)

    summary_df = pd.DataFrame(summary_records)

    # Filter to completed runs
    completed_df = summary_df[summary_df['state'] == 'finished'].copy()

    return summary_df, completed_df


def plot_pareto_frontier(df, x_metric='final_val_loss', y_metric='final_geometry_r2',
                        output_path=None):
    """
    Plot Pareto frontier of two objectives.

    Shows trade-off between validation loss and geometry quality.
    """
    # Remove NaN values
    plot_df = df[[x_metric, y_metric]].dropna()

    if len(plot_df) == 0:
        print("Warning: No valid data for Pareto frontier plot")
        return None, None

    # Find Pareto optimal points (minimize x, maximize y)
    pareto_mask = np.ones(len(plot_df), dtype=bool)
    for i in range(len(plot_df)):
        if pareto_mask[i]:
            # Check if any other point dominates
            x_i, y_i = plot_df.iloc[i][x_metric], plot_df.iloc[i][y_metric]
            dominates = (plot_df[x_metric] <= x_i) & (plot_df[y_metric] >= y_i)
            dominates.iloc[i] = False  # Don't compare to self
            if dominates.any():
                pareto_mask[i] = False

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # All points
    ax.scatter(plot_df[x_metric], plot_df[y_metric],
              alpha=0.5, s=50, label='All runs')

    # Pareto optimal points
    pareto_points = plot_df[pareto_mask]
    if len(pareto_points) > 0:
        ax.scatter(pareto_points[x_metric], pareto_points[y_metric],
                  color='red', s=100, marker='*', label='Pareto optimal', zorder=10)

    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    ax.set_title('Pareto Frontier: Multi-Objective Optimization')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved Pareto frontier to {output_path}")

    return fig, pareto_points


def plot_parameter_importance(df, metric='final_val_loss', output_path=None):
    """
    Analyze which parameters most affect the target metric.
    """
    # Select numeric config columns
    config_cols = [c for c in df.columns if c in [
        'n_layer', 'n_embd', 'n_head', 'd_head', 'n_inner',
        'n_ctx', 'learning_rate', 'batch_size'
    ]]

    if len(config_cols) == 0:
        print("Warning: No numeric config columns found")
        return None, None

    # Remove rows with NaN in metric
    clean_df = df[config_cols + [metric]].dropna()

    if len(clean_df) == 0:
        print("Warning: No valid data for parameter importance")
        return None, None

    # Compute correlations
    correlations = clean_df[config_cols + [metric]].corr()[metric].drop(metric)
    correlations = correlations.abs().sort_values(ascending=False)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    correlations.plot(kind='barh', ax=ax)
    ax.set_xlabel(f'Absolute Correlation with {metric}')
    ax.set_title('Parameter Importance Analysis')
    ax.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved parameter importance to {output_path}")

    return fig, correlations


def generate_sweep_report(sweep_id=None, sweep_name=None, project="hmm-sweep",
                         output_dir="sweep_results"):
    """
    Generate comprehensive sweep analysis report.

    Args:
        sweep_id: WandB sweep ID
        sweep_name: Alternative to sweep_id
        project: WandB project name
        output_dir: Output directory for results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Fetching sweep runs from project: {project}")
    runs = fetch_sweep_runs(sweep_id=sweep_id, sweep_name=sweep_name, project=project)
    print(f"Found {len(runs)} runs")

    print("Aggregating results...")
    summary_df, completed_df = aggregate_sweep_results(runs)

    # Save data
    summary_df.to_csv(output_dir / "sweep_summary.csv", index=False)
    completed_df.to_csv(output_dir / "completed_runs.csv", index=False)
    print(f"Saved data to {output_dir}")

    # Generate plots
    print("Generating visualizations...")

    # 1. Pareto frontier
    plot_pareto_frontier(
        completed_df,
        output_path=output_dir / "pareto_frontier.png"
    )

    # 2. Parameter importance
    plot_parameter_importance(
        completed_df,
        output_path=output_dir / "parameter_importance.png"
    )

    # 3. Best runs summary
    if len(completed_df) > 0:
        best_by_loss = completed_df.nsmallest(min(10, len(completed_df)), 'final_val_loss')
        best_by_geometry = completed_df.nlargest(min(10, len(completed_df)), 'final_geometry_r2')

        with open(output_dir / "best_runs.txt", 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("TOP RUNS BY VALIDATION LOSS\n")
            f.write("=" * 80 + "\n")
            f.write(best_by_loss.to_string())
            f.write("\n\n")

            f.write("=" * 80 + "\n")
            f.write("TOP RUNS BY GEOMETRY R²\n")
            f.write("=" * 80 + "\n")
            f.write(best_by_geometry.to_string())

        print(f"Analysis complete! Results saved to {output_dir}")
        print(f"\nBest run (by val_loss):")
        if len(best_by_loss) > 0:
            best_run = best_by_loss.iloc[0]
            print(f"  Run ID: {best_run['run_id']}")
            print(f"  Val Loss: {best_run['final_val_loss']:.6f}")
            print(f"  Geometry R²: {best_run.get('final_geometry_r2', 'N/A')}")
            print(f"  Config: n_layer={best_run.get('n_layer')}, "
                  f"n_embd={best_run.get('n_embd')}, "
                  f"lr={best_run.get('learning_rate')}")
    else:
        print("No completed runs found")


def main():
    parser = argparse.ArgumentParser(description='Analyze WandB sweep results')
    parser.add_argument('--sweep_id', type=str, default=None,
                       help='WandB sweep ID')
    parser.add_argument('--sweep_name', type=str, default=None,
                       help='WandB sweep name (alternative to sweep_id)')
    parser.add_argument('--project', type=str, default='hmm-sweep',
                       help='WandB project name')
    parser.add_argument('--output_dir', type=str, default='sweep_results',
                       help='Output directory for results')

    args = parser.parse_args()

    if not args.sweep_id and not args.sweep_name:
        parser.error("Must provide either --sweep_id or --sweep_name")

    generate_sweep_report(
        sweep_id=args.sweep_id,
        sweep_name=args.sweep_name,
        project=args.project,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
