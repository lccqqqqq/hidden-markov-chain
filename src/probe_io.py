"""
Serialisation and loading for linear probe results.
"""

import os
import pickle
import numpy as np
import pandas as pd

from linear_probe import ProbeResult


def _prefix(run_id: str) -> str:
    return f"{run_id}_" if run_id else ""


def save_probe_results(
    results: list[ProbeResult],
    output_dir: str,
    run_id: str = "",
    save_regressors: bool = True,
    save_data: bool = False,
) -> None:
    """
    Save probe results to disk.

    Always saves:
        {prefix}metrics.csv — one row per layer with train/test MSE, R², baseline MSE

    Optionally saves:
        {prefix}regressors.pkl — list of fitted LinearRegression objects
        {prefix}data.npz — train/test predictions for each layer

    Args:
        results: List of ProbeResult (one per layer, or single-element for concatenated)
        output_dir: Directory to write into (created if needed)
        run_id: Optional prefix for filenames
        save_regressors: Whether to pickle the regressors
        save_data: Whether to save predictions (can be large)
    """
    os.makedirs(output_dir, exist_ok=True)
    prefix = _prefix(run_id)

    # Metrics CSV
    rows = []
    for i, r in enumerate(results):
        rows.append({
            "layer": i,
            "train_mse": r.train_mse,
            "test_mse": r.test_mse,
            "train_r2": r.train_r2,
            "test_r2": r.test_r2,
            "baseline_mse": r.baseline_mse,
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, f"{prefix}metrics.csv"), index=False)

    # Regressors
    if save_regressors:
        with open(os.path.join(output_dir, f"{prefix}regressors.pkl"), "wb") as f:
            pickle.dump([r.regressor for r in results], f)

    # Data (predictions)
    if save_data:
        data_dict = {}
        for i, r in enumerate(results):
            data_dict[f"layer_{i}_train_predictions"] = r.train_predictions
            data_dict[f"layer_{i}_test_predictions"] = r.test_predictions
        np.savez_compressed(
            os.path.join(output_dir, f"{prefix}data.npz"), **data_dict
        )


def load_probe_results(output_dir: str, run_id: str = "") -> dict:
    """
    Load saved probe results from disk.

    Returns:
        dict with keys:
            'metrics': DataFrame (always present if file exists)
            'regressors': list[LinearRegression] (if saved)
            'data': dict of numpy arrays (if saved)
    """
    prefix = _prefix(run_id)
    results = {}

    metrics_path = os.path.join(output_dir, f"{prefix}metrics.csv")
    if os.path.exists(metrics_path):
        results["metrics"] = pd.read_csv(metrics_path)

    regressors_path = os.path.join(output_dir, f"{prefix}regressors.pkl")
    if os.path.exists(regressors_path):
        with open(regressors_path, "rb") as f:
            results["regressors"] = pickle.load(f)

    data_path = os.path.join(output_dir, f"{prefix}data.npz")
    if os.path.exists(data_path):
        results["data"] = dict(np.load(data_path))

    return results
