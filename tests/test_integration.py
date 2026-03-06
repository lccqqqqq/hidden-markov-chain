"""
End-to-end integration test: Z1R → tiny model → extract → probe → save → load.
"""

import os
import numpy as np
import torch as t
import pytest

from activation_extraction import extract_residual_stream, prepare_probe_data
from linear_probe import probe_layer_wise, probe_concatenated
from probe_io import save_probe_results, load_probe_results


def test_end_to_end(tiny_model, z1r_data, tmp_path):
    """Full pipeline: generate data, extract activations, probe, save, load."""
    sequences = z1r_data["sequences"]
    beliefs = z1r_data["belief_states"]

    # Extract activations
    actvs = extract_residual_stream(tiny_model, sequences, device="cpu")
    assert actvs.shape[0] == 1  # 1 layer
    assert actvs.shape[1] == 200  # batch
    assert actvs.shape[2] == 6  # seq_len

    # Prepare probe data
    layer_actvs, targets = prepare_probe_data(actvs, beliefs, use_last_pos=True)
    assert layer_actvs.shape == (1, 200, 4)
    assert targets.shape == (200, 3)

    # Run layer-wise probe
    results = probe_layer_wise(layer_actvs, targets, test_fraction=0.2, seed=99)
    assert len(results) == 1
    r = results[0]
    assert isinstance(r.test_r2, float)
    assert np.isfinite(r.test_r2)
    assert r.train_predictions.shape[1] == 3
    assert r.test_predictions.shape[1] == 3

    # Run concatenated probe
    concat_result = probe_concatenated(layer_actvs, targets, test_fraction=0.2, seed=99)
    assert isinstance(concat_result.test_r2, float)

    # Save and reload
    output_dir = str(tmp_path / "probe_results")
    save_probe_results(results, output_dir, run_id="integration", save_regressors=True, save_data=True)

    assert os.path.exists(os.path.join(output_dir, "integration_metrics.csv"))
    assert os.path.exists(os.path.join(output_dir, "integration_regressors.pkl"))
    assert os.path.exists(os.path.join(output_dir, "integration_data.npz"))

    loaded = load_probe_results(output_dir, run_id="integration")
    assert "metrics" in loaded
    assert abs(loaded["metrics"]["test_r2"].iloc[0] - r.test_r2) < 1e-10
