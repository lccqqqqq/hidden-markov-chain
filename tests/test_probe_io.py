"""Tests for src/probe_io.py"""

import os
import numpy as np
import pytest

from linear_probe import ProbeResult, fit_probe_with_split
from probe_io import save_probe_results, load_probe_results


@pytest.fixture
def sample_results():
    """Generate a list of ProbeResults for testing I/O."""
    rng = np.random.default_rng(10)
    X = rng.random((100, 5))
    y = rng.random((100, 3))
    return [fit_probe_with_split(X, y, seed=10)]


class TestSaveLoadRoundtrip:
    def test_metrics_roundtrip(self, sample_results, tmp_path):
        """Saved and loaded metrics should match."""
        save_probe_results(sample_results, str(tmp_path), run_id="test")
        loaded = load_probe_results(str(tmp_path), run_id="test")

        assert "metrics" in loaded
        df = loaded["metrics"]
        assert len(df) == 1
        assert abs(df["test_r2"].iloc[0] - sample_results[0].test_r2) < 1e-10
        assert abs(df["train_mse"].iloc[0] - sample_results[0].train_mse) < 1e-10

    def test_regressors_roundtrip(self, sample_results, tmp_path):
        save_probe_results(sample_results, str(tmp_path), run_id="r1", save_regressors=True)
        loaded = load_probe_results(str(tmp_path), run_id="r1")
        assert "regressors" in loaded
        assert len(loaded["regressors"]) == 1

    def test_data_roundtrip(self, sample_results, tmp_path):
        save_probe_results(sample_results, str(tmp_path), run_id="r2", save_data=True)
        loaded = load_probe_results(str(tmp_path), run_id="r2")
        assert "data" in loaded
        assert "layer_0_test_predictions" in loaded["data"]

    def test_run_id_in_filenames(self, sample_results, tmp_path):
        save_probe_results(sample_results, str(tmp_path), run_id="myrun")
        files = os.listdir(tmp_path)
        assert any("myrun_" in f for f in files)

    def test_no_run_id(self, sample_results, tmp_path):
        """Empty run_id should produce filenames without prefix."""
        save_probe_results(sample_results, str(tmp_path), run_id="")
        files = os.listdir(tmp_path)
        assert "metrics.csv" in files
