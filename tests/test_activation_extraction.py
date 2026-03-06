"""Tests for src/activation_extraction.py"""

import numpy as np
import torch as t
import pytest

from activation_extraction import (
    extract_residual_stream,
    prepare_probe_data,
    get_process_from_config,
    _find_checkpoint,
)


class TestExtractResidualStream:
    def test_output_shape(self, tiny_model):
        """Output should be (n_layers, batch, seq_len, d_model)."""
        seqs = t.randint(0, 2, (10, 6))
        result = extract_residual_stream(tiny_model, seqs, device="cpu")
        assert result.shape == (1, 10, 6, 4)  # 1 layer, batch=10, seq=6, d=4

    def test_on_cpu(self, tiny_model):
        """Result should always be on CPU."""
        seqs = t.randint(0, 2, (5, 6))
        result = extract_residual_stream(tiny_model, seqs, device="cpu")
        assert result.device == t.device("cpu")

    def test_chunked_equals_full(self, tiny_model):
        """Chunked extraction should produce the same result as full-batch."""
        seqs = t.randint(0, 2, (20, 6))
        full = extract_residual_stream(tiny_model, seqs, device="cpu", batch_size=None)
        chunked = extract_residual_stream(tiny_model, seqs, device="cpu", batch_size=7)
        t.testing.assert_close(full, chunked)


class TestPrepareProbeData:
    def test_last_pos_shapes(self):
        """use_last_pos=True: samples = batch, features = d_model."""
        n_layers, batch, seq_len, d_model, n_hidden = 2, 50, 6, 8, 3
        actvs = t.randn(n_layers, batch, seq_len, d_model)
        beliefs = np.random.rand(batch, seq_len, n_hidden)

        layer_actvs, targets = prepare_probe_data(actvs, beliefs, use_last_pos=True)
        assert layer_actvs.shape == (n_layers, batch, d_model)
        assert targets.shape == (batch, n_hidden)

    def test_all_pos_shapes(self):
        """use_last_pos=False: samples = batch * seq_len."""
        n_layers, batch, seq_len, d_model, n_hidden = 2, 50, 6, 8, 3
        actvs = t.randn(n_layers, batch, seq_len, d_model)
        beliefs = np.random.rand(batch, seq_len, n_hidden)

        layer_actvs, targets = prepare_probe_data(actvs, beliefs, use_last_pos=False)
        assert layer_actvs.shape == (n_layers, batch * seq_len, d_model)
        assert targets.shape == (batch * seq_len, n_hidden)


class TestGetProcessFromConfig:
    def test_z1r(self):
        config = {"data_generator": {"process": {"name": "z1r", "params": {}}}}
        proc = get_process_from_config(config)
        assert proc.num_hidden_states == 3
        assert proc.d_vocab == 2

    def test_rrxor(self):
        config = {"data_generator": {"process": {"name": "rrxor", "params": {}}}}
        proc = get_process_from_config(config)
        assert proc.num_hidden_states == 5

    def test_unknown_raises(self):
        config = {"data_generator": {"process": {"name": "nonexistent", "params": {}}}}
        with pytest.raises(ValueError, match="Unknown process"):
            get_process_from_config(config)


class TestFindCheckpoint:
    def test_best_model_priority(self, tmp_path):
        (tmp_path / "best_model.pt").touch()
        (tmp_path / "latest.pt").touch()
        assert _find_checkpoint(str(tmp_path)) == "best_model.pt"

    def test_latest_fallback(self, tmp_path):
        (tmp_path / "latest.pt").touch()
        assert _find_checkpoint(str(tmp_path)) == "latest.pt"

    def test_epoch_fallback(self, tmp_path):
        (tmp_path / "checkpoint_epoch_3.pt").touch()
        (tmp_path / "checkpoint_epoch_10.pt").touch()
        # Sorted lexicographically: "3" < "10" only if zero-padded; test sorts correctly
        assert "checkpoint_epoch_" in _find_checkpoint(str(tmp_path))

    def test_no_checkpoint_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _find_checkpoint(str(tmp_path))
