"""Tests for src/visualisation.py"""

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI

from visualisation import barycentric_to_cartesian_2d, plot_simplex, plot_probe_comparison


class TestBarycentricToCartesian:
    def test_vertices(self):
        """Simplex vertices should map to correct triangle corners."""
        vertices = np.array([
            [1, 0, 0],  # -> (0, 0)
            [0, 1, 0],  # -> (1, 0)
            [0, 0, 1],  # -> (0.5, sqrt(3)/2)
        ], dtype=float)
        result = barycentric_to_cartesian_2d(vertices)
        expected = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, np.sqrt(3) / 2],
        ])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_centroid(self):
        """Centroid [1/3, 1/3, 1/3] maps to triangle centroid."""
        centroid = np.array([[1 / 3, 1 / 3, 1 / 3]])
        result = barycentric_to_cartesian_2d(centroid)
        expected = np.array([[0.5, np.sqrt(3) / 6]])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_output_shape(self):
        coords = np.random.dirichlet([1, 1, 1], size=50)
        result = barycentric_to_cartesian_2d(coords)
        assert result.shape == (50, 2)


class TestPlotSimplex:
    def test_smoke_no_labels(self):
        """plot_simplex should not raise with basic input."""
        points = np.random.dirichlet([1, 1, 1], size=20)
        fig, ax = plot_simplex(points)
        assert fig is not None

    def test_smoke_with_labels(self):
        points = np.random.dirichlet([1, 1, 1], size=20)
        labels = np.arange(20)
        fig, ax = plot_simplex(points, labels=labels)
        assert fig is not None

    def test_custom_vertex_labels(self):
        points = np.random.dirichlet([1, 1, 1], size=10)
        fig, ax = plot_simplex(points, vertex_labels=["A", "B", "C"])
        assert fig is not None


class TestPlotProbeComparison:
    def test_smoke(self):
        true = np.random.dirichlet([1, 1, 1], size=30)
        pred = np.random.dirichlet([1, 1, 1], size=30)
        fig, axes = plot_probe_comparison(true, pred)
        assert len(axes) == 2
