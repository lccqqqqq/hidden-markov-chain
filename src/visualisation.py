"""
Simplex plotting utilities for visualising belief states and probe predictions.

Requires matplotlib (optional dependency for the main pipeline).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def barycentric_to_cartesian_2d(simplex_coords: np.ndarray) -> np.ndarray:
    """
    Convert 3D barycentric (simplex) coordinates to 2D Cartesian coordinates.

    Maps points on a 3-simplex (coordinates sum to 1) to an equilateral triangle
    with vertices at (0,0), (1,0), (0.5, sqrt(3)/2).

    Args:
        simplex_coords: (n, 3) array where each row sums to 1

    Returns:
        (n, 2) array of 2D Cartesian positions
    """
    v0 = np.array([0.0, 0.0])
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3) / 2])

    p0 = simplex_coords[:, 0:1]
    p1 = simplex_coords[:, 1:2]
    p2 = simplex_coords[:, 2:3]

    return p0 * v0 + p1 * v1 + p2 * v2


def plot_simplex(
    points: np.ndarray,
    labels: np.ndarray | None = None,
    title: str = "3-Simplex Visualization",
    vertex_labels: list[str] | None = None,
    alpha: float = 0.5,
    s: float = 1,
    figsize: tuple = (10, 10),
    show_grid: bool = True,
    ax: Axes | None = None,
    save_path: str | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot points on a 3-simplex (equilateral triangle).

    Args:
        points: (n, 3) barycentric coordinates (each row sums to 1)
        labels: Optional (n,) array for colouring points
        title: Plot title
        vertex_labels: Labels for the three vertices (default: State 0/1/2)
        alpha: Point transparency
        s: Point size
        figsize: Figure size (ignored if ax is provided)
        show_grid: Whether to show simplex grid lines
        ax: Optional existing Axes for subplot support
        save_path: Optional path to save the figure

    Returns:
        (Figure, Axes) tuple
    """
    if vertex_labels is None:
        vertex_labels = ["State 0", "State 1", "State 2"]

    coords_2d = barycentric_to_cartesian_2d(points)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Draw triangle boundary
    triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
    ax.plot(triangle[:, 0], triangle[:, 1], "k-", linewidth=2)

    # Vertex labels
    ax.text(-0.05, -0.05, vertex_labels[0], fontsize=12, ha="right")
    ax.text(1.05, -0.05, vertex_labels[1], fontsize=12, ha="left")
    ax.text(0.5, np.sqrt(3) / 2 + 0.05, vertex_labels[2], fontsize=12, ha="center")

    # Grid lines
    if show_grid:
        n_lines = 10
        for i in range(1, n_lines):
            t = i / n_lines
            for p_start, p_end in [
                (np.array([[t, 0, 1 - t]]), np.array([[0, t, 1 - t]])),
                (np.array([[t, 1 - t, 0]]), np.array([[0, 1 - t, t]])),
                (np.array([[1 - t, t, 0]]), np.array([[1 - t, 0, t]])),
            ]:
                c_start = barycentric_to_cartesian_2d(p_start)[0]
                c_end = barycentric_to_cartesian_2d(p_end)[0]
                ax.plot(
                    [c_start[0], c_end[0]],
                    [c_start[1], c_end[1]],
                    "k-",
                    alpha=0.1,
                    linewidth=0.5,
                )

    # Scatter
    if labels is not None:
        scatter = ax.scatter(
            coords_2d[:, 0], coords_2d[:, 1], c=labels, s=s, alpha=alpha, cmap="viridis"
        )
        plt.colorbar(scatter, ax=ax, label="Label")
    else:
        ax.scatter(coords_2d[:, 0], coords_2d[:, 1], s=s, alpha=alpha, color="blue")

    ax.set_aspect("equal")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, np.sqrt(3) / 2 + 0.1)
    ax.set_title(title, fontsize=14)
    ax.axis("off")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_probe_comparison(
    true_beliefs: np.ndarray,
    predicted_beliefs: np.ndarray,
    title: str = "True vs Predicted Belief States",
    vertex_labels: list[str] | None = None,
    figsize: tuple = (20, 10),
    save_path: str | None = None,
    **scatter_kwargs,
) -> tuple[Figure, list[Axes]]:
    """
    Side-by-side simplex plots of true and predicted belief states.

    Args:
        true_beliefs: (n, 3) ground-truth barycentric coordinates
        predicted_beliefs: (n, 3) predicted barycentric coordinates
        title: Overall figure title
        vertex_labels: Labels for simplex vertices
        figsize: Figure size
        save_path: Optional path to save the figure
        **scatter_kwargs: Passed to plot_simplex (alpha, s, etc.)

    Returns:
        (Figure, [Axes, Axes]) tuple
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    plot_simplex(
        true_beliefs,
        title="True Beliefs",
        vertex_labels=vertex_labels,
        ax=axes[0],
        **scatter_kwargs,
    )
    plot_simplex(
        predicted_beliefs,
        title="Predicted Beliefs",
        vertex_labels=vertex_labels,
        ax=axes[1],
        **scatter_kwargs,
    )

    fig.suptitle(title, fontsize=16)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, list(axes)
