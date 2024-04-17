from typing import Callable

import numpy as np

import matplotlib as mpl

import plotly.graph_objects as go
from plotly.graph_objs import Surface

from fosco.common.domains import Set
from fosco.plotting.constants import FigureType


def plot_surface(
    func: Callable[[np.ndarray], np.ndarray],
    xrange: tuple[float, float],
    yrange: tuple[float, float],
    levels: list[float] = None,
    label: str = "",
    bins: int = 100,
    level_color: str = "white",
    opacity: float = 1.0,
    fig: go.Figure = None,
):
    """
    Plot the surface of the function over 2d space.
    """
    levels = levels or []

    x = np.linspace(xrange[0], xrange[1], bins)
    y = np.linspace(yrange[0], yrange[1], bins)
    X, Y = np.meshgrid(x, y)
    Xflat = X.reshape(-1, 1)
    Yflat = Y.reshape(-1, 1)
    inputs = np.hstack([Xflat, Yflat])
    z = func(inputs).reshape(bins, bins)

    if fig is None:
        fig = go.Figure(data=[go.Surface(x=x, y=y, z=z)])
    else:
        fig.add_trace(go.Surface(x=x, y=y, z=z, opacity=opacity, name=label))

    for level in levels:
        small_sz = 0.01
        fig.update_traces(
            contours_z=dict(
                show=True,
                color=level_color,
                highlightcolor="limegreen",
                project_z=False,
                start=level,
                end=level + small_sz,
                size=small_sz,
            )
        )

    return fig


def plot_scattered_points3d(
    domain: Set,
    fig: FigureType,
    color: str,
    dim_select: tuple[int, int] = None,
    label: str = "",
) -> FigureType:
    data = domain.generate_data(500)
    dim_select = dim_select or (0, 1)

    X = data[:, dim_select[0]]
    Y = data[:, dim_select[1]]
    Z = np.zeros_like(X)

    if isinstance(fig, mpl.figure.Figure):
        fig = scatter_points3d_mpl(X, Y, Z, fig, label, color)
    else:
        fig = scatter_points3d_plotly(X, Y, Z, fig, label, color)
    return fig


def plot_surface3d_plotly(xs, ys, zs, fig, label="", color=None) -> FigureType:
    if color:
        color = [[0.0, color], [1.0, color]]

    surface = Surface(x=xs, y=ys, z=zs, colorscale=color, showscale=False, name=label)
    fig.add_trace(surface)
    return fig


def plot_surface3d_mpl(xs, ys, zs, fig, label="", color=None) -> FigureType:
    fig.gca().plot_surface(xs, ys, zs, color=color, label=label, alpha=0.80)
    return fig


def scatter_points3d_plotly(xs, ys, zs, fig, label="", color=None) -> FigureType:
    fig.add_scatter3d(
        x=xs, y=ys, z=zs, mode="markers", marker=dict(size=1, color=color), name=label
    )
    return fig


def scatter_points3d_mpl(xs, ys, zs, fig, label="", color=None) -> FigureType:
    fig.gca().scatter(xs, ys, zs, color=color, label=label)
    return fig


if __name__ == "__main__":

    def func(x):
        assert len(x.shape) == 2 and x.shape[1] == 2, "x must be a batch of 2d points"
        return np.sin(x[:, 0])  # + np.cos(x[:, 1])

    fig = plot_surface(func, (-10, 10), (-5, 5), levels=[0], label="test")

    fig.show()
