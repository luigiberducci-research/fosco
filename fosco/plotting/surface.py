from typing import Callable

import numpy as np

import plotly.graph_objects as go

from fosco.common.domains import Set


def plot_surface(
    func: Callable[[np.ndarray], np.ndarray],
    xrange: tuple[float, float],
    yrange: tuple[float, float],
    levels: list[float] = [],
    label: str = "",
    bins: int = 100,
    level_color: str = "white",
    opacity: float = 1.0,
    fig: go.Figure = None,
):
    """
    Plot the surface of the function over 2d space.
    """
    # Read data from a csv

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


if __name__ == "__main__":

    def func(x):
        assert len(x.shape) == 2 and x.shape[1] == 2, "x must be a batch of 2d points"
        return np.sin(x[:, 0]) #+ np.cos(x[:, 1])

    fig = plot_surface(func, (-10, 10), (-5, 5), levels=[0], label="test")

    fig.show()
