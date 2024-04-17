from typing import Optional

import matplotlib
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.graph_objs import graph_objs

from fosco.plotting.constants import FigureType, DOMAIN_COLORS


def scatter_data(
    data: torch.Tensor | np.ndarray,
    fig: FigureType,
    dim_select: tuple[int, int] = None,
    color: str = "blue",
    opacity: float = 1.0,
    point_size: int = 1,
    name: str = None,
):
    """
    Plot the data points as a scatter plot.
    """
    dim_select = dim_select or (0, 1)

    x = data[:, dim_select[0]].numpy()
    y = data[:, dim_select[1]].numpy()

    if isinstance(fig, go.Figure):
        trace = graph_objs.Scatter(
            x=x,
            y=y,
            mode="markers",
            name=name,
            marker=dict(color=color, size=point_size, opacity=opacity),
        )
        fig.add_trace(trace)
    elif isinstance(fig, matplotlib.figure.Figure):
        fig.gca().scatter(x=x, y=y, color=color, s=point_size, alpha=opacity)

    return fig


def scatter_datasets(
    datasets: dict[str, np.ndarray | torch.Tensor],
    counter_examples: dict[str, np.ndarray | torch.Tensor],
    fig: Optional[FigureType] = None,
) -> FigureType:
    fig = fig or go.Figure()

    for i, (name, data) in enumerate(datasets.items()):
        if name not in DOMAIN_COLORS:
            continue
        color = DOMAIN_COLORS[name]
        fig = scatter_data(
            data=data,
            fig=fig,
            color=color,
            dim_select=None,
            name=name,
            point_size=3,
            opacity=0.5,
        )

    if counter_examples is not None:
        for i, (name, data) in enumerate(counter_examples.items()):
            if name not in DOMAIN_COLORS:
                continue
            if data is None:
                continue
            color = DOMAIN_COLORS[name]
            name = f"{name} - counterexamples"
            fig = scatter_data(
                data=data,
                fig=fig,
                color=color,
                dim_select=None,
                name=name,
                point_size=5,
                opacity=1.0,
            )

    return fig


if __name__ == "__main__":
    data1 = torch.randn((100, 3))
    data2 = torch.randn((100, 3)) + 5

    dim_select = None
    fig = go.Figure()

    fig = scatter_data(data1, fig, color="red", dim_select=dim_select, name="data1")
    fig = scatter_data(data2, fig, color="blue", dim_select=dim_select, name="data2")

    fig.show()
