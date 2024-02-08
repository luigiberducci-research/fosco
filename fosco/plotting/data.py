import numpy as np
import torch
import plotly.graph_objects as go
from plotly.graph_objs import graph_objs


def scatter_data(
    data: torch.Tensor | np.ndarray,
    fig: go.Figure,
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

    trace1 = graph_objs.Scatter(
        x=x,
        y=y,
        mode="markers",
        name=name,
        marker=dict(color=color, size=point_size, opacity=opacity),
    )
    """
    ncontours = 10
    colorscale = ['#7A4579', '#D56073', 'rgb(236,158,105)', (1, 1, 0.2), (0.98, 0.98, 0.98)]
    colorscale = clrs.validate_colors(colorscale, "rgb")
    colorscale = make_linear_colorscale(colorscale)
    
    trace2 = graph_objs.Histogram2dContour(
        x=x,
        y=y,
        name="density",
        ncontours=ncontours,
        reversescale=True,
        showscale=False,
        colorscale=colorscale,
        opacity=opacity,
    )
    """

    for trace in [trace1]:
        fig.add_trace(trace)

    return fig


def scatter_datasets(
    datasets: dict[str, np.ndarray | torch.Tensor],
    counter_examples: dict[str, np.ndarray | torch.Tensor],
) -> go.Figure:
    fig = go.Figure()

    colors = ["blue", "red", "green", "purple"]
    for i, (name, data) in enumerate(datasets.items()):
        color = colors[i % len(colors)]
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
            if len(data) == 0:
                continue
            color = colors[i % len(colors)]
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
