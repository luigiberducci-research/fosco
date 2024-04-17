import matplotlib as mpl
from plotly.graph_objs import Figure, Surface
import numpy as np

from fosco.common.domains import Rectangle, Sphere, Set, Union

FigureType = Figure | mpl.figure.Figure


def plot_rectangle(
        domain: Rectangle,
        fig: Figure,
        color: str = None,
        dim_select: tuple[int, int] = None,
        label: str = "",
) -> FigureType:
    """
    Plot the rectangle domain as surface in 3d figure.
    """
    assert isinstance(domain, Rectangle), "plot_rectangle only works for rectangles"

    i0, i1 = dim_select or (0, 1)
    x0 = domain.lower_bounds[i0]
    y0 = domain.lower_bounds[i1]
    x1 = domain.upper_bounds[i0]
    y1 = domain.upper_bounds[i1]

    bins = 10
    X = np.linspace(x0, x1, bins)
    Y = np.linspace(y0, y1, bins)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros((bins, bins))

    if isinstance(fig, mpl.figure.Figure):
        fig = plot_surface_mpl(X, Y, Z, fig, label, color)
    elif isinstance(fig, Figure):
        fig = plot_surface_plotly(X, Y, Z, fig, label, color)
    else:
        raise NotImplementedError(f"plot_sphere not implemented for {type(fig)}")

    return fig


def plot_sphere(
        domain: Sphere,
        fig: FigureType,
        color: str = None,
        dim_select: tuple[int, int] = None,
        label: str = "",
) -> FigureType:
    """
    Plot the sphere domain in 2d.
    """
    assert isinstance(domain, Sphere), "plot_sphere only works for spheres"

    i0, i1 = dim_select or (0, 1)
    x0 = domain.center[i0]
    y0 = domain.center[i1]
    radius = domain.radius

    resolution = 20  # lower resolution is faster but less accurate
    u, v = np.mgrid[0: 2 * np.pi: resolution * 2j, 0: np.pi: resolution * 1j]

    X = radius * np.cos(u) * np.sin(v) + x0
    Y = radius * np.sin(u) * np.sin(v) + y0
    Z = np.zeros(X.shape)

    if isinstance(fig, mpl.figure.Figure):
        fig = plot_surface_mpl(X, Y, Z, fig, label, color)
    elif isinstance(fig, Figure):
        fig = plot_surface_plotly(X, Y, Z, fig, label, color)
    else:
        raise NotImplementedError(f"plot_sphere not implemented for {type(fig)}")

    return fig


def plot_scattered_points(
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
        fig = scatter_points_mpl(X, Y, Z, fig, label, color)
    else:
        fig = scatter_points_plotly(X, Y, Z, fig, label, color)
    return fig


def plot_surface_plotly(xs, ys, zs, fig, label="", color=None) -> FigureType:
    if color:
        color = [[0.0, color], [1.0, color]]

    surface = Surface(
        x=xs, y=ys, z=zs, colorscale=color, showscale=False, name=label
    )
    fig.add_trace(surface)
    return fig


def plot_surface_mpl(xs, ys, zs, fig, label="", color=None) -> FigureType:
    fig.gca().plot_surface(xs, ys, zs, color=color, label=label, alpha=0.80)
    return fig

def scatter_points_plotly(xs, ys, zs, fig, label="", color=None) -> FigureType:
    fig.add_scatter3d(x=xs, y=ys, z=zs, mode="markers", marker=dict(size=1, color=color), name=label)
    return fig

def scatter_points_mpl(xs, ys, zs, fig, label="", color=None) -> FigureType:
    fig.gca().scatter(xs, ys, zs, color=color, label=label)
    return fig

def plot_domain(
        domain: Set,
        fig: FigureType,
        color: str,
        dim_select: tuple[int, int] = None,
        label: str = "",
) -> FigureType:
    """
    Plot the domain in 2d.
    """
    if isinstance(domain, Rectangle):
        fig = plot_rectangle(domain, fig, color, dim_select, label)
    elif isinstance(domain, Sphere):
        fig = plot_sphere(domain, fig, color, dim_select, label)
    elif isinstance(domain, Union):
        is_first = True
        for subdomain in domain.sets:
            label = label if is_first else None
            fig = plot_domain(subdomain, fig, color, dim_select, label)
            is_first = False
    elif hasattr(domain, "generate_data"):
        # not a conventional domain, plot scattered points
        fig = plot_scattered_points(domain, fig, color, dim_select, label)
    else:
        raise NotImplementedError(f"plot_domain not implemented for {type(domain)}")

    return fig


if __name__ == "__main__":
    import numpy as np
    from fosco.plotting.surface import plot_surface

    domain1 = Rectangle(vars=["x0", "x1", "x2"], lb=[0, 1, 0], ub=[2, 3, 4])
    domain2 = Sphere(vars=["x0", "x1", "x2"], center=[0, 1, 3], radius=1.333)

    fig = Figure()


    def func(x):
        assert len(x.shape) == 2 and x.shape[1] == 2, "x must be a batch of 2d points"
        return np.sin(x[:, 0]) + np.cos(x[:, 1])


    fig = plot_surface(
        func, (-5, 5), (-5, 5), levels=[0], label="surface", fig=fig, opacity=0.75
    )

    dim_select = (0, 1)
    fig = plot_domain(domain1, fig, color="red", dim_select=dim_select, label="domain1")
    fig = plot_domain(
        domain2, fig, color="blue", dim_select=dim_select, label="domain2"
    )

    # show legend and hide colorbar
    fig.update_traces(showlegend=True)
    fig.update_traces(showscale=False, selector=dict(type="surface"))

    fig.show()
