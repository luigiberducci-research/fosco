import matplotlib as mpl
import numpy as np
from plotly.graph_objs import Figure

from fosco.common.domains import Rectangle, Sphere, Set, Union
from fosco.plotting.constants import FigureType, DOMAIN_COLORS
from fosco.plotting.utils3d import (
    plot_surface,
    plot_scattered_points3d,
    plot_surface3d_mpl,
    plot_surface3d_plotly,
)


def plot_state_domains(
    domains: dict[str, Set],
    fig: FigureType,
    dim_select: tuple[int, int] = None,
    opacity: float = 1.0,
) -> FigureType:
    for dname, domain in domains.items():
        if dname not in DOMAIN_COLORS:
            continue
        color = DOMAIN_COLORS[dname]
        fig = plot_domain(
            domain=domain,
            fig=fig,
            color=color,
            dim_select=dim_select,
            label=dname,
            opacity=opacity,
        )

    if isinstance(fig, mpl.figure.Figure):
        fig.gca().legend()
    elif isinstance(fig, Figure):
        fig.update_traces(showlegend=True)
    else:
        raise NotImplementedError(f"plot_state_domains not implemented for {type(fig)}")

    return fig


def plot_domain(
    domain: Set,
    fig: FigureType,
    color: str,
    opacity: float = 1.0,
    dim_select: tuple[int, int] = None,
    label: str = "",
    z_start: float = 0.0,
) -> FigureType:
    """
    Plot the domain in 2d.
    """
    if isinstance(domain, Rectangle):
        fig = plot_rectangle(
            domain=domain,
            fig=fig,
            color=color,
            opacity=opacity,
            dim_select=dim_select,
            label=label,
            z_start=z_start,
        )
    elif isinstance(domain, Sphere):
        fig = plot_sphere(
            domain=domain,
            fig=fig,
            color=color,
            opacity=opacity,
            dim_select=dim_select,
            label=label,
            z_start=z_start,
        )
    elif isinstance(domain, Union):
        is_first = True
        for subdomain in domain.sets:
            label = label if is_first else None
            fig = plot_domain(
                domain=subdomain,
                fig=fig,
                color=color,
                opacity=opacity,
                dim_select=dim_select,
                label=label,
                z_start=z_start,
            )
            is_first = False
    elif hasattr(domain, "generate_data"):
        # not a conventional domain, plot scattered points
        fig = plot_scattered_points3d(
            domain=domain,
            fig=fig,
            color=color,
            opacity=opacity,
            dim_select=dim_select,
            label=label,
            z_start=z_start,
        )
    else:
        raise NotImplementedError(f"plot_domain not implemented for {type(domain)}")

    return fig


def plot_rectangle(
    domain: Rectangle,
    fig: FigureType,
    color: str = None,
    opacity: float = 1.0,
    dim_select: tuple[int, int] = None,
    label: str = "",
    z_start: float = 0.0,
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
    Z = z_start * np.ones((bins, bins))

    if isinstance(fig, mpl.figure.Figure):
        fig = plot_surface3d_mpl(
            xs=X, ys=Y, zs=Z, fig=fig, label=label, color=color, opacity=opacity
        )
    elif isinstance(fig, Figure):
        fig = plot_surface3d_plotly(
            xs=X, ys=Y, zs=Z, fig=fig, label=label, color=color, opacity=opacity
        )
    else:
        raise NotImplementedError(f"plot_rectangle not implemented for {type(fig)}")

    return fig


def plot_sphere(
    domain: Sphere,
    fig: FigureType,
    color: str = None,
    opacity: float = 1.0,
    dim_select: tuple[int, int] = None,
    label: str = "",
    z_start: float = 0.0,
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
    u, v = np.mgrid[0 : 2 * np.pi : resolution * 2j, 0 : np.pi : resolution * 1j]

    X = radius * np.cos(u) * np.sin(v) + x0
    Y = radius * np.sin(u) * np.sin(v) + y0
    Z = z_start * np.ones(X.shape)

    if isinstance(fig, mpl.figure.Figure):
        fig = plot_surface3d_mpl(
            xs=X, ys=Y, zs=Z, fig=fig, label=label, color=color, opacity=opacity
        )
    elif isinstance(fig, Figure):
        fig = plot_surface3d_plotly(
            xs=X, ys=Y, zs=Z, fig=fig, label=label, color=color, opacity=opacity
        )
    else:
        raise NotImplementedError(f"plot_sphere not implemented for {type(fig)}")

    return fig


if __name__ == "__main__":
    import numpy as np

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
