from typing import Callable

import numpy as np
import torch
from plotly.graph_objs import Figure

from fosco.common.domains import Set, Rectangle
from fosco.plotting.constants import DOMAIN_COLORS
from fosco.plotting.domains import plot_domain
from fosco.plotting.surface import plot_surface


def plot_func_and_domains(
    func: Callable[[np.ndarray], np.ndarray],
    in_domain: Rectangle,
    levels: list[float] = None,
    domains: dict[str, Set] = None,
    dim_select: tuple[int, int] = None,
) -> Figure:
    """
    Plot the function of a torch module projected onto 2d input space.
    """
    domains = domains or {}
    dim_select = dim_select or (0, 1)
    levels = levels or []

    assert len(dim_select) == 2, "dim_select must be a tuple of 2 int"
    assert (
        type(dim_select[0]) == type(dim_select[1]) == int
    ), "dim_select must be a tuple of 2 int"
    assert isinstance(
        in_domain, Rectangle
    ), f"in_domain must be a Rectangle, got {type(in_domain)}"
    assert isinstance(levels, list) and all(
        [isinstance(l, float) for l in levels]
    ), "levels must be a list of floats"

    def proj_func(x):
        """
        Extend 2d input to the full input space to evaluate n-dim function.
        The dimensions not in dim_select are set to the mean of the domain.
        """
        assert (
            len(x.shape) == 2 and x.shape[1] == len(dim_select) == 2
        ), "x must be a batch of 2d points"

        lb, ub = np.array(in_domain.lower_bounds), np.array(in_domain.upper_bounds)
        x_mean = lb + (lb + ub) / 2.0
        x_ext = x_mean[None].repeat(x.shape[0], 0)
        x_ext[:, dim_select] = x

        x_ext = torch.from_numpy(x_ext).float()
        return func(x_ext).detach().numpy().squeeze()

    xrange = (
        in_domain.lower_bounds[dim_select[0]],
        in_domain.upper_bounds[dim_select[0]],
    )
    yrange = (
        in_domain.lower_bounds[dim_select[1]],
        in_domain.upper_bounds[dim_select[1]],
    )

    fig = Figure()

    fig = plot_surface(
        func=proj_func,
        xrange=xrange,
        yrange=yrange,
        levels=levels,
        fig=fig,
        opacity=0.5,
    )

    for dname, domain in domains.items():
        color = DOMAIN_COLORS[dname] if dname in DOMAIN_COLORS else None
        fig = plot_domain(domain, fig, color=color, dim_select=dim_select, label=dname)

    # show legend and hide colorbar
    fig.update_traces(showlegend=True)
    fig.update_traces(showscale=False)

    return fig


def lie_derivative_fn(certificate, f, ctrl) -> Callable[[np.ndarray], np.ndarray]:
    def lie_derivative(x):
        grad_net = certificate.gradient(x)
        xdot = f(x, ctrl(x))
        grad_net = grad_net.reshape(-1, 1, grad_net.shape[-1])  # (batch, 1, dim)
        xdot = xdot.reshape(-1, xdot.shape[-1], 1)  # (batch, dim, 1)
        # (batch, 1, dim) @ (batch, dim, 1) = (batch, 1, 1)
        return (grad_net @ xdot)[:, 0]

    return lie_derivative


def cbf_condition_fn(
    certificate, alpha, f, ctrl, sigma=None
) -> Callable[[np.ndarray], np.ndarray]:
    def cbf_condition(x):
        if sigma:
            return (
                lie_derivative_fn(certificate, f, ctrl)(x).squeeze()
                - sigma(x).squeeze()
                + alpha(certificate(x)).squeeze()
            )
        else:
            return (
                lie_derivative_fn(certificate, f, ctrl)(x).squeeze()
                + alpha(certificate(x)).squeeze()
            )

    return cbf_condition
