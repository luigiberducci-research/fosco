import torch

from fosco.common.consts import DomainName, TimeDomain
from typing import Callable

import numpy as np
import torch
from plotly.graph_objs import Figure

from fosco.common.domains import Set, Rectangle
from fosco.models import TorchSymDiffFn, TorchSymFn
from fosco.plotting.constants import DOMAIN_COLORS
from fosco.plotting.domains import plot_domain
from fosco.plotting.surface import plot_surface

import plotly.graph_objects as go

from fosco.systems import ControlAffineDynamics, UncertainControlAffineDynamics


def plot_torch_function(function: TorchSymDiffFn, domains: dict[str, Set]) -> go.Figure:
    in_domain: Rectangle = domains[DomainName.XD.value]
    other_domains = {
        k: v
        for k, v in domains.items()
        if k in [DomainName.XI.value, DomainName.XU.value]
    }
    fig = plot_func_and_domains(
        func=function,
        in_domain=in_domain,
        levels=[0.0],
        domains=other_domains,
    )
    return fig


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
        x_mean = lb + (ub - lb) / 2
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
    fig.update_traces(showscale=False, selector=dict(type="surface"))

    return fig


def plot_torch_function_grads(
    function: TorchSymDiffFn, domains: dict[str, Set]
) -> tuple[list[go.Figure], list[str]]:
    in_domain: Rectangle = domains[DomainName.XD.value]
    other_domains = {
        k: v
        for k, v in domains.items()
        if k in [DomainName.XI.value, DomainName.XU.value]
    }

    figs = []
    titles = []
    for dim in range(function.input_size):
        func = lambda x: function.gradient(x)[:, dim]
        fig = plot_func_and_domains(
            func=func,
            in_domain=in_domain,
            levels=[0.0],
            domains=other_domains,
        )
        figs.append(fig)
        titles.append(str(dim))
    return figs, titles


def plot_lie_derivative(
    function: TorchSymDiffFn,
    system: ControlAffineDynamics,
    domains: dict[str, Set],
) -> tuple[list[go.Figure], list[str]]:
    time_domain = system.time_domain
    in_domain: Rectangle = domains[DomainName.XD.value]
    u_domain = domains[DomainName.UD.value]
    other_domains = {
        k: v
        for k, v in domains.items()
        if k in [DomainName.XI.value, DomainName.XU.value]
    }

    assert isinstance(
        u_domain, Rectangle
    ), "only rectangular domains are supported for u"

    figs = []
    titles = []

    u_vertices = u_domain.get_vertices()
    zero_u = np.zeros_like(u_vertices[0])
    for u_orig in np.concatenate((u_vertices, [zero_u])):
        ctrl = (
            lambda x: torch.ones((x.shape[0], system.n_controls))
            * torch.tensor(u_orig).float()
        )
        if isinstance(system, UncertainControlAffineDynamics):
            f = lambda x, u: system._f_torch(x, u, z=None, only_nominal=True)
        else:
            f = lambda x, u: system._f_torch(x, u)

        # lie derivative
        is_dt = time_domain == TimeDomain.DISCRETE
        func = lambda x: lie_derivative_fn(
            certificate=function, f=f, ctrl=ctrl, is_dt=is_dt
        )(x)
        fig = plot_func_and_domains(
            func=func,
            in_domain=in_domain,
            levels=[0.0],
            domains=other_domains,
            dim_select=(0, 1),
        )

        figs.append(fig)
        titles.append(str(u_orig).replace(" ", ""))

    return figs, titles


def lie_derivative_fn(
    certificate: TorchSymDiffFn, f: Callable, ctrl: Callable, is_dt: bool
) -> Callable[[np.ndarray], np.ndarray]:
    def lie_derivative(x):
        grad_net = certificate.gradient(x)
        xdot = f(x, ctrl(x))
        grad_net = grad_net.reshape(-1, 1, grad_net.shape[-1])  # (batch, 1, dim)
        xdot = xdot.reshape(-1, xdot.shape[-1], 1)  # (batch, dim, 1)
        # (batch, 1, dim) @ (batch, dim, 1) = (batch, 1, 1)
        return (grad_net @ xdot)[:, 0]

    def delta_h(x):
        hx = certificate(x)
        next_hx = certificate(f(x, ctrl(x)))
        return next_hx - hx

    if is_dt:
        return delta_h
    return lie_derivative


def plot_cbf_condition(
    barrier: TorchSymDiffFn,
    system: ControlAffineDynamics,
    domains: dict[str, Set],
    compensator: TorchSymFn = None,
    alpha: Callable[[np.ndarray], np.ndarray] = None,
) -> tuple[list[go.Figure], list[str]]:
    if alpha is None:
        alpha = lambda x: 1.0 * x

    time_domain = system.time_domain
    in_domain: Rectangle = domains[DomainName.XD.value]
    u_domain = domains[DomainName.UD.value]
    other_domains = {
        k: v
        for k, v in domains.items()
        if k in [DomainName.XI.value, DomainName.XU.value]
    }

    assert isinstance(
        u_domain, Rectangle
    ), "only rectangular domains are supported for u"

    if isinstance(system, UncertainControlAffineDynamics):
        f = lambda x, u: system._f_torch(x, u, z=None, only_nominal=True)
    else:
        f = lambda x, u: system._f_torch(x, u)

    figs = []
    titles = []

    u_vertices = u_domain.get_vertices()
    zero_u = np.zeros_like(u_vertices[0])
    for u_orig in np.concatenate((u_vertices, [zero_u])):
        ctrl = (
            lambda x: torch.ones((x.shape[0], system.n_controls))
            * torch.tensor(u_orig).float()
        )

        is_dt = time_domain == TimeDomain.DISCRETE
        func = lambda x: cbf_condition_fn(
            certificate=barrier,
            alpha=alpha,
            f=f,
            ctrl=ctrl,
            sigma=compensator,
            is_dt=is_dt,
        )(x)
        fig = plot_func_and_domains(
            func=func,
            in_domain=in_domain,
            levels=[0.0],
            domains=other_domains,
            dim_select=(0, 1),
        )

        figs.append(fig)
        titles.append(str(u_orig).replace(" ", ""))

    return figs, titles


def cbf_condition_fn(
    certificate, alpha, f, ctrl, sigma=None, is_dt=False
) -> Callable[[np.ndarray], np.ndarray]:
    def cbf_condition(x):
        if sigma:
            return (
                lie_derivative_fn(certificate, f, ctrl, is_dt)(x).squeeze()
                - sigma(x).squeeze()
                + alpha(certificate(x)).squeeze()
            )
        else:
            return (
                lie_derivative_fn(certificate, f, ctrl, is_dt)(x).squeeze()
                + alpha(certificate(x)).squeeze()
            )

    return cbf_condition
