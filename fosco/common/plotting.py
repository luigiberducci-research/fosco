import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm


def benchmark_plane(model, ctrl, certificate, domains, levels, xrange, yrange, ax=None):
    """
    Plot the dynamical model phase plane and the domains with coloured labels.

    If a certificate is provided, it plots the levelsets of the certificate, as
    defined by the levels argument.
    """

    ax = ax or plt.gca()

    ax = model.plot(ctrl=ctrl, ax=ax, xrange=xrange, yrange=yrange)
    ax = plot_domains(domains, ax=ax)

    if certificate is not None:
        ax = certificate_countour(certificate, ax=ax, levels=levels)

    ax = add_legend(ax)
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    ax.set_title("Phase Plane")
    return ax


def certificate_countour(certificate, ax=None, levels=[0]):
    """Plot contours of the certificate.

    Args:
        certificate (NNLearner): certificate to plot.
        ax : matplotlib axis. Defaults to None, in which case an axis is created.
        levels (list, optional): Level sets of the certificate to plot. Defaults to zero contour.
    """

    ax = ax or plt.gca()
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    XT = torch.tensor(X, dtype=torch.float32)
    YT = torch.tensor(Y, dtype=torch.float32)
    ZT = certificate(torch.cat((XT.reshape(-1, 1), YT.reshape(-1, 1)), dim=1))
    Z = ZT.detach().numpy().reshape(X.shape)
    levels.sort()
    CS = ax.contour(X, Y, Z, levels=levels, colors="black", linestyles="dashed")
    # ax.clabel(CS, inline=True, fontsize=10)
    return ax


def benchmark_3d(
    func, domains, levels: list[float], xrange, yrange, title: str = "", fig=None,
):
    if fig is None:
        fig = plt.gcf()

    ax = fig.add_subplot(1, 1, 1, projection="3d")

    ax = certificate_surface(func, ax=ax, levels=levels, xrange=xrange, yrange=yrange)
    ax = plot_domains(domains, ax)
    ax = add_legend(ax)
    ax.set_title(title)

    return ax


def benchmark_lie(model, ctrl, certificate, domains, levels, xrange, yrange, fig=None):
    """
    Plot the lie derivative of the certificate benchmark.
    If the domains are provided, they are plotted as well.
    """

    fig = fig or plt.figure()

    ax = fig.add_subplot(1, 1, 1, projection="3d")

    ax = certificate_lie(
        certificate=certificate,
        model=model,
        ctrl=ctrl,
        ax=ax,
        xrange=xrange,
        yrange=yrange,
    )
    ax = plot_domains(domains, ax)
    ax = add_legend(ax)
    ax.set_title(f"Lie Derivative")

    return ax


def certificate_lie(certificate, model, ctrl, ax, xrange, yrange):
    """Plot the surface of the lie derivative of the certificate."""
    x = np.linspace(xrange[0], xrange[1], 100)
    y = np.linspace(yrange[0], yrange[1], 100)
    X, Y = np.meshgrid(x, y)
    XT = torch.tensor(X, dtype=torch.float32)
    YT = torch.tensor(Y, dtype=torch.float32)
    ZT = certificate.compute_net_gradnet(
        torch.cat((XT.reshape(-1, 1), YT.reshape(-1, 1)), dim=1)
    )[1]
    Z = ZT.detach().numpy()

    obs = torch.stack([XT.ravel(), YT.ravel()]).T.float()
    uu = ctrl(obs)

    dx, dy = model.f(v=obs, u=uu).detach().numpy().T
    df = np.stack([dx, dy], axis=1)
    lie = (df @ Z.T).diagonal()
    lie = lie.reshape(X.shape)
    ax.plot_surface(X, Y, lie, cmap=cm.coolwarm, alpha=0.7, rstride=5, cstride=5)
    ax.contour(
        X, Y, lie, levels=[0], colors="k", linestyles="dashed", linewidths=2.5,
    )
    return ax


def plot_domains(domains, ax):
    for lab, dom in domains.items():
        try:
            dom.plot(None, ax, label=lab)
        except AttributeError:
            pass
    return ax


def certificate_surface(
    certificate, ax=None, xrange=[-3, 3], yrange=[-3, 3], levels=[0]
):
    """Plot the surface of the certificate.
    Args:
        certificate (NNLearner): certificate to plot.
        ax : matplotlib axis. Defaults to None, in which case an axis is created.
        levels (list, optional): Level sets of the certificate to plot. Defaults to zero contour.
        xrange (tuple, optional): Range of the x-axis. Defaults to [-3, 3].
        yrange (tuple, optional): Range of the y-axis. Defaults to [-3, 3].
    """
    ax = ax or plt.gca()
    x = np.linspace(xrange[0], xrange[1], 100)
    y = np.linspace(yrange[0], yrange[1], 100)
    X, Y = np.meshgrid(x, y)
    XT = torch.tensor(X, dtype=torch.float32)
    YT = torch.tensor(Y, dtype=torch.float32)
    ZT = certificate(torch.cat((XT.reshape(-1, 1), YT.reshape(-1, 1)), dim=1))
    Z = ZT.detach().numpy().reshape(X.shape)
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.7, rstride=5, cstride=5)
    levels.sort()
    ax.contour(
        X, Y, Z, levels=levels, colors="k", linestyles="dashed", linewidths=2.5,
    )
    return ax


def add_legend(ax):
    """Add legend to the axis without duplicate labels."""
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right")
    return ax


def get_plot_colour(label):
    if label == "goal":
        return "green", "Goal"
    elif label == "unsafe":
        return "red", "Unsafe"
    elif label == "safe":
        return "tab:cyan", "Safe"
    elif label == "init":
        return "blue", "Initial"
    elif label == "lie":
        return "black", "Domain"
    elif label == "final":
        return "orange", "Final"
    else:
        # We don't want to plot border sets
        raise AttributeError("Label does not correspond to a plottable set.")
