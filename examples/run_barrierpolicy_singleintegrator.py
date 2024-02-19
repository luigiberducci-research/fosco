import numpy as np
import torch
from matplotlib import pyplot as plt

from barriers import make_barrier
from models.policy import BarrierPolicy
from systems import make_system, make_domains

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


def construct_cbf_problem(ud: int, umin: np.ndarray, umax: np.ndarray) -> CvxpyLayer:
    Q = np.diag(np.ones(ud))
    px = cp.Parameter(ud)

    Lfhx = cp.Parameter(1)
    Lghx = cp.Parameter(ud)
    alphahx = cp.Parameter(1)

    u = cp.Variable(ud)

    constraints = []
    # input constraint: u in U
    constraints += [u >= umin, u <= umax]

    # constraint: hdot(x,u) + alpha(h(x)) >= 0
    constraints += [Lfhx + Lghx @ u + alphahx >= 0.0]

    # objective: u.T Q u + p.T u
    objective = 1 / 2 * cp.quad_form(u, Q) + px.T @ u
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return CvxpyLayer(problem, variables=[u], parameters=[px, Lfhx, Lghx, alphahx])


def main():
    system_id = "SingleIntegrator"

    seed = 425 #np.random.randint(0, 1000)
    N = 10
    T = 20.0
    dt = 0.1

    # seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"Seed {seed}")

    system = make_system(system_id=system_id)
    f = system()
    barrier = make_barrier(system=f)["barrier"]
    sets = make_domains(system_id=system_id)

    pi = BarrierPolicy(
        system=f,
        barrier=barrier,
    )

    x = sets["init"].generate_data(batch_size=N).float()

    t = 0.0
    traj = {"x": [x], "u": [], "hx": [], "px": []}
    while t < T:
        with torch.no_grad():
            u = pi(x=x)
            u = u.detach().numpy()

        x = x + dt * f(x, u)
        t += dt

        traj["x"].append(x)
        traj["u"].append(u)
        traj["hx"].append(pi.hx)
        traj["px"].append(pi.px)

    # plt
    traj["x"] = np.array(traj["x"]).squeeze()
    traj["u"] = np.array(traj["u"]).squeeze()
    traj["hx"] = np.array(traj["hx"]).squeeze()
    traj["px"] = np.array(traj["px"]).squeeze()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    for i in range(traj["x"].shape[1]):
        ax1.plot(traj["x"][:, i, 0], traj["x"][:, i, 1], color="blue")
        ax1.scatter(traj["x"][0, i, 0], traj["x"][0, i, 1], marker="x", color="k")

    # draw circle unsafe set
    cx, r = sets["unsafe"].center, sets["unsafe"].radius
    ax1.plot(
        cx[0] + r * np.cos(np.linspace(0, 2 * np.pi, 25)),
        cx[1] + r * np.sin(np.linspace(0, 2 * np.pi, 25)),
        color="r",
        linestyle="dashed"
    )

    ax1.set_xlim(-5, +5)
    ax1.set_ylim(-5, +5)
    ax1.legend()

    # plt hx
    ax2.plot(traj["hx"], label="h(x)")

    plt.show()


if __name__ == "__main__":
    main()
