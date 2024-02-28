import numpy as np
import torch
from matplotlib import pyplot as plt

from barriers import make_barrier
from models.cbf_agent import BarrierPolicy

from systems import make_system

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

from systems.system_env import SystemEnv
import time


def dummy_term_fn(actions, next_obss):
    return torch.zeros((actions.shape[0],), dtype=torch.bool)


def dummy_reward_fn(actions, next_obss):
    return torch.ones((next_obss.shape[0],), dtype=torch.float32)

def main():
    system_id = "SingleIntegrator"
    batch_size = 10
    max_steps = 200
    dt = 0.1

    # this seed will give a policy which navigates towards the obstacle (demo)
    seed = 425 #np.random.randint(0, 1000)

    # to ensure same policy
    np.random.seed(seed)
    torch.manual_seed(seed)

    t0 = time.time()
    system = make_system(system_id=system_id)()
    env = SystemEnv(
        system=system,
        reward_fn=dummy_reward_fn,
        termination_fn=dummy_term_fn,
        dt=dt,
        max_steps=max_steps,
    )

    barrier = make_barrier(system=system)["barrier"]
    pi = BarrierPolicy(
        system=system,
        barrier=barrier,
    )
    print(f"Init time: {time.time() - t0} seconds")

    # simulation
    t0 = time.time()
    print(f"Seed {seed}")
    obs, info = env.reset(seed=seed, options={"batch_size": batch_size, "return_as_np": False})
    terminations = truncations = np.zeros(batch_size, dtype=bool)

    traj = {"x": [obs], "u": [], "hx": [], "px": []}
    while not (any(terminations) or any(truncations)):
        with torch.no_grad():
            obs = obs[None] if len(obs.shape) == 1 else obs
            u = pi(x=obs)
            u = u.detach().numpy()

        obs, rewards, terminations, truncations, infos = env.step(u)

        traj["x"].append(obs)
        traj["u"].append(u)
        traj["hx"].append(pi.hx)
        traj["px"].append(pi.px)
    print(f"Sim time: {time.time() - t0} seconds")

    # plt
    t0 = time.time()
    traj["x"] = np.array(traj["x"])
    traj["u"] = np.array(traj["u"])
    traj["hx"] = np.array(traj["hx"])
    traj["px"] = np.array(traj["px"])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    for i in range(traj["x"].shape[1]):
        xs = traj["x"][:, i, 0]
        ys = traj["x"][:, i, 1]
        ax1.plot(xs, ys, color="blue")
        ax1.scatter(xs[0], ys[0], marker="x", color="k")

    # draw circle unsafe set
    cx, r = system.domains["unsafe"].center, system.domains["unsafe"].radius
    ax1.plot(
        cx[0] + r * np.cos(np.linspace(0, 2 * np.pi, 25)),
        cx[1] + r * np.sin(np.linspace(0, 2 * np.pi, 25)),
        color="r",
        linestyle="dashed"
    )

    ax3.set_title("Space Trajectories")
    ax3.set_xlabel("coordinate x")
    ax3.set_ylabel("coordinate y")

    ax1.set_xlim(-5, +5)
    ax1.set_ylim(-5, +5)
    ax1.legend()

    # plot u
    for i in range(traj["u"].shape[1]):
        vxs = traj["u"][:, i, 0]
        vys = traj["u"][:, i, 1]
        ax2.plot(vxs, label="u[0]", color="blue")
        ax2.plot(vys, label="u[1]", color="red")
        ax2.scatter(0.0, vxs[0], marker="x", color="k")
        ax2.scatter(0.0, vys[0], marker="x", color="k")
    ax3.set_title("Input Trajectories")
    ax3.set_xlabel("time")
    ax3.set_ylabel("control input")

    # plt hx
    ax3.set_title("Barrier Trajectories")
    ax3.plot(traj["hx"].squeeze(), label="h(x)")
    ax3.set_xlabel("time")
    ax3.set_ylabel("h(x)")

    print(f"Plotting time: {time.time() - t0} seconds")
    plt.show()


if __name__ == "__main__":
    main()

