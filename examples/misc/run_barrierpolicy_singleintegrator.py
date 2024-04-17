import numpy as np
import torch
from matplotlib import pyplot as plt

from barriers import make_barrier, make_compensator
from fosco.systems.gym_env.system_env import SystemEnv
from fosco.systems.uncertainty import add_uncertainty
from rl_trainer.safe_ppo.safeppo_agent import BarrierPolicy

from fosco.systems import make_system, EulerDTSystem

import time


def main():
    system_id = "SingleIntegrator"
    uncertainty_id = "AdditiveBounded"
    barrier_to_load = "default"
    compensator_to_load = "default"
    batch_size = 5
    max_steps = 750
    dt = 0.01

    # this seed will give a policy which navigates towards the obstacle (demo)
    seed = 425  # np.random.randint(0, 1000)

    # to ensure same policy
    np.random.seed(seed)
    torch.manual_seed(seed)

    t0 = time.time()
    system = make_system(system_id=system_id)()
    system = add_uncertainty(system=system, uncertainty_type=uncertainty_id)

    env = SystemEnv(system=system, dt=dt, max_steps=max_steps,)

    barrier = make_barrier(system=system, model_to_load=barrier_to_load)
    compensator = make_compensator(system=system, model_to_load=compensator_to_load)
    pi = BarrierPolicy(system=system, barrier=barrier, compensator=compensator)

    print(f"Init time: {time.time() - t0} seconds")

    # simulation
    t0 = time.time()
    print(f"Seed {seed}")
    obs, info = env.reset(
        seed=seed, options={"batch_size": batch_size, "return_as_np": False}
    )
    terminations = truncations = np.zeros(batch_size, dtype=bool)

    traj = {"x": [obs], "u": [], "hx": [], "px": []}
    while not (any(terminations) or any(truncations)):
        # with torch.no_grad():
        obs = obs[None] if len(obs.shape) == 1 else obs
        u = pi(x=obs)
        u = u.detach().numpy()

        obs, rewards, terminations, truncations, infos = env.step(u)

        traj["x"].append(obs)
        traj["u"].append(u)
        traj["hx"].append(pi.hx.detach().numpy())
        traj["px"].append(pi.px.detach().numpy())
    print(f"Sim time: {time.time() - t0} seconds")

    # plt
    plt.rcParams.update({"font.size": 20})

    t0 = time.time()
    traj["x"] = np.array(traj["x"])
    traj["u"] = np.array(traj["u"])
    traj["hx"] = np.array(traj["hx"])
    traj["px"] = np.array(traj["px"])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 15))
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
        linestyle="dashed",
        label="obstacle",
    )

    ax1.set_title("Space Trajectories")
    ax1.set_xlabel("x[0]")
    ax1.set_ylabel("x[1]")
    ax1.set_xlim(-5, +5)
    ax1.set_ylim(-5, +5)
    ax1.axis("equal")
    ax1.invert_yaxis()
    ax1.legend()

    # plot u
    for i in range(traj["u"].shape[1]):
        label_0 = "u[0]" if i == 0 else None
        label_1 = "u[1]" if i == 0 else None
        vxs = traj["u"][:, i, 0]
        vys = traj["u"][:, i, 1]
        ax2.plot(vxs, label=label_0, color="blue")
        ax2.plot(vys, label=label_1, color="red")
        ax2.scatter(0.0, vxs[0], marker="x", color="k")
        ax2.scatter(0.0, vys[0], marker="x", color="k")
    ax2.set_title("Input Trajectories")
    ax2.set_xlabel("time")
    ax2.set_ylabel("control input")
    ax2.legend()

    # plt hx
    ax3.set_title("Barrier Trajectories")
    ax3.plot(traj["hx"].squeeze(), label="h(x)")
    ax3.hlines(0, 0, traj["hx"].shape[0], color="r", linestyle="dashed")
    ax3.set_xlabel("time")
    ax3.set_ylabel("h(x)")

    print(f"Plotting time: {time.time() - t0} seconds")

    plt.tight_layout()
    barrier_name = (
        barrier_to_load if barrier_to_load in ["default", "tunable"] else "learned"
    )
    compensator_name = (
        compensator_to_load
        if compensator_to_load in ["default", "tunable"]
        else "learned"
    )
    plt.savefig(
        f"{system.id}-Barrier{barrier_name}-Compensator{compensator_name}-Seed{seed}-N{batch_size}.pdf"
    )
    plt.show()


if __name__ == "__main__":
    main()
