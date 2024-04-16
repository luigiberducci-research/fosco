"""
This example demonstrates how to learn a valid control barrier function (CBF) for a single integrator system
using the FOSCO library. The CBF is learned using a neural network, and the learned CBF is used to verify the
safety of the system.
"""

import time

import numpy as np
import torch
from matplotlib import pyplot as plt

from barriers import make_barrier, make_compensator
from fosco.cegis import Cegis
from fosco.common.domains import Sphere, Rectangle
from fosco.config import CegisConfig
from fosco.plotting.functions import plot_torch_function
from fosco.systems import make_system
from fosco.systems.gym_env.system_env import SystemEnv
from fosco.systems.uncertainty import add_uncertainty
from rl_trainer.safe_ppo.safeppo_agent import SafeActorCriticAgent


def main():
    # system parameters
    system_type = "SingleIntegrator"
    uncertainty_type = "AdditiveBounded"
    seed = 133636  # seed for reproducibility, None for random seed
    verbose = 1

    # learning parameters
    params = {
        "activations": ["htanh"],
        "n_hidden_neurons": [20],
        "max_iters": 20,
        "n_data_samples": 5000,
        "n_resampling": 100,
        "resampling_std": 0.1,
    }

    # simulation parameters - these are not used for learning the barrier
    sim_dt = 0.2
    sim_max_steps = 250
    sim_batch_size = 25

    # create system
    system = make_system(system_id=system_type)()
    system = add_uncertainty(uncertainty_type=uncertainty_type, system=system)

    # learn control barrier function
    # barrier, compensator = learn_barrier_and_compensator(system=system, params=params, seed=seed, verbose=verbose)
    barrier = make_barrier(system=system)
    compensator = make_compensator(system=system)
    for func in [barrier, compensator]:
        fig = plot_torch_function(function=func, domains=system.domains)
        fig.show()

    # create simulation environment
    env = SystemEnv(
        system=system,
        dt=sim_dt,
        max_steps=sim_max_steps,
    )

    # create safe policy
    pi = SafeActorCriticAgent(envs=env, barrier=barrier, compensator=compensator)

    # run simulation
    t0 = time.time()
    print(f"Run Simulation: seed {seed}")

    obs, info = env.reset(
        seed=seed, options={"batch_size": sim_batch_size, "return_as_np": False}
    )
    terminations = truncations = np.zeros(sim_batch_size, dtype=bool)

    traj = {"x": [obs], "u": []}
    while not (any(terminations) or any(truncations)):
        # with torch.no_grad():
        obs = obs[None] if len(obs.shape) == 1 else obs
        results = pi.get_action_and_value(x=obs)
        u = results["safe_action"].detach().numpy()

        obs, rewards, terminations, truncations, infos = env.step(u)
        env.render()

        traj["x"].append(obs)
        traj["u"].append(u)

    print(f"Sim time: {time.time() - t0} seconds")

    # plotting
    xs = np.array(traj["x"])[:, :, 0]
    ys = np.array(traj["x"])[:, :, 1]

    # plotting
    fig, ax = plt.subplots()
    unsafe_domain: Sphere = env.system.unsafe_domain
    circle = plt.Circle(
        unsafe_domain.center[:2], unsafe_domain.radius, color="r", fill=False
    )
    ax.add_artist(circle)

    ax.plot(xs, ys)

    state_domain: Rectangle = env.system.state_domain
    ax.set_xlim(state_domain.lower_bounds[0], state_domain.upper_bounds[0])
    ax.set_ylim(state_domain.lower_bounds[1], state_domain.upper_bounds[1])
    ax.set_aspect("equal", "box")

    plt.show()


def learn_barrier_and_compensator(system, params, seed, verbose):
    sets = system.domains

    # data generator
    data_gen = {
        "init": lambda n: sets["init"].generate_data(n),
        "unsafe": lambda n: sets["unsafe"].generate_data(n),
        "lie": lambda n: torch.concatenate(
            [
                sets["lie"].generate_data(n),
                torch.zeros(n, sets["input"].dimension),
                sets["uncertainty"].generate_data(n),
            ],
            dim=1,
        ),
        "uncertainty": lambda n: torch.concatenate(
            [
                sets["lie"].generate_data(n),
                sets["input"].generate_data(n),
                sets["uncertainty"].generate_data(n),
            ],
            dim=1,
        ),
    }

    config = CegisConfig(
        EXP_NAME="robust_cbf",
        CERTIFICATE="rcbf",
        VERIFIER="z3",
        RESAMPLING_N=params["n_resampling"],
        RESAMPLING_STDDEV=params["resampling_std"],
        ACTIVATION=params["activations"],
        N_HIDDEN_NEURONS=params["n_hidden_neurons"],
        CEGIS_MAX_ITERS=params["max_iters"],
        N_DATA=params["n_data_samples"],
        SEED=seed,
        LOGGER=None,
        LOSS_WEIGHTS={
            "init": 1.0,
            "unsafe": 1.0,
            "lie": 1.0,
            "robust": 1.0,
            "conservative_b": 1.0,
            "conservative_sigma": 0.1,
        },
    )
    cegis = Cegis(
        system=system, domains=sets, config=config, data_gen=data_gen, verbose=verbose
    )

    result = cegis.solve()
    return result.barrier, result.compensator


if __name__ == "__main__":
    main()
