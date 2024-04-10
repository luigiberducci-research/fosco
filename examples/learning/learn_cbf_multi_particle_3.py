"""
This example demonstrates how to learn a valid control barrier function (CBF) for a single integrator system
using the FOSCO library. The CBF is learned using a neural network, and the learned CBF is verified to be valid
for the system.

After having found a valid CBF or having reached the maximum number of iterations, the learned function is used
in simulation with a random explorative policy to demonstrate it can preserve safety while exploring the state space.
"""
import time

import numpy as np
import torch
from matplotlib import pyplot as plt

from fosco.cegis import Cegis
from fosco.common.domains import Sphere, Rectangle
from fosco.config import CegisConfig
from fosco.plotting.functions import plot_torch_function, plot_func_only
from fosco.systems import make_system
from fosco.systems.gym_env.system_env import SystemEnv
from rl_trainer.safe_ppo.safeppo_agent import SafeActorCriticAgent


def main():
    # system parameters
    system_type = "MultiParticleSingleIntegrator"
    n_agents = 3
    seed = 133636   # seed for reproducibility, None for random seed
    verbose = 1

    # learning parameters
    params = {
        "activations": ["htanh"],
        "n_hidden_neurons": [20],
        "max_iters": 20,
        "n_data_samples": 10000,
        "n_resampling": 250,
        "resampling_std": 0.1,
    }

    # simulation parameters - these are not used for learning the barrier
    sim_dt = 0.2
    sim_max_steps = 250
    sim_batch_size = 25

    # create system
    system = make_system(system_id=system_type)(n_agents=n_agents)

    # learn control barrier function
    barrier = learn_barrier(system=system, params=params, seed=seed, verbose=verbose)
    custom_plot(barrier=barrier, system=system)


    # create simulation environment
    env = SystemEnv(
        system=system,
        dt=sim_dt,
        max_steps=sim_max_steps,
        render_mode="human"
    )

    # create safe policy
    pi = SafeActorCriticAgent(envs=env, barrier=barrier)

    # run simulation
    t0 = time.time()
    print(f"Run Simulation: seed {seed}")

    obs, info = env.reset(
        seed=seed, options={
            "batch_size": sim_batch_size,
            "init_state": np.array([-3.0, 0.0, 3.0, 0.0]),
            "return_as_np": False
        }
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
    fig, ax = plt.subplots()

    for i in range(sim_batch_size):
        x1s = np.array(traj["x"])[i, :, 0]
        y1s = np.array(traj["x"])[i, :, 1]
        x2s = np.array(traj["x"])[i, :, 2]
        y2s = np.array(traj["x"])[i, :, 3]

        rnd_colors = np.random.rand(3,)
        ax.plot(x1s, y1s, c=rnd_colors)
        ax.plot(x2s, y2s, c=rnd_colors)

    # circle = plt.Circle(unsafe_domain.center[:2], unsafe_domain.radius, color='r', fill=False)
    # ax.add_artist(circle)
    state_domain: Rectangle = env.system.state_domain
    ax.set_xlim(state_domain.lower_bounds[0], state_domain.upper_bounds[0])
    ax.set_ylim(state_domain.lower_bounds[1], state_domain.upper_bounds[1])
    ax.set_aspect('equal', 'box')

    plt.show()



def learn_barrier(system, params, seed, verbose) -> torch.nn.Module:
    """
    Learn a control barrier function for the system.

    Parameters
    ----------
    system: System
    params: dict of training parameters
    seed: int or None
    verbose: int

    Returns
    -------
    barrier: torch.nn.Module
        Training barrier function
    """
    sets = system.domains

    # data generator
    data_gen = {
        "init": lambda n: sets["init"].generate_data(n),
        "unsafe": lambda n: sets["unsafe"].generate_data(n),
        "lie": lambda n: torch.concatenate(
            [sets["lie"].generate_data(n), sets["input"].generate_data(n)], dim=1
        )
    }

    config = CegisConfig(
        EXP_NAME="simple_cbf",
        CERTIFICATE="cbf",
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
            "conservative_b": 2.0,
        },
    )
    cegis = Cegis(
        system=system,
        domains=sets,
        config=config,
        data_gen=data_gen,
        verbose=verbose
    )

    result = cegis.solve()
    return result.barrier

def custom_plot(barrier, system):
    mask_state = np.array([1.0, 0.0, 1.0, 0.0]) # mask out the dy dimensions

    domain = system.state_domain
    lb = np.array(domain.lower_bounds) * mask_state
    ub = np.array(domain.upper_bounds) * mask_state

    new_domain = Rectangle(
        vars=domain.vars,
        lb=tuple(lb),
        ub=tuple(ub),
    )

    fig = plot_func_only(func=barrier, in_domain=new_domain, levels=[0.0], dim_select=(0, 2))
    fig.show()

if __name__ == "__main__":
    main()