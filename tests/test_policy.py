import unittest

import numpy as np
import torch
from scipy.integrate import odeint
from torch.autograd import Variable

from barriers import make_barrier
from models.cbf_agent import BarrierPolicy
from systems import make_system
from systems.system_env import SystemEnv


class TestPolicy(unittest.TestCase):

    def test_barrier_policy_2d_single_integrator(self):
        debug_plot = False
        device = torch.device("cpu")
        seed = np.random.randint(0, 1000)
        print(f"seed: {seed}")

        # set seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        # system
        system_id = "SingleIntegrator"
        system = make_system(system_id=system_id)()
        barrier = make_barrier(system=system)["barrier"]

        # create random safe policy
        model = BarrierPolicy(
            system=system,
            barrier=barrier
        ).to(device)
        model.eval()

        # simulation
        env = SystemEnv(
            system=system,
            max_steps=500,
            dt=0.1,
        )
        obs, infos = env.reset(seed=seed)
        termination = truncation = False

        # running on a vehicle
        safety, loc_x, loc_y = [], [], []
        (obs_x, obs_y), R = system.unsafe_domain.center, system.unsafe_domain.radius
        while not termination and not truncation:
            # get safety metric
            px, py = obs
            safe = (px - obs_x) ** 2 + (py - obs_y) ** 2 - R ** 2

            safety.append(safe)
            loc_x.append(px)
            loc_y.append(py)

            # prepare for model input
            with torch.no_grad():
                x_r = torch.from_numpy(obs)[None, :].to(device)
                action = model(x_r)[0]

            # update state
            obs, reward, termination, truncation, infos = env.step(actions=action)


        # debug
        if debug_plot:
            import matplotlib.pyplot as plt

            theta = np.linspace(0, 2*np.pi, 100)
            ox = obs_x + R * np.cos(theta)
            oy = obs_y + R * np.sin(theta)

            plt.plot(ox, oy, label='obstacle')
            plt.plot(loc_x, loc_y, label='robot', marker='o')
            plt.scatter(loc_x[0], loc_y[0], label='start')
            plt.scatter(loc_x[-1], loc_y[-1], label='end')
            plt.legend()

            plt.show()

        tol = 1e-3
        self.assertTrue(np.all(np.array(safety) > -tol), f"got {safety}")

    def test_barrier_policy_2d_single_integrator_gpu(self):
        debug_plot = False
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        seed = np.random.randint(0, 1000)
        print(f"seed: {seed}")

        # set seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        # system
        system_id = "SingleIntegrator"
        system = make_system(system_id=system_id)()
        barrier = make_barrier(system=system)["barrier"]

        # create random safe policy
        model = BarrierPolicy(
            system=system,
            barrier=barrier,
            device=device
        ).to(device)
        model.eval()

        # simulation
        env = SystemEnv(
            system=system,
            max_steps=500,
            dt=0.1,
            device=device
        )
        obs, infos = env.reset(seed=seed)
        termination = truncation = False

        # running on a vehicle
        safety, loc_x, loc_y = [], [], []
        (obs_x, obs_y), R = system.unsafe_domain.center, system.unsafe_domain.radius
        while not termination and not truncation:
            # get safety metric
            px, py = obs
            safe = (px - obs_x) ** 2 + (py - obs_y) ** 2 - R ** 2

            safety.append(safe)
            loc_x.append(px)
            loc_y.append(py)

            # prepare for model input
            with torch.no_grad():
                x_r = torch.from_numpy(obs)[None, :].to(device)
                action = model(x_r)[0]

            # update state
            obs, reward, termination, truncation, infos = env.step(actions=action)


        # debug
        if debug_plot:
            import matplotlib.pyplot as plt

            theta = np.linspace(0, 2*np.pi, 100)
            ox = obs_x + R * np.cos(theta)
            oy = obs_y + R * np.sin(theta)

            plt.plot(ox, oy, label='obstacle')
            plt.plot(loc_x, loc_y, label='robot', marker='o')
            plt.scatter(loc_x[0], loc_y[0], label='start')
            plt.scatter(loc_x[-1], loc_y[-1], label='end')
            plt.legend()

            plt.show()

        tol = 1e-3
        self.assertTrue(np.all(np.array(safety) > -tol), f"got {safety}")