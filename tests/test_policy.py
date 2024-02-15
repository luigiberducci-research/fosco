import unittest

import numpy as np
import torch
from scipy.integrate import odeint
from torch.autograd import Variable

from models.policy import BarrierPolicy

class TestPolicy(unittest.TestCase):

    def test_barrier_policy_2d_robot(self):
        debug_plot = True
        device = torch.device("cpu")
        seed = 424
        print(f"seed: {seed}")

        # set seed
        np.random.seed(seed)
        torch.manual_seed(seed)


        # dynamics
        def dynamics(y, t):
            dxdt = y[3] * np.cos(y[2])
            dydt = y[3] * np.sin(y[2])
            dttdt = y[4]  # u1
            dvdt = y[5]  # u2
            return [dxdt, dydt, dttdt, dvdt, 0, 0]

        # create random policy
        nFeatures, nHidden1, nHidden21, nHidden22, nCls = 5, 128, 32, 32, 2
        mean = np.zeros(nFeatures)
        std = np.ones(nFeatures)
        model = BarrierPolicy(nFeatures, nHidden1, nHidden21, nHidden22, nCls,
                              mean=mean, std=std, device=device, bn=False).to(device)
        model.eval()

        # simulation
        obs_x, obs_y, R = model.obs_x, model.obs_y, model.R
        state = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        mina, maxa = -1.0, 1.0
        dt = [0, 0.05]

        # running on a vehicle
        safety, loc_x, loc_y = [], [], []
        for i in range(500):  # train0, 10
            obs = (state[:5] - mean) / std

            # get safety metric
            px, py, theta, speed, dsty, dstx = state
            safe = (px - obs_x) ** 2 + (py - obs_y) ** 2 - R ** 2

            safety.append(safe)
            loc_x.append(px)
            loc_y.append(py)

            # prepare for model input
            with torch.no_grad():
                x_r = Variable(torch.from_numpy(obs), requires_grad=False)
                x_r = torch.reshape(x_r, (1, nFeatures))
                x_r = x_r.to(device)
                ctrl = model(x_r, 0)

            # update state
            state[-2] = np.clip(ctrl[0], mina, maxa)
            state[-1] = np.clip(ctrl[1], mina, maxa)

            # update dynamics
            rt = np.float32(odeint(dynamics, state, dt))
            state[0] = rt[1][0]
            state[1] = rt[1][1]
            state[2] = rt[1][2]
            state[3] = rt[1][3]


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

    def test_barrier_policy_2d_single_integrator(self):
        debug_plot = True
        device = torch.device("cpu")
        seed = np.random.randint(0, 1000)
        print(f"seed: {seed}")

        # set seed
        np.random.seed(seed)
        torch.manual_seed(seed)


        # dynamics
        def dynamics(y, t):
            dxdt = y[2]
            dydt = y[3]
            return [dxdt, dydt, 0, 0]

        # create random policy
        nFeatures, nHidden1, nHidden21, nHidden22, nCls = 2, 128, 32, 32, 2
        mean = np.zeros(nFeatures)
        std = np.ones(nFeatures)
        model = BarrierPolicy(nFeatures, nHidden1, nHidden21, nHidden22, nCls,
                              mean=mean, std=std, device=device, bn=False).to(device)
        model.eval()

        # simulation
        obs_x, obs_y, R = model.obs_x, model.obs_y, model.R
        state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        mina, maxa = -1.0, 1.0
        dt = [0, 0.1]

        # running on a vehicle
        safety, loc_x, loc_y = [], [], []
        for i in range(5000):  # train0, 10
            obs = (state[:nFeatures] - mean) / std

            # get safety metric
            px, py, vx, vy = state
            safe = (px - obs_x) ** 2 + (py - obs_y) ** 2 - R ** 2

            safety.append(safe)
            loc_x.append(px)
            loc_y.append(py)

            # prepare for model input
            with torch.no_grad():
                x_r = Variable(torch.from_numpy(obs), requires_grad=False)
                x_r = torch.reshape(x_r, (1, nFeatures))
                x_r = x_r.to(device)
                ctrl = model(x_r, 0)

            # update state
            state[-2] = np.clip(ctrl[0], mina, maxa)
            state[-1] = np.clip(ctrl[1], mina, maxa)

            # update dynamics
            rt = np.float32(odeint(dynamics, state, dt))
            state[0] = rt[1][0]
            state[1] = rt[1][1]
            state[2] = rt[1][2]
            state[3] = rt[1][3]


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