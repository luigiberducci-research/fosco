import unittest

import numpy as np
import torch

from fosco.common.domains import Rectangle, Sphere
from fosco.models.network import TorchMLP
from systems import make_system
from systems.system_env import SystemEnv
from fosco.planner.safety_filter import CBFSafetyFilter


class TestPlanner(unittest.TestCase):
    def _test_single_integrator_cbf_planner(self):
        # make env for single integrator
        svars, uvars = ["x0", "x1"], ["u0", "u1"]
        system = make_system(id="single_integrator")()
        domains = {
            "input": Rectangle(vars=uvars, lb=(-5.0, -5.0), ub=(5.0, 5.0)),
            "init": Rectangle(vars=svars, lb=(-5.0, -5.0), ub=(-4.0, -4.0)),
            "unsafe": Sphere(vars=svars, centre=[0.0, 0.0], radius=1.0),
        }
        env = SystemEnv(system=system, domains=domains)

        # load cbf torch model
        cbf_model = TorchMLP.load("cbf_single_int")
        # make planner
        cbf = CBFSafetyFilter(env=env, h_model=cbf_model)

        # simulate
        truncated, terminated = False, False
        observation, _ = env.reset()
        while not truncated and not terminated:
            nom_action = np.ones(2)
            safe_action = cbf(action=nom_action, observation=observation)
            observation, reward, terminated, truncated, info = env.step(
                action=safe_action
            )
            env.render()
            print(safe_action)

        env.close()
