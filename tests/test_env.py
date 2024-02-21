import unittest

import torch
from gymnasium.utils.env_checker import check_env

from fosco.common.domains import Rectangle, Sphere
from systems import make_system
from systems.system_env import SystemEnv


def dummy_term_fn(actions, next_obss):
    return torch.zeros((actions.shape[0],), dtype=torch.bool)


def dummy_reward_fn(actions, next_obss):
    return torch.ones((next_obss.shape[0],), dtype=torch.float32)

class TestEnv(unittest.TestCase):
    def test_single_integrator_system_env(self):
        system = make_system(system_id="SingleIntegrator")()

        env = SystemEnv(
            system=system,
            termination_fn=dummy_term_fn,
            reward_fn=dummy_reward_fn,
        )
        check_env(env, skip_render_check=True)

    def test_double_integrator_system_env(self):
        system = make_system(system_id="DoubleIntegrator")()
        env = SystemEnv(
            system=system,
            termination_fn=dummy_term_fn,
            reward_fn=dummy_reward_fn
        )
        check_env(env, skip_render_check=True)
