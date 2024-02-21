import unittest

import numpy as np
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
    def test_system_envs_with_dummy_fns(self):
        for system_id in ["SingleIntegrator", "DoubleIntegrator"]:
            system = make_system(system_id=system_id)()

            env = SystemEnv(
                system=system,
                termination_fn=dummy_term_fn,
                reward_fn=dummy_reward_fn,
            )
            check_env(env, skip_render_check=True)

    def test_numpy_batch_step(self):
        batch_size = 1000

        for system_id in ["SingleIntegrator", "DoubleIntegrator"]:
            system = make_system(system_id=system_id)()
            env = SystemEnv(
                system=system, termination_fn=dummy_term_fn, reward_fn=dummy_reward_fn
            )

            obss, infos = env.reset(options={"batch_size": batch_size})
            actions = np.stack([env.action_space.sample() for _ in range(batch_size)])
            next_obss, rewards, terminations, truncations, infos = env.step(actions)

            self.assertTrue(
                isinstance(obss, np.ndarray),
                f"next_obss is not a numpy array, got {type(next_obss)}",
            )
            self.assertTrue(
                isinstance(next_obss, np.ndarray),
                f"next_obss is not a numpy array, got {type(next_obss)}",
            )
            self.assertTrue(
                obss.shape[0] == actions.shape[0] == next_obss.shape[0],
                "mismatch batch sizes",
            )

    def test_tensor_batch_step(self):
        batch_size = 1000

        for system_id in ["SingleIntegrator", "DoubleIntegrator"]:
            system = make_system(system_id=system_id)()
            env = SystemEnv(
                system=system,
                termination_fn=dummy_term_fn,
                reward_fn=dummy_reward_fn,
                return_np=False,
            )

            obss, infos = env.reset(options={"batch_size": batch_size})
            actions = np.stack([env.action_space.sample() for _ in range(batch_size)])
            next_obss, rewards, terminations, truncations, infos = env.step(actions)

            self.assertTrue(
                isinstance(obss, torch.Tensor),
                f"next_obss is not a numpy array, got {type(next_obss)}",
            )
            self.assertTrue(
                isinstance(next_obss, torch.Tensor),
                f"next_obss is not a numpy array, got {type(next_obss)}",
            )
            self.assertTrue(
                len(obss.shape) == len(actions.shape) == len(next_obss.shape),
                "mismatch batch dimensions",
            )
            self.assertTrue(
                obss.shape[0] == actions.shape[0] == next_obss.shape[0],
                "mismatch batch sizes",
            )

    def test_sequential_step(self):
        batch_size = 1000

        for system_id in ["SingleIntegrator", "DoubleIntegrator"]:
            system = make_system(system_id=system_id)()
            env = SystemEnv(
                system=system, termination_fn=dummy_term_fn, reward_fn=dummy_reward_fn
            )

            for i in range(batch_size):
                obss, infos = env.reset()
                actions = env.action_space.sample()
                next_obss, rewards, terminations, truncations, infos = env.step(actions)

            self.assertTrue(
                isinstance(obss, np.ndarray),
                f"next_obss is not a numpy array, got {type(next_obss)}",
            )
            self.assertTrue(
                isinstance(next_obss, np.ndarray),
                f"next_obss is not a numpy array, got {type(next_obss)}",
            )
            self.assertTrue(
                len(obss.shape) == len(actions.shape) == len(next_obss.shape) == 1,
                "mismatch batch sizes",
            )
