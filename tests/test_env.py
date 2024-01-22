import unittest

from gymnasium.utils.env_checker import check_env

from fosco.common.domains import Rectangle, Sphere
from systems import make_system
from systems.system_env import SystemEnv


class TestEnv(unittest.TestCase):
    def test_single_integrator_system_env(self):
        svars, uvars = ["x0", "x1"], ["u0", "u1"]
        system = make_system(system_id="single_integrator")()
        domains = {
            "input": Rectangle(vars=uvars, lb=(-5.0, -5.0), ub=(5.0, 5.0)),
            "init": Rectangle(vars=svars, lb=(-5.0, -5.0), ub=(-4.0, -4.0)),
            "unsafe": Sphere(vars=svars, centre=[0.0, 0.0], radius=1.0),
        }

        env = SystemEnv(system=system, domains=domains)
        check_env(env, skip_render_check=True)

    def test_double_integrator_system_env(self):
        svars, uvars = ["x0", "x1", "x2", "x3"], ["u0", "u1"]
        system = make_system(system_id="double_integrator")()
        domains = {
            "input": Rectangle(vars=uvars, lb=(-5.0, -5.0), ub=(5.0, 5.0)),
            "init": Rectangle(
                vars=svars, lb=(-5.0, -5.0, -5.0, -5.0), ub=(-4.0, -4.0, 5.0, 5.0)
            ),
            "unsafe": Rectangle(
                vars=svars, lb=(-1.0, -1.0, -5.0, -5.0), ub=(1.0, 1.0, 5.0, 5.0)
            ),
        }

        env = SystemEnv(system=system, domains=domains)
        check_env(env, skip_render_check=True)
