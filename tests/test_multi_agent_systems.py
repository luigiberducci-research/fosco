import unittest

from fosco.common import domains
from fosco.systems import make_system


class TestMultiAgentSystems(unittest.TestCase):
    def test_multi_particle_single_integrator(self):
        si_system = make_system("SingleIntegrator")()
        system = make_system("MultiParticleSingleIntegrator")(
            n_agents=3,
            initial_distance=123.321,
            collision_distance=0.123,
        )

        self.assertEqual(system.id, "MultiParticleSingleIntegrator3")
        self.assertEqual(system.vars, ("dx0_1", "dx1_1", "dx0_2", "dx1_2"))
        self.assertEqual(system.controls, ("u0_0", "u1_0"))

        self.assertEqual(system.state_domain.vars, system.vars)
        self.assertEqual(system.state_domain.lower_bounds[:2], si_system.state_domain.lower_bounds)
        self.assertEqual(system.state_domain.upper_bounds[:2], si_system.state_domain.upper_bounds)

        self.assertEqual(system.input_domain.vars, system.controls)
        self.assertEqual(system.input_domain.lower_bounds, si_system.state_domain.lower_bounds)
        self.assertEqual(system.input_domain.upper_bounds, si_system.state_domain.upper_bounds)

        self.assertTrue(isinstance(system.unsafe_domain, domains.Sphere))
        self.assertEqual(system.unsafe_domain.vars, system.vars)
        self.assertEqual(system.unsafe_domain.center, (0, 0, 0, 0))
        self.assertEqual(system.unsafe_domain.radius, 0.123)

        self.assertTrue(isinstance(system.init_domain, domains.Complement))
        self.assertEqual(system.init_domain.vars, system.vars)
        self.assertEqual(system.init_domain.set.center, (0, 0, 0, 0))
        self.assertEqual(system.init_domain.set.radius, 123.321)

