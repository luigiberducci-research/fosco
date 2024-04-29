import unittest

import numpy as np
import torch

from fosco.common.consts import VerifierType
from fosco.systems import make_system
from fosco.systems.multi_agent_particles import MultiParticle
from fosco.verifier import make_verifier


class TestMultiAgentSystem(unittest.TestCase):
    def test_system_variables(self):
        single_sys = make_system("SingleIntegrator")()
        multi_sys = MultiParticle(single_agent_dynamics=single_sys, n_agents=3)

        self.assertEqual(multi_sys.id, "MultiParticleSingleIntegrator3")
        self.assertEqual(
            multi_sys.vars, ("x0_0", "x1_0", "dx0_1", "dx1_1", "dx0_2", "dx1_2")
        )
        self.assertEqual(multi_sys.controls, ("u0_0", "u1_0"))

    def test_domains(self):
        collision_distance = 1.0
        initial_distance = 3.0

        single_sys = make_system("SingleIntegrator")()
        multi_sys = MultiParticle(
            single_agent_dynamics=single_sys,
            n_agents=3,
            collision_distance=collision_distance,
            initial_distance=initial_distance,
        )

        self.assertEqual(multi_sys.time_domain, single_sys.time_domain)
        self.assertTrue(
            multi_sys.state_domain.lower_bounds[:2]
            == single_sys.state_domain.lower_bounds
        )
        self.assertTrue(
            multi_sys.state_domain.upper_bounds[:2]
            == single_sys.state_domain.upper_bounds
        )
        self.assertTrue(
            multi_sys.input_domain.lower_bounds == single_sys.input_domain.lower_bounds
        )
        self.assertTrue(
            multi_sys.input_domain.upper_bounds == single_sys.input_domain.upper_bounds
        )

        # to check unsafe set, use samples
        samples = multi_sys.unsafe_domain.generate_data(1000)
        min_distances = torch.min(torch.abs(samples), axis=1)
        self.assertTrue(torch.all(min_distances.values < collision_distance))

        # to check init set, use samples
        # for each pair, at least one of the dimension has to be > initial_distance
        samples = multi_sys.init_domain.generate_data(1000)
        for aid in range(1, multi_sys._n_agents):
            max_distance = torch.max(
                torch.abs(samples[:, 2 * aid : 2 * (aid + 1)]), axis=1
            )
            self.assertTrue(torch.all(max_distance.values > initial_distance))

    def test_dynamics(self):
        single_sys = make_system("SingleIntegrator")()
        multi_sys = MultiParticle(single_agent_dynamics=single_sys, n_agents=3,)

        states = multi_sys.state_domain.generate_data(10).reshape(10, -1, 1)

        # check the dynamics numerically
        fx = multi_sys.fx_torch(states)
        gx = multi_sys.gx_torch(states)

        self.assertTrue(fx.shape[0] == states.shape[0])
        self.assertTrue(fx.shape[1] == multi_sys.n_vars)
        self.assertTrue(fx.shape[2] == 1)

        self.assertTrue(gx.shape[0] == states.shape[0])
        self.assertTrue(gx.shape[1] == multi_sys.n_vars)
        self.assertTrue(gx.shape[2] == multi_sys.n_controls)

        states_np = states.numpy()
        fx_np = multi_sys.fx_torch(states_np)
        gx_np = multi_sys.gx_torch(states_np)

        self.assertTrue(fx_np.shape[0] == states.shape[0])
        self.assertTrue(fx_np.shape[1] == multi_sys.n_vars)
        self.assertTrue(fx_np.shape[2] == 1)

        self.assertTrue(gx_np.shape[0] == states.shape[0])
        self.assertTrue(gx_np.shape[1] == multi_sys.n_vars)
        self.assertTrue(gx_np.shape[2] == multi_sys.n_controls)

        self.assertTrue(np.allclose(fx_np, fx.numpy()))

        # check the dynamics symbolically
        verifier_fn = make_verifier(type=VerifierType.Z3)
        vars = verifier_fn.new_vars(var_names=list(multi_sys.vars))

        fx_smt = multi_sys.fx_smt(vars)
        gx_smt = multi_sys.gx_smt(vars)

        self.assertTrue(len(fx_smt) == multi_sys.n_vars)
        self.assertTrue(len(gx_smt) == multi_sys.n_vars)
        self.assertTrue(len(gx_smt[0]) == multi_sys.n_controls)

        # fx all zeros
        self.assertTrue(all(fx_smt == 0))

        # gx = [I, -I, -I, -I]
        self.assertTrue(np.allclose(gx_smt[:2, :], np.eye(multi_sys.n_controls)))
        for i in range(1, multi_sys._n_agents):
            self.assertTrue(
                np.allclose(
                    gx_smt[i * 2 : (i + 1) * 2, :], -np.eye(multi_sys.n_controls)
                )
            )
