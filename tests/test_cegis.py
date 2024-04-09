import unittest
from typing import Callable

import numpy as np
import torch

import fosco
from fosco.cegis import Cegis
from fosco.common.domains import Set
from fosco.config import CegisConfig
from fosco.common.consts import (
    TimeDomain,
    ActivationType,
    VerifierType,
    CertificateType,
    DomainName,
)
from fosco.systems import SingleIntegrator, ControlAffineDynamics


class TestCEGIS(unittest.TestCase):
    @staticmethod
    def _get_single_integrator_config() -> tuple[
        ControlAffineDynamics, dict[str, Set], dict[str, Callable], CegisConfig
    ]:
        system = SingleIntegrator()


        dn = DomainName
        domains = system.domains

        data_gen = {
            dn.XD.value: lambda n: torch.concatenate(
                [system.state_domain.generate_data(n), system.input_domain.generate_data(n)], dim=1
            ),
            dn.XI.value: lambda n: system.init_domain.generate_data(n),
            dn.XU.value: lambda n: system.unsafe_domain.generate_data(n),
        }

        config = CegisConfig(
            CERTIFICATE="cbf",
            VERIFIER="z3",
            CEGIS_MAX_ITERS=5,
            ROUNDING=3,
            N_DATA=1000,
            LEARNING_RATE=1e-3,
            WEIGHT_DECAY=1e-4,
            N_HIDDEN_NEURONS=(5, ),
            ACTIVATION=("square",),
            SEED=0,
        )

        return system, domains, data_gen, config

    def test_loop(self):
        system, domains, data_gen, config = self._get_single_integrator_config()
        config.N_EPOCHS = (
            0  # make sure we don't train to check correctness of the cegis loop
        )

        c = Cegis(
            system=system, domains=domains, config=config, data_gen=data_gen, verbose=0
        )
        results = c.solve()

        infos = results.infos
        self.assertTrue(
            infos["iter"] == config.CEGIS_MAX_ITERS,
            f"Did not run for {config.CEGIS_MAX_ITERS} iterations, iter={infos['iter']}",
        )

    def test_single_integrator_cbf_example(self):
        """
        Test the single integrator example. We expect to find a certificate in a couple of iterations.
        """

        system, domains, data_gen, config = self._get_single_integrator_config()
        config.SEED = 916104

        cegis = Cegis(
            system=system, domains=domains, config=config, data_gen=data_gen, verbose=0
        )

        result = cegis.solve()

        print("result: ", result)

        self.assertTrue(
            result.found,
            f"Did not find a certificate in {config.CEGIS_MAX_ITERS} iterations",
        )

    def test_single_integrator_rcbf_example(self):
        """
        Test the use of RCBF for single-integrator with additive uncertainty.
        Since at the moment, we dont have a simple example with uncertainty>0 that works in a couple of iter,
        this test simply checks that the code runs using a dummy zero uncertainty.
        """
        import fosco
        from fosco.systems import make_system
        from fosco.systems.uncertainty import add_uncertainty
        from fosco.common import domains

        seed = 916104
        system_name = "SingleIntegrator"
        activations = ("relu", "linear")
        n_data_samples = 1000
        n_hidden_neurons = (5,) * len(activations)
        certificate_type = "rcbf"
        verbose = 0

        system = make_system(system_id=system_name)()
        system = add_uncertainty(uncertainty_type="AdditiveBounded", system=system)

        sets = system.domains

        # quick test: zero uncertainty, cegis should find a certificate quickly
        sets["uncertainty"] = domains.Rectangle(vars=["z0", "z1"], lb=(0.0, 0.0), ub=(0.0, 0.0))

        data_gen = {
            "init": lambda n: system.init_domain.generate_data(n),
            "unsafe": lambda n: system.unsafe_domain.generate_data(n),
            "lie": lambda n: torch.concatenate(
                [system.state_domain.generate_data(n), system.input_domain.generate_data(n)], dim=1
            ),
            "uncertainty": lambda n: torch.concatenate(
                [system.state_domain.generate_data(n), system.input_domain.generate_data(n), sets["uncertainty"].generate_data(n)], dim=1
            ),
        }

        config = fosco.cegis.CegisConfig(
            CERTIFICATE=certificate_type,
            VERIFIER="z3",
            ACTIVATION=activations,
            N_HIDDEN_NEURONS=n_hidden_neurons,
            LOSS_RELU="softplus",
            N_EPOCHS=1000,
            CEGIS_MAX_ITERS=20,
            N_DATA=n_data_samples,
            SEED=seed,
        )
        cegis = Cegis(
            system=system, domains=sets, config=config, data_gen=data_gen, verbose=verbose
        )

        result = cegis.solve()

        print("result: ", result)

        self.assertTrue(
            result.found,
            f"Did not find a certificate in {config.CEGIS_MAX_ITERS} iterations",
        )

    def test_reproducibility(self):
        """
        Test seeding in cegis to ensure reproducibility.
        """
        for _ in range(1):
            seed = np.random.randint(0, 1000000)
            print("Random Seed:", seed)

            # first iter
            system, domains, data_gen, config = self._get_single_integrator_config()
            config.SEED = seed
            config.CEGIS_MAX_ITERS = 1
            cegis = fosco.cegis.Cegis(
                system=system,
                domains=domains,
                config=config,
                data_gen=data_gen,
                verbose=0,
            )
            results = cegis.solve()
            model = results.barrier
            params = list(model.parameters())

            for run_id in range(3):
                print(f"Running {run_id}")
                system, domains, data_gen, config = self._get_single_integrator_config()
                config.SEED = seed
                config.CEGIS_MAX_ITERS = 1

                cegis = fosco.cegis.Cegis(
                    system=system,
                    domains=domains,
                    config=config,
                    data_gen=data_gen,
                    verbose=0,
                )
                results = cegis.solve()
                model = results.barrier
                new_params = list(model.parameters())

                # check that the parameters are the same
                self.assertTrue(
                    all(torch.allclose(a, b) for a, b in zip(params, new_params)),
                    f"Parameters are not the same in run {run_id}",
                )
