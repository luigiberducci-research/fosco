import unittest

import torch

from fosco.cegis import Cegis
from fosco.config import CegisConfig
from fosco.common.domains import Rectangle, Sphere
from fosco.common.consts import (
    TimeDomain,
    ActivationType,
    VerifierType,
    CertificateType,
    DomainNames,
)
from systems.single_integrator import SingleIntegrator


class TestCEGIS(unittest.TestCase):
    def _get_single_integrator_config(self) -> CegisConfig:
        XD = Rectangle(vars=["x0", "x1"], lb=(-5.0, -5.0), ub=(5.0, 5.0))
        UD = Rectangle(vars=["u0", "u1"], lb=(-5.0, -5.0), ub=(5.0, 5.0))
        XI = Rectangle(vars=["x0", "x1"], lb=(-5.0, -5.0), ub=(-4.0, -4.0))
        XU = Sphere(vars=["x0", "x1"], centre=[0.0, 0.0], radius=1.0, dim_select=[0, 1])

        dn = DomainNames
        domains = {
            name: domain
            for name, domain in zip(
                [dn.XD.value, dn.UD.value, dn.XI.value, dn.XU.value], [XD, UD, XI, XU]
            )
        }

        data_gen = {
            dn.XD.value: lambda n: torch.concatenate(
                [XD.generate_data(n), UD.generate_data(n)], dim=1
            ),
            dn.XI.value: lambda n: XI.generate_data(n),
            dn.XU.value: lambda n: XU.generate_data(n),
        }

        config = CegisConfig(
            SYSTEM=SingleIntegrator,
            DOMAINS=domains,
            TIME_DOMAIN=TimeDomain.CONTINUOUS,
            CERTIFICATE=CertificateType.CBF,
            VERIFIER=VerifierType.Z3,
            CEGIS_MAX_ITERS=5,
            ROUNDING=3,
            DATA_GEN=data_gen,
            N_DATA=500,
            LEARNING_RATE=1e-3,
            WEIGHT_DECAY=1e-4,
            N_HIDDEN_NEURONS=(10, 10,),
            ACTIVATION=(ActivationType.RELU, ActivationType.LINEAR),
            SEED=0,
        )

        return config

    def test_loop(self):
        config = self._get_single_integrator_config()
        config.N_EPOCHS = 0  # make sure we don't train to check correctness of the cegis loop

        c = Cegis(config=config, verbose=2)
        results = c.solve()

        infos = results.infos
        self.assertTrue(
            infos["iter"] == config.CEGIS_MAX_ITERS,
            f"Did not run for {config.CEGIS_MAX_ITERS} iterations, iter={infos['iter']}",
        )

    def test_single_integrator_example(self):
        """
        Test the single integrator example. We expect to find a certificate in a couple of iterations.
        """
        import fosco
        from systems import make_system
        from fosco.common import domains

        seed = 916104
        system_name = "single_integrator"
        n_hidden_neurons = 5
        activations = (ActivationType.RELU, ActivationType.LINEAR)
        n_data_samples = 1000
        n_hidden_neurons = (n_hidden_neurons,) * len(activations)
        certificate_type = CertificateType.CBF
        verbose = 0

        system = make_system(id=system_name)

        XD = domains.Rectangle(vars=["x0", "x1"], lb=(-5.0, -5.0), ub=(5.0, 5.0))
        UD = domains.Rectangle(vars=["u0", "u1"], lb=(-5.0, -5.0), ub=(5.0, 5.0))
        XI = domains.Rectangle(vars=["x0", "x1"], lb=(-5.0, -5.0), ub=(-4.0, -4.0))
        XU = domains.Sphere(vars=["x0", "x1"], centre=[0.0, 0.0], radius=1.0, dim_select=[0, 1])

        sets = {
            "lie": XD,
            "input": UD,
            "init": XI,
            "unsafe": XU,
        }
        data_gen = {
            "lie": lambda n: torch.concatenate([XD.generate_data(n), UD.generate_data(n)], dim=1),
            "init": lambda n: XI.generate_data(n),
            "unsafe": lambda n: XU.generate_data(n),
        }

        config = fosco.cegis.CegisConfig(
            SYSTEM=system,
            DOMAINS=sets,
            DATA_GEN=data_gen,
            CERTIFICATE=certificate_type,
            TIME_DOMAIN=TimeDomain.CONTINUOUS,
            VERIFIER=VerifierType.Z3,
            ACTIVATION=activations,
            N_HIDDEN_NEURONS=n_hidden_neurons,
            LOSS_RELU=fosco.common.consts.LossReLUType.SOFTPLUS,
            N_EPOCHS=1000,
            CEGIS_MAX_ITERS=5,
            N_DATA=n_data_samples,
            SEED=seed,
        )
        cegis = fosco.cegis.Cegis(config=config, verbose=verbose)

        result = cegis.solve()

        print("result: ", result)

        self.assertTrue(result.found, f"Did not find a certificate in {config.CEGIS_MAX_ITERS} iterations")

