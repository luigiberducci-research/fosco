import logging
import random
import time

import torch
from matplotlib import pyplot as plt

import fosco.cegis
from fosco.common import domains
from fosco.common.consts import ActivationType
from fosco.common.consts import CertificateType, TimeDomain, VerifierType
from fosco.common.plotting import benchmark_3d, benchmark_plane, benchmark_lie
from logger import LoggerType
from systems import make_system
from systems.system import UncertainControlAffineControllableDynamicalModel


def main():
    seed = 916104
    system_name = "noisy_single_integrator"
    n_hidden_neurons = 5
    activations = (ActivationType.RELU, ActivationType.LINEAR)
    n_data_samples = 100
    verbose = 1

    log_levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    logging.basicConfig(level=log_levels[verbose])

    n_hidden_neurons = (n_hidden_neurons,) * len(activations)

    system = make_system(id=system_name)
    ZD = None   # no uncertainty by default
    if system_name == "single_integrator":
        XD = domains.Rectangle(vars=["x0", "x1"], lb=(-5.0, -5.0), ub=(5.0, 5.0))
        UD = domains.Rectangle(vars=["u0", "u1"], lb=(-5.0, -5.0), ub=(5.0, 5.0))
        XI = domains.Rectangle(vars=["x0", "x1"], lb=(-5.0, -5.0), ub=(-4.0, -4.0))
        XU = domains.Sphere(
            vars=["x0", "x1"], centre=[0.0, 0.0], radius=1.0, dim_select=[0, 1]
        )
    elif system_name == "noisy_single_integrator":
        XD = domains.Rectangle(vars=["x0", "x1"], lb=(-5.0, -5.0), ub=(5.0, 5.0))
        UD = domains.Rectangle(vars=["u0", "u1"], lb=(-5.0, -5.0), ub=(5.0, 5.0))
        ZD = domains.Rectangle(vars=["z0", "z1"], lb=(-5.0, -5.0), ub=(5.0, 5.0))
        XI = domains.Rectangle(vars=["x0", "x1"], lb=(-5.0, -5.0), ub=(-4.0, -4.0))
        XU = domains.Sphere(
            vars=["x0", "x1"], centre=[0.0, 0.0], radius=1.0, dim_select=[0, 1]
        )

    elif system_name == "double_integrator":
        XD = domains.Rectangle(
            vars=["x0", "x1", "x2", "x3"],
            lb=(-5.0, -5.0, -5.0, -5.0),
            ub=(5.0, 5.0, 5.0, 5.0),
        )
        UD = domains.Rectangle(vars=["u0", "u1"], lb=(-5.0, -5.0), ub=(5.0, 5.0))
        XI = domains.Rectangle(
            vars=["x0", "x1", "x2", "x3"],
            lb=(-5.0, -5.0, -5.0, -5.0),
            ub=(-4.0, -4.0, 5.0, 5.0),
        )
        XU = domains.Rectangle(
            vars=["x0", "x1", "x2", "x3"],
            lb=(-1.0, -1.0, -5.0, -5.0),
            ub=(1.0, 1.0, 5.0, 5.0),
        )
    else:
        raise NotImplementedError(f"System {system_name} not implemented")

    # seeding
    if seed is None:
        seed = random.randint(0, 1000000)
    print("Seed:", seed)

    # add uncertainty if applicable
    # todo: clean this
    if ZD is None:
        sets = {
            "lie": XD,
            "input": UD,
            "init": XI,
            "unsafe": XU,
        }
        data_gen = {
            "lie": lambda n: torch.concatenate(
                [XD.generate_data(n), UD.generate_data(n)], dim=1
            ),
            "init": lambda n: XI.generate_data(n),
            "unsafe": lambda n: XU.generate_data(n),
        }
        certificate_type = CertificateType.CBF
    else:
        sets = {
            "lie": XD,
            "input": UD,
            "uncertainty": ZD,
            "init": XI,
            "unsafe": XU,
        }
        data_gen = {
            "init": lambda n: XI.generate_data(n),
            "unsafe": lambda n: XU.generate_data(n),
            "lie": lambda n: torch.concatenate(
                [XD.generate_data(n),
                torch.zeros(n, UD.dimension), ZD.generate_data(n)], dim=1
            ),
            "uncertainty": lambda n: torch.concatenate(
                [XD.generate_data(n), UD.generate_data(n), ZD.generate_data(n)], dim=1
            ),
        }
        certificate_type = CertificateType.RCBF

    config = fosco.cegis.CegisConfig(
        SYSTEM=system,
        DOMAINS=sets,
        DATA_GEN=data_gen,
        CERTIFICATE=certificate_type,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.Z3,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        CEGIS_MAX_ITERS=500,
        N_DATA=n_data_samples,
        SEED=seed,
        LOGGER=LoggerType.AIM,
        LOSS_WEIGHTS={"init": 1.0, "unsafe": 1.0, "lie": 1.0, "robust": 1.0},
    )
    cegis = fosco.cegis.Cegis(config=config, verbose=verbose)

    result = cegis.solve()

    # save model
    result.net.save("tests/cbf_single_int")


if __name__ == "__main__":
    main()
