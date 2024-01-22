import random

import torch

import fosco.cegis
from fosco.common.consts import ActivationType, LossReLUType
from fosco.common.consts import CertificateType, TimeDomain, VerifierType
from logger import LoggerType
from systems import make_system, make_domains


def main():
    seed = 916104
    system_name = "single_integrator"
    certificate_type = CertificateType.CBF
    activations = (ActivationType.RELU, ActivationType.LINEAR)
    n_hidden_neurons = (5,) * len(activations)
    n_data_samples = 1000
    max_iters = 100
    n_epochs = 1000
    verbose = 2

    system = make_system(system_id=system_name)
    sets = make_domains(system_id=system_name)
    if not certificate_type == CertificateType.RCBF:
        sets = {k: s for k, s in sets.items() if k in ["lie", "input", "init", "unsafe"]}

    # todo seeding in cegis, test reproducibility
    if seed is None:
        seed = random.randint(0, 1000000)
    print("Seed:", seed)

    # data generator
    data_gen = {
        "init": lambda n: sets["init"].generate_data(n),
        "unsafe": lambda n: sets["unsafe"].generate_data(n),
    }

    if certificate_type == CertificateType.CBF:
        data_gen["lie"] = lambda n: torch.concatenate(
                [sets["lie"].generate_data(n), sets["input"].generate_data(n)], dim=1
            )
    else:
        data_gen["lie"] = lambda n: torch.concatenate(
                [sets["lie"].generate_data(n),
                torch.zeros(n, sets["input"].dimension), sets["uncertainty"].generate_data(n)], dim=1
            )
        data_gen["uncertainty"] = lambda n: torch.concatenate(
                [sets["lie"].generate_data(n), sets["input"].generate_data(n), sets["uncertainty"].generate_data(n)], dim=1
            )

    config = fosco.cegis.CegisConfig(
        SYSTEM=system,
        DOMAINS=sets,
        DATA_GEN=data_gen,
        CERTIFICATE=certificate_type,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.Z3,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        CEGIS_MAX_ITERS=max_iters,
        N_DATA=n_data_samples,
        SEED=seed,
        LOGGER=LoggerType.AIM,
        N_EPOCHS=n_epochs,
        LOSS_MARGINS={"init": 0.0, "unsafe": 0.0, "lie": 0.0, "robust": 0.0},
        LOSS_WEIGHTS={"init": 1.0, "unsafe": 1.0, "lie": 1.0, "robust": 1.0},
        LOSS_RELU=LossReLUType.RELU,
    )
    cegis = fosco.cegis.Cegis(config=config, verbose=verbose)

    result = cegis.solve()


if __name__ == "__main__":
    main()
