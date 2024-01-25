"""
Example of verifying a known valid CBF for single-integrator dynamics.
"""
import torch

import fosco
from fosco.cegis import Cegis
from fosco.common.consts import CertificateType
from fosco.config import CegisConfig
from systems import make_system, add_uncertainty, make_domains


def main(args):
    system_id = "single_integrator"
    uncertainty_id = "additive_bounded"
    verbose = 1

    system_fn = make_system(system_id=system_id)
    system_fn = add_uncertainty(uncertainty_type=uncertainty_id, system_fn=system_fn)
    sets = make_domains(system_id=system_id)

    # data generator
    data_gen = {
        "init": lambda n: sets["init"].generate_data(n),
        "unsafe": lambda n: sets["unsafe"].generate_data(n),
        "lie": lambda n: torch.concatenate(
            [
                sets["lie"].generate_data(n),
                torch.zeros(n, sets["input"].dimension),
                sets["uncertainty"].generate_data(n),
            ],
            dim=1,
        ),
        "uncertainty": lambda n: torch.concatenate(
            [
                sets["lie"].generate_data(n),
                sets["input"].generate_data(n),
                sets["uncertainty"].generate_data(n),
            ],
            dim=1,
        ),
    }

    config = CegisConfig(
        SYSTEM=system_fn,
        DOMAINS=sets,
        DATA_GEN=data_gen,
        CERTIFICATE=CertificateType.RCBF,
        USE_INIT_MODELS=True,
        CEGIS_MAX_ITERS=1,
    )
    cegis = Cegis(config=config, verbose=verbose)

    result = cegis.solve()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    main(args)