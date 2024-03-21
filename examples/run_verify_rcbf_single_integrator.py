"""
Example of verifying a known valid CBF for single-integrator dynamics.
"""
import torch

from fosco.cegis import Cegis
from fosco.common.consts import CertificateType
from fosco.config import CegisConfig
from fosco.systems import make_system
from fosco.systems.uncertainty import add_uncertainty


def main(args):
    system_id = "SingleIntegrator"
    uncertainty_id = "AdditiveBounded"
    model_to_load = "default" #"tunable"
    verbose = 1

    system_fn = make_system(system_id=system_id)
    system_fn = add_uncertainty(uncertainty_type=uncertainty_id, system_fn=system_fn)
    sets = system_fn().domains

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
        CERTIFICATE="rcbf",
        VERIFIER="dreal",
        MODEL_TO_LOAD=model_to_load,
        CEGIS_MAX_ITERS=1,
        LOGGER="aim",
        EXP_NAME=f"RCBF_{model_to_load}",
    )
    cegis = Cegis(
        system=system_fn(),
        domains=sets,
        data_gen=data_gen,
        config=config,
        verbose=verbose
    )

    result = cegis.solve()
    print("result: ", result)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    main(args)
