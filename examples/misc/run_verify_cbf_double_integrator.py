"""
Example of verifying a known valid CBF for single-integrator dynamics.
"""

from typing import Optional

import torch

from fosco.cegis import Cegis
from fosco.config import CegisConfig
from fosco.plotting.functions import plot_torch_function
from fosco.systems import make_system
from fosco.systems.discrete_time.system_dt import EulerDTSystem


def main(args):
    system_id: str = "DoubleIntegrator"  # system id
    dt: Optional[float] = (
        None  # discretization time step or None for continuous-time cbf
    )
    verbose: int = 1  # verbosity level (0, 1, 2)

    # make system
    system = make_system(system_id=system_id)()
    if dt is not None:
        system = EulerDTSystem(system=system, dt=dt)
    sets = system.domains

    # data generator
    data_gen = {
        "init": lambda n: sets["init"].generate_data(n),
        "unsafe": lambda n: sets["unsafe"].generate_data(n),
        "lie": lambda n: torch.concatenate(
            [sets["lie"].generate_data(n), sets["input"].generate_data(n)], dim=1
        ),
    }

    config = CegisConfig(
        CERTIFICATE="cbf", BARRIER_TO_LOAD="default", CEGIS_MAX_ITERS=1,
    )
    cegis = Cegis(
        system=system, domains=sets, data_gen=data_gen, config=config, verbose=verbose,
    )

    result = cegis.solve()

    print("result: ", result)

    fig = plot_torch_function(function=result.barrier, domains=sets)
    fig.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    main(args)
