"""
Example of verifying a known valid CBF for single-integrator dynamics with convex hull uncertainty.
"""
import torch
import numpy as np

from fosco.cegis import Cegis
from fosco.config import CegisConfig
from fosco.logger import LoggerType
from fosco.models import TorchSymFn
from fosco.systems import make_system
from fosco.systems.uncertainty import add_uncertainty


# we are instaniating this class one by one, or by batch?
class UncertainFunc(TorchSymFn):
    def __init__(self, uncertain_func) -> None:
        super().__init__()
        self.uncertain_func = uncertain_func
    
    # TODO
    def input_size(self) -> int:
        raise NotImplementedError

    def output_size(self) -> int:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: shape is (batch_size, state_dim) or (batch_size, state_dim, 1)
        """
        if len(x.size()) == 3:
            x = torch.squeeze(x, dim=2)
        return (x * torch.tensor(self.uncertain_func)).unsqueeze(dim=2)
    
    def forward_smt(self, x: list) -> list | np.ndarray:
        return np.array(self.uncertain_func) * np.array(x)

def main(args):
    system_id = "SingleIntegrator"
    uncertainty_id = "ConvexHull"
    verbose = 1
    
    model_to_load = "default"
    sigma_to_load = "ConvexHull"

    uncertain_func_list = [[0.2, 0.2], [0.3, 0.2], [0.5, 0.1]] # define uncertain function to be linear
    f_uncertainty = []
    for uncertain_func in uncertain_func_list:
        f_uncertainty.append(UncertainFunc(uncertain_func))

    system_fn = make_system(system_id=system_id)
    system_fn = add_uncertainty(uncertainty_type=uncertainty_id, system_fn=system_fn, f_uncertainty=f_uncertainty)

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
        VERIFIER="z3",
        BARRIER_TO_LOAD=model_to_load,
        SIGMA_TO_LOAD=sigma_to_load,
        CEGIS_MAX_ITERS=1,
        N_EPOCHS=0,
        EXP_NAME=f"RCBF_{model_to_load}",
    )
    
    cegis = Cegis(
        system=system_fn,
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
