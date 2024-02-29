"""
Example of verifying a known valid CBF for single-integrator dynamics with convex hull uncertainty.
"""
import torch
import numpy as np

from fosco.cegis import Cegis
from fosco.common.consts import CertificateType
from fosco.config import CegisConfig
from fosco.common import domains
from systems import make_system, make_domains
from systems.uncertainty import add_uncertainty
from models.torchsym import TorchSymFn

# we are instaniating this class one by one, or by batch?
class UncertainFunc(TorchSymFn):
    def __init__(self, uncertain_func) -> None:
        super().__init__()
        self.uncertain_func = uncertain_func

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # check the dim consistence, batch input, etc
        if len(x.size()) == 2:
            x = torch.unsqueeze(x, dim=2)
        return (x.squeeze(dim=2) * torch.tensor(self.uncertain_func)).unsqueeze(dim=2)
    
    def forward_smt(self, x: list) -> list | np.ndarray:
        return np.array(self.uncertain_func) * np.array(x)

def main(args):
    system_id = "SingleIntegrator"
    uncertainty_id = "ConvexHull"
    verbose = 1

    uncertain_func_list = [[0.2, 0.2], [0.3, 0.2], [0.5, 0.1]] # linear functions
    uncertain_func_coefficiet_list = [0.3, 0.3, 0.4] 
    f_uncertainty = []
    for uncertain_func in uncertain_func_list:
        f_uncertainty.append(UncertainFunc(uncertain_func))

    system_fn = make_system(system_id=system_id)
    system_fn = add_uncertainty(uncertainty_type=uncertainty_id, system_fn=system_fn, f_uncertainty=f_uncertainty, 
                                f_uncertainty_params=uncertain_func_coefficiet_list)
    
    xvars = ["x0", "x1"]
    uvars = ["u0", "u1"]
    zvars = ["z0", "z1", "z2"]

    XD = domains.Rectangle(vars=xvars, lb=(-5.0, -5.0), ub=(5.0, 5.0))
    UD = domains.Rectangle(vars=uvars, lb=(-5.0, -5.0), ub=(5.0, 5.0))
    XI = domains.Rectangle(vars=xvars, lb=(-5.0, -5.0), ub=(-4.0, -4.0))
    XU = domains.Sphere(
        vars=xvars, centre=[0.0, 0.0], radius=1.0, dim_select=[0, 1], include_boundary=False
    )
    # todo: the domain of z has to be s.t. sum_i z_i = 1.0
    # ZD = domains.Sphere(vars=zvars, centre=(0.0, 0.0, 0.0), radius=1.0)
    ZD = domains.SumToOneSet(vars=zvars)


    sets ={"init": XI, 
           "unsafe": XU,
           "lie": XD,
           "uncertainty": ZD,
           "input": UD}
    
    # sets = make_domains(system_id=system_id)

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
