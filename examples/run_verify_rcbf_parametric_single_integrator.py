"""
Example of verifying a known valid CBF for single-integrator dynamics with parametric uncertainty in the vector fields f and g.
""" 
import torch
import numpy as np

from fosco.cegis import Cegis
from fosco.common.consts import CertificateType
from fosco.config import CegisConfig
from fosco.common import domains
from fosco.logger import LoggerType
from fosco.systems import make_system
from fosco.systems.uncertainty import add_uncertainty


def main(args):
    system_id = "SingleIntegrator"
    uncertainty_id = "ParametricUncertainty"
    verbose = 1

    # \dot{x} = f(x) + g(x) @ u + unknown_f_matrix @ parameters_f + unknown_g_matrix @ diag(u) @ parameters_g
    # subject to: uncertain_bound_A @ [parameters_f parameters_g] <= uncertain_bound_b

    # unknow_matrixs are dependent on states, linear functions for examples
    # draw a specific f and g from paper's use case
    unkown_f_matrix = torch.rand(2, 2) 
    unkown_g_matrix = torch.rand(2, 2)

    # Question: what if some constraints are not satisfiable simulataneousely?
    # draw a specific A and b from paper's use case
    constrain_num = 5
    uncertain_bound_A = (torch.rand(constrain_num, 2) - 0.5) * 2.
    uncertain_bound_b = torch.rand(constrain_num)

    system_fn = make_system(system_id=system_id)
    system_fn = add_uncertainty(uncertainty_type=uncertainty_id, system_fn=system_fn, \
                                f_uncertainty=unkown_f_matrix, g_uncertainty=unkown_g_matrix, \
                                uncertain_bound_A=uncertain_bound_A, uncertain_bound_b=uncertain_bound_b)
    
    xvars = ["x0", "x1"]
    uvars = ["u0", "u1"]
    zvars = ["z0", "z1"]

    XD = domains.Rectangle(vars=xvars, lb=(-5.0, -5.0), ub=(5.0, 5.0))
    UD = domains.Rectangle(vars=uvars, lb=(-5.0, -5.0), ub=(5.0, 5.0))
    XI = domains.Rectangle(vars=xvars, lb=(-5.0, -5.0), ub=(-4.0, -4.0))
    XU = domains.Sphere(
        vars=xvars, centre=[0.0, 0.0], radius=1.0, dim_select=[0, 1], include_boundary=False
    )
    ZD = domains.Polytope(vars=zvars, lhs_A=uncertain_bound_A, rhs_b=uncertain_bound_b)


    sets ={"init": XI, 
           "unsafe": XU,
           "lie": XD,
           "uncertainty": ZD,
           "input": UD}

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
        LOGGER=LoggerType.AIM
    )
    cegis = Cegis(config=config, verbose=verbose)

    result = cegis.solve()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    main(args)
