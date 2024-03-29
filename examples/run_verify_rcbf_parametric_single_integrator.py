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

    model_to_load = "default"
    sigma_to_load = "default"

    # \dot{x} = f(x) + g(x) @ u + unknown_f_matrix @ parameters_f + unknown_g_matrix @ diag(u) @ parameters_g
    # subject to: uncertain_bound_A @ [parameters_f parameters_g] <= uncertain_bound_b
    # paper: https://arxiv.org/pdf/2208.05955.pdf

    # note that the original paper is not using exactly the single integrator, so i am using a hand-crafted uncertain single integrator sys
    # unkown_f = [theta1*x1 theta2*x2] = f_matrix @ [x x] @ [theta1 theta2]
    # unkown_g = torch.zeros
    f_matrix = torch.eye(2)
    g_matrix = torch.zeros((2, 2))

    # two parameters, [-0.1, 0.1] x [-0.1, 0.1]
    uncertain_para_num = 2
    uncertain_bound_A = torch.cat((torch.eye(uncertain_para_num), -torch.eye(uncertain_para_num)), 0)
    uncertain_bound_b = torch.tensor([0.2, 0.1, 0.2, 0.1])

    system_fn = make_system(system_id=system_id)
    system_fn = add_uncertainty(uncertainty_type=uncertainty_id, system_fn=system_fn, \
                                f_uncertainty=f_matrix, g_uncertainty=g_matrix, \
                                uncertain_bound_A=uncertain_bound_A, uncertain_bound_b=uncertain_bound_b)
    
    xvars = ["x0", "x1"]
    uvars = ["u0", "u1"]
    zvars = ["z0", "z1"]

    XD = domains.Rectangle(vars=xvars, lb=(-5.0, -5.0), ub=(5.0, 5.0))
    UD = domains.Rectangle(vars=uvars, lb=(-5.0, -5.0), ub=(5.0, 5.0))
    XI = domains.Rectangle(vars=xvars, lb=(-5.0, -5.0), ub=(-4.0, -4.0))
    XU = domains.Sphere(
        vars=xvars, center=[0.0, 0.0], radius=1.0, dim_select=[0, 1], include_boundary=False
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
        CERTIFICATE="rcbf",
        VERIFIER="z3",
        BARRIER_TO_LOAD=model_to_load,
        SIGMA_TO_LOAD=sigma_to_load,
        CEGIS_MAX_ITERS=1,
        N_EPOCHS=0,
        EXP_NAME=f"RCBF_{model_to_load}",
    )
    cegis = Cegis(
        system=system_fn(),
        domains=sets,
        data_gen=data_gen,
        config=config,
        verbose=verbose,
    )

    result = cegis.solve()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    main(args)
