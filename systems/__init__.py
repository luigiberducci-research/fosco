from typing import Type

from fosco.common import domains
from fosco.common.domains import Set
from .system import (
    ControlAffineDynamics,
    UncertainControlAffineDynamics,
)
from .single_integrator import SingleIntegrator
from .double_integrator import DoubleIntegrator
from .unicycle_model import Unicycle



def make_system(system_id: str) -> Type[ControlAffineDynamics]:
    """
    Factory function for systems.
    """
    if system_id in system.SYSTEM_REGISTRY:
        return system.SYSTEM_REGISTRY[system_id]
    else:
        raise NotImplementedError(f"System {system_id} not implemented")





def make_domains(system_id: str) -> dict[str, Set]:
    XD, UD, ZD, XI, XU = None, None, None, None, None
    if system_id == "SingleIntegrator":
        xvars = ["x0", "x1"]
        uvars = ["u0", "u1"]
        zvars = ["z0", "z1"]

        XD = domains.Rectangle(vars=xvars, lb=(-5.0, -5.0), ub=(5.0, 5.0))
        UD = domains.Rectangle(vars=uvars, lb=(-5.0, -5.0), ub=(5.0, 5.0))
        ZD = domains.Sphere(vars=zvars, centre=(0.0, 0.0), radius=1.0)
        XI = domains.Rectangle(vars=xvars, lb=(-5.0, -5.0), ub=(-4.0, -4.0))
        XU = domains.Sphere(
            vars=xvars, centre=[0.0, 0.0], radius=1.0, dim_select=[0, 1], include_boundary=False
        )
    elif system_id == "DoubleIntegrator":
        xvars = ["x0", "x1", "x2", "x3"]
        uvars = ["u0", "u1"]
        zvars = ["z0", "z1", "z2", "z3"]

        XD = domains.Rectangle(
            vars=xvars, lb=(-5.0,) * len(xvars), ub=(5.0,) * len(xvars)
        )
        UD = domains.Rectangle(
            vars=uvars, lb=(-5.0,) * len(uvars), ub=(5.0,) * len(uvars)
        )
        ZD = domains.Rectangle(
            vars=zvars, lb=(-1.0,) * len(zvars), ub=(1.0,) * len(zvars)
        )
        XI = domains.Rectangle(
            vars=xvars, lb=(-5.0,) * len(xvars), ub=(-4.0, -4.0, 5.0, 5.0)
        )
        XU = domains.Rectangle(
            vars=xvars, lb=(-1.0, -1.0, -5.0, -5.0), ub=(1.0, 1.0, 5.0, 5.0)
        )
    else:
        NotImplementedError(f"Domains for system {system_id} not implemented")

    return {
        "lie": XD,
        "input": UD,
        "uncertainty": ZD,
        "init": XI,
        "unsafe": XU,
    }
