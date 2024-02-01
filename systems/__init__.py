from typing import Type

from fosco.common import domains
from fosco.common.domains import Set
from .system import (
    ControlAffineDynamics,
    UncertainControlAffineDynamics,
)


def make_system(system_id: str) -> Type[ControlAffineDynamics]:
    if system_id == "single_integrator":
        from systems.single_integrator import SingleIntegrator

        return SingleIntegrator
    elif system_id == "double_integrator":
        from systems.double_integrator import DoubleIntegrator

        return DoubleIntegrator
    else:
        raise NotImplementedError(f"System {system_id} not implemented")


def add_uncertainty(uncertainty_type: str | None, system_fn: callable) -> callable:
    from systems.uncertainty.additive_bounded import AdditiveBoundedUncertainty

    if uncertainty_type is None:
        return system_fn
    if uncertainty_type == "additive_bounded" or "tunable_additive_bounded":
        return lambda: AdditiveBoundedUncertainty(system=system_fn())
    else:
        raise NotImplementedError(f"Uncertainty {uncertainty_type} not implemented")


def make_domains(system_id: str) -> dict[str, Set]:
    XD, UD, ZD, XI, XU = None, None, None, None, None
    if system_id == "single_integrator":
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
    elif system_id == "double_integrator":
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
