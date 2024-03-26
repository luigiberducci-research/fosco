import numpy as np
import torch

from fosco.common import domains
from fosco.common.consts import TimeDomain
from fosco.common.domains import Set
from fosco.systems import SingleIntegrator
from fosco.systems.core.system import register, ControlAffineDynamics


class StaticOffice(ControlAffineDynamics):
    """
    Extends a given dynamical system with domains of a static office environment.
    """

    def __init__(self, system: ControlAffineDynamics):
        super().__init__()
        self._system = system

    @property
    def vars(self) -> tuple[str, ...]:
        return self._system.vars

    @property
    def controls(self) -> tuple[str, ...]:
        return self._system.controls

    @property
    def time_domain(self) -> TimeDomain:
        return TimeDomain.DISCRETE

    def fx_torch(self, x) -> np.ndarray | torch.Tensor:
        return self._system.fx_torch(x)

    def fx_smt(self, x) -> np.ndarray | torch.Tensor:
        return self._system.fx_smt(x)

    def gx_torch(self, x) -> np.ndarray | torch.Tensor:
        return self._system.gx_torch(x)

    def gx_smt(self, x) -> np.ndarray | torch.Tensor:
        return self._system.gx_smt(x)

    @property
    def id(self) -> str:
        return self.__class__.__name__

    @property
    def state_domain(self) -> Set:
        # ensure original domain is a rectangle
        system_domain = self._system.state_domain
        if not isinstance(system_domain, domains.Rectangle):
            raise TypeError(f"State domain {system_domain} is not a Rectangle")

        lbounds, ubounds = system_domain.lower_bounds, system_domain.upper_bounds

        # override x, y (first two dims)
        lbounds = (-10.0, -10.0,) + lbounds[2:]
        ubounds = (10.0, 10.0,) + ubounds[2:]

        return domains.Rectangle(vars=self.vars, lb=lbounds, ub=ubounds,)

    @property
    def input_domain(self) -> Set:
        return self._system.input_domain

    @property
    def init_domain(self) -> Set:
        # ensure original domain is a rectangle
        system_domain = self._system.state_domain
        if not isinstance(system_domain, domains.Rectangle):
            raise TypeError(f"State domain {system_domain} is not a Rectangle")

        lbounds, ubounds = system_domain.lower_bounds, system_domain.upper_bounds

        lbounds = (-10.0, -10.0,) + lbounds[2:]
        ubounds = (-9.0, -9.0,) + ubounds[2:]

        return domains.Rectangle(vars=self.vars, lb=lbounds, ub=ubounds,)

    @property
    def unsafe_domain(self) -> Set:
        human_radius = 0.36
        plant_radius = 0.4

        humans = [
            domains.Sphere(
                vars=self.vars,
                center=(0.44956875, 3.6988103),
                radius=human_radius,
                include_boundary=False,
            ),
            domains.Sphere(
                vars=self.vars,
                center=(4.0826755, -1.6512532),
                radius=human_radius,
                include_boundary=False,
            ),
            domains.Sphere(
                vars=self.vars,
                center=(3.0811677, -1.0228925),
                radius=human_radius,
                include_boundary=False,
            ),
            domains.Sphere(
                vars=self.vars,
                center=(3.03672, -2.3108509),
                radius=human_radius,
                include_boundary=False,
            ),
            domains.Sphere(
                vars=self.vars,
                center=(-0.69713455, 1.9523418),
                radius=human_radius,
                include_boundary=False,
            ),
        ]
        plants = [
            domains.Sphere(
                vars=self.vars,
                center=(2.2344275, 3.9294362),
                radius=plant_radius,
                include_boundary=False,
            ),
            domains.Sphere(
                vars=self.vars,
                center=(-2.9687812, -4.31725),
                radius=plant_radius,
                include_boundary=False,
            ),
        ]

        table_cx = -0.6971345718589101
        table_cy = 0.7023418244845709
        table_w = 1.5
        table_l = 3.0
        tables = [
            domains.Rectangle(
                vars=self.vars,
                lb=(table_cx - table_w / 2, table_cy - table_l / 2),
                ub=(table_cx + table_w / 2, table_cy + table_l / 2),
            )
        ]

        return domains.Union(sets=humans + plants + tables)


register(
    name="OfficeSingleIntegrator",
    entrypoint=lambda: StaticOffice(system=SingleIntegrator()),
)
