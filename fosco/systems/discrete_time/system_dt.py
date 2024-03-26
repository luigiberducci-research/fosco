import numpy as np
import torch

from fosco.common.consts import TimeDomain
from fosco.common.domains import Set
from fosco.systems import ControlAffineDynamics


class EulerDTSystem(ControlAffineDynamics):
    """
    Wrap a control affine system with a discretized time step.
    The f, g functions are updated to keep the same control-affine structure

    x(t+1) = x(t) + dt * (f(x(t)) + g(x(t)) u(t))
           = x(t) + dt * f(x(t)) + dt * g(x(t)) u(t)
    The new fx, gx become:
        f'(x) = x(t) + dt * f(x(t))
        g'(x) = dt * g(x(t))
    """

    def __init__(self, system: ControlAffineDynamics, dt: float):
        if system.time_domain != TimeDomain.CONTINUOUS:
            raise ValueError("EulerDTSystem only supports continuous time systems")

        self.system = system
        self.dt = dt

    @property
    def id(self) -> str:
        return f"{self.system.id}-dt{self.dt}"

    @property
    def vars(self) -> tuple[str, ...]:
        return self.system.vars

    @property
    def controls(self) -> tuple[str, ...]:
        return self.system.controls

    @property
    def time_domain(self) -> TimeDomain:
        return TimeDomain.DISCRETE

    @property
    def state_domain(self) -> Set:
        return self.system.state_domain

    @property
    def input_domain(self) -> Set:
        return self.system.input_domain

    @property
    def init_domain(self) -> Set:
        return self.system.init_domain

    @property
    def unsafe_domain(self) -> Set:
        return self.system.unsafe_domain

    def fx_torch(self, x) -> np.ndarray | torch.Tensor:
        return x + self.dt * self.system.fx_torch(x)

    def fx_smt(self, x) -> np.ndarray | torch.Tensor:
        return x + self.dt * self.system.fx_smt(x)

    def gx_torch(self, x) -> np.ndarray | torch.Tensor:
        return self.dt * self.system.gx_torch(x)

    def gx_smt(self, x) -> np.ndarray | torch.Tensor:
        return self.dt * self.system.gx_smt(x)
