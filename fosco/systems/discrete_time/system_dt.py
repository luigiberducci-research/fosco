import numpy as np
import torch

from fosco.common.consts import TimeDomain
from fosco.common.domains import Set
from fosco.common.utils import contains_object
from fosco.systems import ControlAffineDynamics, UncertainControlAffineDynamics
from fosco.verifier.types import SYMBOL
from fosco.verifier.utils import get_solver_simplify


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

        self._base_system = system
        self.dt = dt

    @property
    def id(self) -> str:
        return f"{self._base_system.id}-dt{self.dt}"

    @property
    def vars(self) -> tuple[str, ...]:
        return self._base_system.vars

    @property
    def controls(self) -> tuple[str, ...]:
        return self._base_system.controls

    @property
    def time_domain(self) -> TimeDomain:
        return TimeDomain.DISCRETE

    @property
    def state_domain(self) -> Set:
        return self._base_system.state_domain

    @property
    def input_domain(self) -> Set:
        return self._base_system.input_domain

    @property
    def init_domain(self) -> Set:
        return self._base_system.init_domain

    @property
    def unsafe_domain(self) -> Set:
        return self._base_system.unsafe_domain

    def fx_torch(self, x) -> np.ndarray | torch.Tensor:
        return x + self.dt * self._base_system.fx_torch(x)

    def fx_smt(self, x) -> np.ndarray | torch.Tensor:
        return x + self.dt * self._base_system.fx_smt(x)

    def gx_torch(self, x) -> np.ndarray | torch.Tensor:
        return self.dt * self._base_system.gx_torch(x)

    def gx_smt(self, x) -> np.ndarray | torch.Tensor:
        return self.dt * self._base_system.gx_smt(x)


class UncertainEulerDTSystem(EulerDTSystem, UncertainControlAffineDynamics):
    def __init__(self, system: UncertainControlAffineDynamics, dt: float):
        if not isinstance(system, UncertainControlAffineDynamics):
            raise ValueError("UncertainEulerDTSystem only supports UncertainControlAffineDynamics")

        super().__init__(system, dt)

    @property
    def uncertain_vars(self) -> tuple[str, ...]:
        return self._base_system.uncertain_vars

    @property
    def uncertainty_domain(self) -> Set:
        return self._base_system.uncertainty_domain

    def fz_torch(self, x, z) -> np.ndarray | torch.Tensor:
        return self.dt * self._base_system.fz_torch(x, z)

    def fz_smt(self, x, z) -> np.ndarray | torch.Tensor:
        return self.dt * self._base_system.fz_smt(x, z)

    def gz_torch(self, x, z) -> np.ndarray | torch.Tensor:
        return self.dt * self._base_system.gz_torch(x, z)

    def gz_smt(self, x, z) -> np.ndarray | torch.Tensor:
        return self.dt * self._base_system.gz_smt(x, z)

    def f(
        self,
        v: np.ndarray | torch.Tensor,
        u: np.ndarray | torch.Tensor,
        z: np.ndarray | torch.Tensor,
        only_nominal: bool = False,
    ) -> np.ndarray | torch.Tensor:
        if torch.is_tensor(v) or isinstance(v, np.ndarray):
            return self._f_torch(v=v, u=u, z=z, only_nominal=only_nominal)
        elif contains_object(v, SYMBOL):
            if only_nominal:
                dvs = self.fx_smt(v) + self.gx_smt(v) @ u
            else:
                dvs = (
                        self.fx_smt(v)
                        + self.gx_smt(v) @ u
                        + self.fz_smt(v, z)
                        + self.gz_smt(v, z) @ u
                )
            simplify_fn = get_solver_simplify(v)
            return np.array([simplify_fn(dv) for dv in dvs])
        else:
            raise NotImplementedError(f"Unsupported type {type(v)}")



