from abc import abstractmethod, ABC
from typing import Callable

import numpy as np
import torch

from fosco.common.consts import DomainName
from fosco.common.domains import Set
from fosco.common.utils import contains_object
from fosco.verifier.verifier import SYMBOL

SYSTEM_REGISTRY = {}


def register(entrypoint: Callable, name: str = None):
    """
    Decorator to register a system entrypoint in the systems registry.
    """
    name = name or entrypoint.__name__
    SYSTEM_REGISTRY[name] = entrypoint
    return entrypoint


class ControlAffineDynamics(ABC):
    """
    Implements a controllable dynamical model with control-affine dynamics dx = f(x) + g(x) u
    """

    @property
    @abstractmethod
    def id(self) -> str:
        raise NotImplementedError()

    @property
    @abstractmethod
    def vars(self) -> list[str]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def controls(self) -> list[str]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def state_domain(self) -> Set:
        raise NotImplementedError()

    @property
    @abstractmethod
    def input_domain(self) -> Set:
        raise NotImplementedError()

    @property
    @abstractmethod
    def init_domain(self) -> Set:
        raise NotImplementedError()

    @property
    @abstractmethod
    def unsafe_domain(self) -> Set:
        raise NotImplementedError()

    @property
    def domains(self) -> dict[str, Set]:
        return {
            DomainName.XD.value: self.state_domain,
            DomainName.UD.value: self.input_domain,
            DomainName.XI.value: self.init_domain,
            DomainName.XU.value: self.unsafe_domain,
        }

    @property
    def n_vars(self) -> int:
        return len(self.vars)

    @property
    def n_controls(self) -> int:
        return len(self.controls)

    @abstractmethod
    def fx_torch(self, x) -> np.ndarray | torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def fx_smt(self, x) -> np.ndarray | torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def gx_torch(self, x) -> np.ndarray | torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def gx_smt(self, x) -> np.ndarray | torch.Tensor:
        raise NotImplementedError()

    def f(
        self, v: np.ndarray | torch.Tensor, u: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        if torch.is_tensor(v) or isinstance(v, np.ndarray):
            return self._f_torch(v, u)
        elif contains_object(v, SYMBOL):
            dvs = self.fx_smt(v) + self.gx_smt(v) @ u
            return dvs
        else:
            raise NotImplementedError(f"Unsupported type {type(v)}")

    def _f_torch(self, v: torch.Tensor, u: torch.Tensor) -> list:
        v = v.reshape(-1, self.n_vars, 1)
        u = u.reshape(-1, self.n_controls, 1)
        vdot = self.fx_torch(v) + self.gx_torch(v) @ u
        return vdot.reshape(-1, self.n_vars)

    def __call__(
        self, v: np.ndarray | torch.Tensor, u: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        return self.f(v, u)


class UncertainControlAffineDynamics(ControlAffineDynamics):
    """
    Extends a control-affine dynamical model with additive/multiplicative uncertainty dependent on variables z.
        dx = f(x) + g(x) u + fz(x, z) + gz(x, z) u
    """

    @property
    @abstractmethod
    def uncertain_vars(self) -> list[str]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def uncertainty_domain(self) -> Set:
        raise NotImplementedError()

    @property
    def n_uncertain(self) -> int:
        return len(self.uncertain_vars)

    @property
    def domains(self) -> dict[str, Set]:
        domains = super().domains
        domains[DomainName.ZD.value] = self.uncertainty_domain
        return domains

    @abstractmethod
    def fz_torch(self, x, z) -> np.ndarray | torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def fz_smt(self, x, z) -> np.ndarray | torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def gz_torch(self, x, z) -> np.ndarray | torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def gz_smt(self, x, z) -> np.ndarray | torch.Tensor:
        raise NotImplementedError()

    # todo: overrides f but with a different signature, how to handle this?
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
                dvs = self._base_system.fx_smt(v) + self._base_system.gx_smt(v) @ u
            else:
                dvs = (
                    self._base_system.fx_smt(v)
                    + self._base_system.gx_smt(v) @ u
                    + self.fz_smt(v, z)
                    + self.gz_smt(v, z) @ u
                )
            return dvs
        else:
            raise NotImplementedError(f"Unsupported type {type(v)}")

    def _f_torch(
        self,
        v: torch.Tensor,
        u: torch.Tensor,
        z: torch.Tensor,
        only_nominal: bool = False,
    ) -> list:
        v = v.reshape(-1, self.n_vars, 1)
        u = u.reshape(-1, self.n_controls, 1)
        z = z.reshape(-1, self.n_uncertain, 1)

        fx_torch = self._base_system.fx_torch
        gx_torch = self._base_system.gx_torch
        if only_nominal:
            vdot = fx_torch(v) + gx_torch(v) @ u
        else:
            vdot = (
                fx_torch(v)
                + gx_torch(v) @ u
                + self.fz_torch(v, z)
                + self.gz_torch(v, z) @ u
            )

        return vdot.reshape(-1, self.n_vars)

    def __call__(
        self,
        v: np.ndarray | torch.Tensor,
        u: np.ndarray | torch.Tensor,
        z: np.ndarray | torch.Tensor,
        only_nominal: bool = False,
    ) -> np.ndarray | torch.Tensor:
        return self.f(v, u, z, only_nominal=only_nominal)
