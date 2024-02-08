from abc import abstractmethod, ABC

import numpy as np
import torch
import z3

from fosco.common.utils import contains_object

SYSTEM_REGISTRY = {}


def register(cls):
    """
    Decorator to register a system class in the systems registry.
    """
    SYSTEM_REGISTRY[cls.__name__] = cls
    return cls


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
    def n_vars(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def n_controls(self) -> int:
        raise NotImplementedError()

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
        elif contains_object(v, z3.ArithRef):
            dvs = self.fx_smt(v) + self.gx_smt(v) @ u
            return [z3.simplify(dv) for dv in dvs]
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
    def n_uncertain(self) -> int:
        raise NotImplementedError()

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
        elif contains_object(v, z3.ArithRef):
            if only_nominal:
                dvs = self._base_system.fx_smt(v) + self._base_system.gx_smt(v) @ u
            else:
                dvs = (
                    self._base_system.fx_smt(v)
                    + self._base_system.gx_smt(v) @ u
                    + self.fz_smt(v, z)
                    + self.gz_smt(v, z) @ u
                )
            return [z3.simplify(dv) for dv in dvs]
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


