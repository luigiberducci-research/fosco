from abc import ABC, abstractmethod

import numpy as np
import torch

from fosco.common.domains import Set
from fosco.systems.system import (
    UncertainControlAffineDynamics,
    ControlAffineDynamics,
)


UNCERTAINTY_REGISTRY = {}


def register(cls):
    """
    Decorator to register a system class in the systems registry.
    """
    UNCERTAINTY_REGISTRY[cls.__name__] = cls
    return cls


class UncertaintyWrapper(UncertainControlAffineDynamics, ABC):
    def __init__(self, system: ControlAffineDynamics):
        super().__init__()
        self._base_system = system

    @property
    @abstractmethod
    def uncertainty_id(self) -> str:
        raise NotImplementedError()

    @property
    def id(self) -> str:
        return f"{self._base_system.id}_{self.uncertainty_id}"

    @property
    def vars(self) -> list[str]:
        return self._base_system.vars

    @property
    def controls(self) -> list[str]:
        return self._base_system.controls

    @property
    def n_vars(self) -> int:
        return self._base_system.n_vars

    @property
    def n_controls(self) -> int:
        return self._base_system.n_controls

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
        return self._base_system.fx_torch(x)

    def fx_smt(self, x) -> np.ndarray | torch.Tensor:
        return self._base_system.fx_smt(x)

    def gx_torch(self, x) -> np.ndarray | torch.Tensor:
        return self._base_system.gx_torch(x)

    def gx_smt(self, x) -> np.ndarray | torch.Tensor:
        return self._base_system.gx_smt(x)
