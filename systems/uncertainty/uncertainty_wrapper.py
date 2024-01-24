from abc import ABC

import numpy as np
import torch

from systems.system import (
    UncertainControlAffineDynamics,
    ControlAffineDynamics,
)


class UncertaintyWrapper(UncertainControlAffineDynamics, ABC):

    def __init__(self, system: ControlAffineDynamics):
        super().__init__()
        self._base_system = system

    @property
    def id(self) -> str:
        return self._base_system.id

    @property
    def n_vars(self) -> int:
        return self._base_system.n_vars

    @property
    def n_controls(self) -> int:
        return self._base_system.n_controls

    def fx_torch(self, x) -> np.ndarray | torch.Tensor:
        return self._base_system.fx_torch(x)

    def fx_smt(self, x) -> np.ndarray | torch.Tensor:
        return self._base_system.fx_smt(x)

    def gx_torch(self, x) -> np.ndarray | torch.Tensor:
        return self._base_system.gx_torch(x)

    def gx_smt(self, x) -> np.ndarray | torch.Tensor:
        return self._base_system.gx_smt(x)
