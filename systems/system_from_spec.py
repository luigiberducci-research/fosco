from typing import Callable

import numpy as np
import torch

from fosco.common.consts import DomainName as dn
from fosco.common.domains import Set
from systems import ControlAffineDynamics


class System(ControlAffineDynamics):
    def __init__(
        self,
        id: str,
        variables: list[str],
        controls: list[str],
        doms: dict[str, Set],
        dynamics: dict[str, Callable]
    ):
        self._id = id
        self._variables = variables
        self._controls = controls
        self._domains = doms
        self._dynamics = dynamics

        assert all([dname.value in self._domains for dname in [dn.XD, dn.UD, dn.XI, dn.XU]])
        assert all([fname in self._dynamics for fname in ["fx_torch", "fx_smt", "gx_torch", "gx_smt"]])

    @property
    def id(self) -> str:
        return self._id

    @property
    def vars(self) -> list[str]:
        return self._variables

    @property
    def controls(self) -> list[str]:
        return self._controls

    @property
    def state_domain(self) -> Set:
        return self._domains[dn.XD]

    @property
    def input_domain(self) -> Set:
        return self._domains[dn.UD]

    @property
    def init_domain(self) -> Set:
        return self._domains[dn.XI]

    @property
    def unsafe_domain(self) -> Set:
        return self._domains[dn.XU]

    def fx_torch(self, x) -> np.ndarray | torch.Tensor:
        return self._dynamics["fx_torch"](x)

    def fx_smt(self, x) -> np.ndarray | torch.Tensor:
        return self._dynamics["fx_smt"](x)

    def gx_torch(self, x) -> np.ndarray | torch.Tensor:
        return self._dynamics["gx_torch"](x)

    def gx_smt(self, x) -> np.ndarray | torch.Tensor:
        return self._dynamics["gx_smt"](x)
