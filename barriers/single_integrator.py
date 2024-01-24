from typing import Iterable

import numpy as np
import torch

from fosco.verifier import SYMBOL, FUNCTIONS
from models.torchsym import TorchSymModel
from systems import ControlAffineDynamics, UncertainControlAffineDynamics


class SingleIntegratorCBF(TorchSymModel):

    def __init__(self, system: ControlAffineDynamics):
        super().__init__()
        self._system = system
        self._safety_dist = 1.0 # todo this should be taken from system


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        h(x) = | x - x_o |^2 - R^2
        """
        self._assert_forward_input(x=x)
        hx = x[:, 0]**2 + x[:, 1]**2 - self._safety_dist ** 2
        self._assert_forward_output(x=hx)
        return hx

    def forward_smt(self, x: list[SYMBOL]) -> SYMBOL:
        self._assert_forward_smt_input(x=x)
        hx = x[0] ** 2 + x[1] ** 2 - self._safety_dist ** 2
        self._assert_forward_smt_output(x=hx)
        return hx

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        dhdx0 = 2 * x[0]
        dhdx1 = 2 * x[1]
        """
        self._assert_forward_input(x=x)
        hx = torch.stack([
            2 * x[:, 0],
            2 * x[:, 1]
        ], dim=-1)
        self._assert_gradient_output(x=hx)
        return hx
    def gradient_smt(self, x: Iterable[SYMBOL]) -> Iterable[SYMBOL]:
        self._assert_forward_smt_input(x=x)
        hx = np.array([[
            2 * x[0],
            2 * x[1]
        ]])
        self._assert_gradient_smt_output(x=hx)
        return hx

    def _assert_forward_input(self, x: torch.Tensor) -> None:
        state_dim = self._system.n_vars
        assert isinstance(x, torch.Tensor), f"expected batch input to be tensor, got {type(x)}"
        assert len(x.shape) == 2, f"expected batch input (batch, {state_dim}), got {x.shape}"
        assert x.shape[1] == state_dim, f"expected batch input (batch, {state_dim}), got {x.shape}"

    def _assert_forward_output(self, x: torch.Tensor) -> None:
        assert isinstance(x, torch.Tensor), f"expected batch output to be tensor, got {type(x)}"
        assert len(x.shape) == 1, f"expected output of shape (batch,), got {x.shape}"

    def _assert_gradient_output(self, x: torch.Tensor) -> None:
        state_dim = self._system.n_vars
        assert isinstance(x, torch.Tensor), f"expected batch output to be tensor, got {type(x)}"
        assert len(x.shape) == 2, f"expected batch output (batch, {state_dim}), got {x.shape}"
        assert x.shape[1] == state_dim, f"expected batch output (batch, {state_dim}), got {x.shape}"

    def _assert_forward_smt_input(self, x: Iterable[SYMBOL]) -> None:
        state_dim = self._system.n_vars
        assert all([isinstance(xi, SYMBOL) for xi in x]), f"expected symbolic input, got {x}"
        assert len(x) == state_dim, f"expected {state_dim} state variables, got input of length {len(x)}"

    def _assert_forward_smt_output(self, x: Iterable[SYMBOL]) -> None:
        assert isinstance(x, SYMBOL), f"expected symbolic output, got {x}"

    def _assert_gradient_smt_output(self, x: Iterable[SYMBOL]) -> None:
        state_dim = self._system.n_vars
        assert (len(x.shape) == 2, f"expected shape (1, {state_dim}), got {x.shape}")
        assert x.shape[1] == state_dim, f"expected shape (1, {state_dim}), got {x.shape}"
        assert all([isinstance(xi, SYMBOL) for xi in x[0]]), f"expected symbolic grad w.r.t. input, got {x}"

    def save(self, outdir: str):
        pass

    @staticmethod
    def load(logdir: str):
        pass


class SingleIntegratorCompensatorAdditiveBoundedUncertainty(TorchSymModel):

    def __init__(self, h: TorchSymModel, system: UncertainControlAffineDynamics):
        super().__init__()
        self._system = system
        self._h = h # CBF (we use its gradient)
        self._safety_dist = 1.0 # todo this should be taken from system
        self._z_bound = 1.0 # todo this should be taken from system uncertainty


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        sigma(x) = || h.gradient(x) || * z_bound
        """
        self._assert_forward_input(x=x)
        dhdx = self._h.gradient(x=x)
        sigma = torch.norm(dhdx, dim=-1) * self._z_bound
        self._assert_forward_output(x=sigma)
        return sigma

    def forward_smt(self, x: list[SYMBOL]) -> SYMBOL:
        self._assert_forward_smt_input(x=x)
        _Sqrt = FUNCTIONS["Sqrt"]
        dhdx = self._h.gradient_smt(x=x)
        sigma = _Sqrt(dhdx[0] ** 2 + dhdx[1] ** 2) * self._z_bound
        self._assert_forward_smt_output(x=sigma)
        return sigma

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Never used. Do we need it?
        """
        raise NotImplementedError("the gradient of the compensator is not used in the optimization")
    def gradient_smt(self, x: Iterable[SYMBOL]) -> Iterable[SYMBOL]:
        """
        Never used. Do we need it?
        """
        raise NotImplementedError("the gradient of the compensator is not used in the optimization")

    def _assert_forward_input(self, x: torch.Tensor) -> None:
        state_dim = self._system.n_vars
        assert isinstance(x, torch.Tensor), f"expected batch input to be tensor, got {type(x)}"
        assert len(x.shape) == 2, f"expected batch input (batch, {state_dim}), got {x.shape}"
        assert x.shape[1] == state_dim, f"expected batch input (batch, {state_dim}), got {x.shape}"

    def _assert_forward_output(self, x: torch.Tensor) -> None:
        assert isinstance(x, torch.Tensor), f"expected batch output to be tensor, got {type(x)}"
        assert len(x.shape) == 1, f"expected output of shape (batch,), got {x.shape}"

    def _assert_forward_smt_input(self, x: Iterable[SYMBOL]) -> None:
        state_dim = self._system.n_vars
        assert all([isinstance(xi, SYMBOL) for xi in x]), f"expected symbolic input, got {x}"
        assert len(x) == state_dim, f"expected {state_dim} state variables, got input of length {len(x)}"

    def _assert_forward_smt_output(self, x: Iterable[SYMBOL]) -> None:
        assert isinstance(x, SYMBOL), f"expected symbolic output, got {x}"


    def save(self, outdir: str):
        pass

    @staticmethod
    def load(logdir: str):
        pass