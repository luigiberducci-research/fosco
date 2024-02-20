from typing import Iterable

import numpy as np
import torch
import z3

from fosco.verifier import SYMBOL, FUNCTIONS, VerifierZ3
from models.torchsym import TorchSymDiffModel, TorchSymModel
from systems import ControlAffineDynamics, UncertainControlAffineDynamics


class SingleIntegratorCBF(TorchSymDiffModel):

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

    def forward_smt(self, x: list[SYMBOL]) -> tuple[SYMBOL, Iterable[SYMBOL]]:
        self._assert_forward_smt_input(x=x)
        hx = x[0] ** 2 + x[1] ** 2 - self._safety_dist ** 2
        self._assert_forward_smt_output(x=hx)
        return hx, []

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
    def gradient_smt(self, x: Iterable[SYMBOL]) ->  tuple[Iterable[SYMBOL], Iterable[SYMBOL]]:
        self._assert_forward_smt_input(x=x)
        hx = np.array([[
            2 * x[0],
            2 * x[1]
        ]])
        self._assert_gradient_smt_output(x=hx)
        return hx, []

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
        assert len(x.shape) == 2, f"expected shape (1, {state_dim}), got {x.shape}"
        assert x.shape[1] == state_dim, f"expected shape (1, {state_dim}), got {x.shape}"
        assert all([isinstance(xi, SYMBOL) for xi in x[0]]), f"expected symbolic grad w.r.t. input, got {x}"

    def save(self, outdir: str):
        pass

    @staticmethod
    def load(logdir: str):
        pass


class SingleIntegratorCompensatorAdditiveBoundedUncertainty(TorchSymModel):

    def __init__(self, h: TorchSymDiffModel, system: UncertainControlAffineDynamics):
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

    def forward_smt(self, x: list[SYMBOL]) -> tuple[SYMBOL, Iterable[SYMBOL]]:
        """
        Problem: z3 is not able to cope with sqrt.

        Solution: forward pass as conjunction of two constraints
        - note that sigma = sqrt(dhdx[0] ** 2 + dhdx[1] ** 2) is equivalent to
          sigma ** 2 = dhdx[0] ** 2 + dhdx[1] ** 2
        - then we forward pass as
        And(sigma * z_bound, sigma ** 2 = dhdx[0] ** 2 + dhdx[1] ** 2)
        """
        self._assert_forward_smt_input(x=x)
        _And = FUNCTIONS["And"]

        dhdx, dhdx_constraints = self._h.gradient_smt(x=x)
        norm = VerifierZ3.new_vars(n=1, base="norm")[0]
        norm_constraint = [
            norm * norm == dhdx[0, 0] ** 2 + dhdx[0, 1] ** 2,
            norm >= 0.0
        ]
        sigma = norm * self._z_bound

        self._assert_forward_smt_output(x=sigma)
        return sigma, dhdx_constraints + norm_constraint

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


class SingleIntegratorTunableCompensatorAdditiveBoundedUncertainty(TorchSymModel):

    def __init__(self, h: TorchSymDiffModel, system: UncertainControlAffineDynamics):
        super().__init__()
        self._system = system
        self._h = h # CBF (we use its gradient)
        self._safety_dist = 1.0 # todo this should be taken from system
        self._z_bound = 1.0 # todo this should be taken from system uncertainty


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        sigma(x) = || h.gradient(x) || * z_bound * k(hx)
        """
        self._assert_forward_input(x=x)
        dhdx = self._h.gradient(x=x)
        sigma = torch.norm(dhdx, dim=-1) * self._z_bound * self._k_function(self._h(x))
        self._assert_forward_output(x=sigma)
        return sigma

    def forward_smt(self, x: list[SYMBOL]) -> tuple[SYMBOL, Iterable[SYMBOL]]:
        """
        Problem: z3 is not able to cope with sqrt.

        Solution: forward pass as conjunction of two constraints
        - note that sigma = sqrt(dhdx[0] ** 2 + dhdx[1] ** 2) is equivalent to
          sigma ** 2 = dhdx[0] ** 2 + dhdx[1] ** 2
        - then we forward pass as
        And(sigma * z_bound, sigma ** 2 = dhdx[0] ** 2 + dhdx[1] ** 2)
        """
        self._assert_forward_smt_input(x=x)
        _And = FUNCTIONS["And"]

        dhdx, dhdx_constraints = self._h.gradient_smt(x=x)
        hx = self._h(x=x) # to check if this is correct or should we use something like self._h_smt(x=x)
        norm = VerifierZ3.new_vars(n=1, base="norm")[0]
        norm_constraint = [
            norm * norm == dhdx[0, 0] ** 2 + dhdx[0, 1] ** 2,
            norm >= 0.0
        ]
        sigma = norm * self._z_bound* self._k_function(hx)

        self._assert_forward_smt_output(x=sigma)
        return sigma, dhdx_constraints + norm_constraint

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

    def _k_function(self, hx: torch.Tensor) -> torch.Tensor:
        '''
        It is a continuous, non-increasing function κ : R≥0 → R with κ(0) = 1
        '''
        return 2 / (1 + torch.exp(hx)) # z3 does not supports exp, try polynomial; also we can see what is the range of hx,

        # k_function = 1 / (hx ^ m + 1)

    def save(self, outdir: str):
        pass

    @staticmethod
    def load(logdir: str):
        pass

# system, verify (fail), compensator, verify (work)
# create an example in /examples
    
class SingleIntegratorConvexHullUncertainty(TorchSymModel):

    def __init__(self, h: TorchSymDiffModel, system: UncertainControlAffineDynamics):
        super().__init__()  
        self._system = system
        self._h = h # CBF (we use its gradient)
        self._z_bound = 1.0 # todo this should be taken from system uncertainty

    def uncertain_phi(self, x: torch.Tensor) -> torch.Tensor:
        state_dim = self._system.n_vars
        # phi should be function from R^n to R^n where n is the system state dim
        # fix the phi, avoid many randomizations
        phi_1 = torch.rand(state_dim) * x
        phi_2 = torch.rand(state_dim) * x
        phi_3 = torch.rand(state_dim) * x
        all_phi = torch.cat((phi_1, phi_2, phi_3), 0)
        print(all_phi.shape)

        return all_phi


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        sigma(x) = || h.gradient(x) || * z_bound
        """
        self._assert_forward_input(x=x)
        dhdx = self._h.gradient(x=x)
        self.all_phi = self.uncertain_phi(x=x)
        sigma = torch.min(dhdx * self.all_phi)
        self._assert_forward_output(x=sigma)
        return sigma

    def forward_smt(self, x: list[SYMBOL]) -> tuple[SYMBOL, Iterable[SYMBOL]]:
        """
        Problem: z3 is not able to cope with sqrt.

        Solution: forward pass as conjunction of two constraints
        - note that sigma = sqrt(dhdx[0] ** 2 + dhdx[1] ** 2) is equivalent to
          sigma ** 2 = dhdx[0] ** 2 + dhdx[1] ** 2
        - then we forward pass as
        And(sigma * z_bound, sigma ** 2 = dhdx[0] ** 2 + dhdx[1] ** 2)
        """
        self._assert_forward_smt_input(x=x)
        _And = FUNCTIONS["And"]

        dhdx, dhdx_constraints = self._h.gradient_smt(x=x)
        norm = VerifierZ3.new_vars(n=1, base="norm")[0]
        norm_constraint = [
            norm * norm == dhdx[0, 0] ** 2 + dhdx[0, 1] ** 2,
            norm >= 0.0
        ]
        sigma = norm * self.all_phi # take the min, how to do it in z3

        self._assert_forward_smt_output(x=sigma)
        return sigma, dhdx_constraints + norm_constraint

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


class SingleIntegratorPolytopeUncertainty(TorchSymModel):

    def __init__(self, h: TorchSymDiffModel, system: UncertainControlAffineDynamics):
        super().__init__()
        self._system = system
        self._h = h # CBF (we use its gradient)
        self._safety_dist = 1.0 # todo this should be taken from system
        self._phi_x_u = torch.tensor([])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        sigma(x) = || h.gradient(x) || * z_bound
        """
        self._assert_forward_input(x=x)
        dhdx = self._h.gradient(x=x)
        sigma = torch.norm(dhdx, dim=-1) * self._z_bound
        self._assert_forward_output(x=sigma)
        return sigma

    def forward_smt(self, x: list[SYMBOL]) -> tuple[SYMBOL, Iterable[SYMBOL]]:
        """
        Problem: z3 is not able to cope with sqrt.

        Solution: forward pass as conjunction of two constraints
        - note that sigma = sqrt(dhdx[0] ** 2 + dhdx[1] ** 2) is equivalent to
          sigma ** 2 = dhdx[0] ** 2 + dhdx[1] ** 2
        - then we forward pass as
        And(sigma * z_bound, sigma ** 2 = dhdx[0] ** 2 + dhdx[1] ** 2)
        """
        self._assert_forward_smt_input(x=x)
        _And = FUNCTIONS["And"]

        dhdx, dhdx_constraints = self._h.gradient_smt(x=x)
        norm = VerifierZ3.new_vars(n=1, base="norm")[0]
        norm_constraint = [
            norm * norm == dhdx[0, 0] ** 2 + dhdx[0, 1] ** 2,
            norm >= 0.0
        ]
        sigma = norm * self._z_bound

        self._assert_forward_smt_output(x=sigma)
        return sigma, dhdx_constraints + norm_constraint

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