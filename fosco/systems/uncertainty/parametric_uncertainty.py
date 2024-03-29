import numpy as np
import torch

from fosco.common import domains
from fosco.common.domains import Set
from fosco.models import TorchSymFn
from fosco.systems import ControlAffineDynamics
from fosco.systems.uncertainty.uncertainty_wrapper import register, UncertaintyWrapper


@register
class ParametricUncertainty(UncertaintyWrapper):
    """
    Wrapper to add parametric uncertainty to a system.
        - X, U as in the base system
        - Z=[Z_1; Z_2]=[z_1, ..., z_n; z_{n+1}, ..., z_2n] 
        - where Z_1 = z_{1:n} is the uncertainty parameters for x_i 
        - and Z_2 = z_{n+1:2n} is the uncertainty parameters for control input for x_i 

    f(x) = f_base(x) + delta_f(x) + delta_g(x) @ u = f_base(x) + F(x)@z_f + G(x)@diag(u)@z_g

    (paper link: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9992667)

    TODO: to implement gz part!
    """

    def __init__(self, system: ControlAffineDynamics, f_uncertainty: np.ndarray | torch.Tensor, g_uncertainty: np.ndarray | torch.Tensor, \
                                uncertain_bound_A: np.ndarray | torch.Tensor, uncertain_bound_b: np.ndarray | torch.Tensor) -> None:
        super().__init__(system)

        self.f_uncertainty = f_uncertainty # unkown_f_matrix
        self.g_uncertainty = g_uncertainty # unkown_g_matrix
        self.uncertain_bound_A = uncertain_bound_A
        self.uncertain_bound_b = uncertain_bound_b

    @property
    def uncertainty_id(self) -> str:
        return self.__class__.__name__
    
    @property
    def uncertain_vars(self) -> list[str]:
        return [f"z{i}" for i in range(self.n_vars)]

    @property
    def uncertainty_domain(self) -> Set:
        return domains.Polytope(
            vars=self.uncertain_vars,
            lhs_A=self.uncertain_bound_A,
            rhs_b=self.uncertain_bound_b,
        )

    # return 2n variables
    @property
    def n_uncertain(self) -> int:
        return self.uncertain_bound_A.shape[1]

    # z is the 2n variables, we should return z[0:n] @ f_uncertainty
    def fz_torch(
        self, x: np.ndarray | torch.Tensor, z: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        self._assert_batched_input(x, z)
        self._assert_satisfy_matrix_constraint(z, self.uncertain_bound_A, self.uncertain_bound_b)
        f_uncertain_x = self.f_uncertainty @ z[0:self.n_uncertain]

        return f_uncertain_x

    # symbolic version of fz_torch
    def fz_smt(self, x: list, z: list) -> np.ndarray | torch.Tensor:
        self._assert_symbolic_input(x, z)
        f_uncertain_x = np.zeros_like(x, dtype=float)
        for index in range(len(self.f_uncertainty)):
            f_uncertain_x = f_uncertain_x + self.f_uncertainty[index].forward_smt(x) * z[index]
        # to change the forward_smt
        return f_uncertain_x

    # z is the 2n variables, we should return z[n:2n] @ g_uncertainty, double check the dim of g_uncertainty
    def gz_torch(
        self, x: np.ndarray | torch.Tensor, z: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        raise NotImplementedError

    def gz_smt(self, x: list, z: list) -> np.ndarray | torch.Tensor:
        self._assert_symbolic_input(x, z)
        return np.zeros((self.n_vars, self.n_controls))

    @staticmethod
    def _assert_batched_input(
            x: np.ndarray | torch.Tensor, z: np.ndarray | torch.Tensor
    ) -> None:
        assert (
            len(x.shape) == 3
        ), "expected batched x with shape (batch_size, state_dim, 1)"
        assert (
            len(z.shape) == 3
        ), "expected batched z with shape (batch_size, state_dim * 2, 1)"
        assert x.shape[0] == z.shape[0], "expected same batch size for x and z"
        # assert x.shape[1] * 2 == z.shape[1], "expected z's dim is twice the state_dim"
        assert isinstance(
            x, type(z)
        ), f"expected same type for x and z, got {type(x)} and {type(z)}"

    @staticmethod
    def _assert_satisfy_matrix_constraint(z: np.ndarray | torch.Tensor, A: np.ndarray | torch.Tensor, b: np.ndarray | torch.Tensor) -> None:
        if isinstance(z, np.ndarray):
            assert A @ z <= b, "expected A @ z <= b"
        else:
            assert A @ z <= b, "expected A @ z <= b"
    

    def _assert_symbolic_input(self, x: list, z: list) -> None:
        assert isinstance(
            x, list
        ), "expected list of symbolic state variables, [x0, x1, ...]"
        assert isinstance(
            z, list
        ), "expected list of symbolic state variables, [z0, z1, ...]"
