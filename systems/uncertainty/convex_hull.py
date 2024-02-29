import numpy as np
import torch
from systems.system import ControlAffineDynamics

from systems.uncertainty.uncertainty_wrapper import register
from systems.uncertainty.uncertainty_wrapper import UncertaintyWrapper

from models.torchsym import TorchSymFn


@register
class ConvexHull(UncertaintyWrapper):
    """
    Wrapper to add convex hull type uncertainty to a system.
        - X, U as in the base system
        - Z=[Z_1; Z_2]=[z_1, ..., z_n; z_{n+1}, ..., z_2n] 
        - where Z_1 = z_{1:n} is the additive convex hull uncertainty for x_i 
        - and Z_2 = z_{n+1:2n} is the additive convex hull uncertainty for control input for x_i 

    f(x) = f_base(x) + co(Z_1) + co(Z_2) u = f_base(x) + fz + gz u
    """

    def __init__(self, system: ControlAffineDynamics, f_uncertainty: list[TorchSymFn], f_uncertainty_params: list) -> None:
        super().__init__(system)

        self.f_uncertainty = f_uncertainty # list of function, each function input n_vars dim, output n_vars
        self.f_uncertainty_params = f_uncertainty_params # linear combination parameters for uncertain functions
        assert len(self.f_uncertainty_params) == len(self.f_uncertainty), "the number of uncertain functions and parameters are not the same"
        # self.g_uncertainty = g_uncertainty # list of function, ignore this first

    @property
    def uncertainty_id(self) -> str:
        return self.__class__.__name__

    # return 2n variables
    @property
    def n_uncertain(self) -> int:
        return len(self.f_uncertainty) # + len(self.g_uncertainty)

    # z is the 2n variables, we should return z[0:n] @ f_uncertainty; f_uncertainty idealy should be a function
    def fz_torch(
        self, x: np.ndarray | torch.Tensor, z: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        self._assert_batched_input(x, z)
        # self._assert_sum_to_one(z[:, 0:len(self.f_uncertainty), 0])
        # another assertion on the dim of z
        
        f_uncertain_x = torch.zeros_like(x)
        for index in range(len(self.f_uncertainty)):
            f_uncertain_x = f_uncertain_x + self.f_uncertainty[index].forward(x) * z[:, index:index+1, :]
        if isinstance(x, np.ndarray):
            f_uncertain_x = np.array(f_uncertain_x)
        else:
            f_uncertain_x = torch.tensor(f_uncertain_x)
        return f_uncertain_x

    # return symbolic variable, double check!
    def fz_smt(self, x: list, z: list) -> np.ndarray | torch.Tensor:
        self._assert_symbolic_input(x, z)
        f_uncertain_x = np.zeros_like(x, dtype=float)
        for index in range(len(self.f_uncertainty)):
            f_uncertain_x = f_uncertain_x + self.f_uncertainty[index].forward_smt(x) * z[index]
        f_uncertain_x = np.array(f_uncertain_x)
        return f_uncertain_x

    # z is the 2n variables, we should return z[n:2n] @ g_uncertainty, double check the dim of g_uncertainty
    def gz_torch(
        self, x: np.ndarray | torch.Tensor, z: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        self._assert_batched_input(x, z)
        # self._assert_sum_to_one(z[:, self.n_vars:2*self.n_vars, 0])

        if isinstance(x, np.ndarray):
            gx = np.zeros((self.n_vars, self.n_controls))[None].repeat(
                x.shape[0], axis=0
            )
        else:
            gx = torch.zeros((self.n_vars, self.n_controls))[None].repeat(
                (x.shape[0], 1, 1)
            )
        return gx

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
    def _assert_sum_to_one(self, z: np.ndarray | torch.Tensor) -> None:
        if isinstance(z, np.ndarray):
            assert np.sum(z, axis=1) == np.ones(z.shape[0]), "expected z is summed to 1"
        else:
            assert torch.sum(z, axis=1) == torch.ones(z.shape[0]), "expected z is summed to 1"

    def _assert_symbolic_input(self, x: list, z: list) -> None:
        assert isinstance(
            x, list
        ), "expected list of symbolic state variables, [x0, x1, ...]"
        assert isinstance(
            z, list
        ), "expected list of symbolic state variables, [z0, z1, ...]"
