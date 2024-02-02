import numpy as np
import torch

from systems.uncertainty.uncertainty_wrapper import register
from systems.uncertainty.uncertainty_wrapper import UncertaintyWrapper


@register
class AdditiveBounded(UncertaintyWrapper):
    """
    Wrapper to add additive bounded uncertainty to a system.
        - X, U as in the base system
        - Z=[z_0, ..., z_n] where z_i is the additive uncertainty for x_i

    f(x) = f_base(x) + I Z + 0 Z u = f_base(x) + z
    """

    @property
    def uncertainty_id(self) -> str:
        return self.__class__.__name__

    @property
    def n_uncertain(self) -> int:
        return self.n_vars

    def fz_torch(
        self, x: np.ndarray | torch.Tensor, z: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        self._assert_batched_input(x, z)
        return z

    def fz_smt(self, x: list, z: list) -> np.ndarray | torch.Tensor:
        self._assert_symbolic_input(x, z)
        return z

    def gz_torch(
        self, x: np.ndarray | torch.Tensor, z: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        self._assert_batched_input(x, z)
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
        ), "expected batched z with shape (batch_size, state_dim, 1)"
        assert x.shape[0] == z.shape[0], "expected same batch size for x and z"
        assert isinstance(
            x, type(z)
        ), f"expected same type for x and z, got {type(x)} and {type(z)}"

    def _assert_symbolic_input(self, x: list, z: list) -> None:
        assert isinstance(
            x, list
        ), "expected list of symbolic state variables, [x0, x1, ...]"
        assert isinstance(
            z, list
        ), "expected list of symbolic state variables, [z0, z1, ...]"
