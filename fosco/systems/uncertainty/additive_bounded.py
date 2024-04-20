import numpy as np
import torch

from fosco.common import domains
from fosco.common.domains import Set
from fosco.systems import ControlAffineDynamics
from fosco.systems.uncertainty.uncertainty_wrapper import register, UncertaintyWrapper


@register
class AdditiveBounded(UncertaintyWrapper):
    """
    Wrapper to add additive bounded uncertainty to a system.
        - X, U as in the base system
        - Z=[z_0, ..., z_n] where z_i is the additive uncertainty for x_i

    f(x) = f_base(x) + I Z + 0 Z u = f_base(x) + z
    """

    def __init__(self, system: ControlAffineDynamics, radius: float = 1.0):
        super().__init__(system=system)
        self._radius = radius  # uncertainty radius

    @property
    def uncertainty_id(self) -> str:
        return self.__class__.__name__

    @property
    def uncertain_vars(self) -> list[str]:
        return [f"z{i}" for i in range(self.n_vars)]

    @property
    def uncertainty_domain(self) -> Set:
        return domains.Sphere(
            vars=self.uncertain_vars,
            center=(0.0,) * self.n_uncertain,
            radius=self._radius,
        )

    def fz_torch(
        self, x: np.ndarray | torch.Tensor, z: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        self._assert_batched_input(x, z)
        return z

    # if z is a list, how come the returned is np or torch?
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
        assert x.shape[0] == z.shape[0], f"expected same batch size for x and z, got {x.shape[0]} and {z.shape[0]}"
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
