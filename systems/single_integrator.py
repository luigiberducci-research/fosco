import numpy as np
import torch

from fosco.models.network import TorchMLP
from systems import ControlAffineControllableDynamicalModel
from systems.system import UncertainControlAffineControllableDynamicalModel


class SingleIntegrator(ControlAffineControllableDynamicalModel):
    """
    Single integrator system. X=[x, y], U=[vx, vy]
    dX/dt = [vx, vy]
    """

    @property
    def n_vars(self) -> int:
        return 2

    @property
    def n_controls(self) -> int:
        return 2

    def fx_torch(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        assert (
            len(x.shape) == 3
        ), f"expected batched input with shape (batch_size, state_dim, 1), got shape {x.shape}"
        if isinstance(x, np.ndarray):
            fx = np.zeros_like(x)
        else:
            fx = torch.zeros_like(x)
        return fx

    def fx_smt(self, x: list) -> np.ndarray | torch.Tensor:
        assert isinstance(
            x, list
        ), "expected list of symbolic state variables, [x0, x1, ...]"
        return np.zeros(len(x))

    def gx_torch(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        assert (
            len(x.shape) == 3
        ), "expected batched input with shape (batch_size, state_dim, 1)"
        if isinstance(x, np.ndarray):
            gx = np.eye(x.shape[1])[None].repeat(x.shape[0], axis=0)
        else:
            gx = torch.eye(x.shape[1])[None].repeat((x.shape[0], 1, 1))
        return gx

    def gx_smt(self, x: list) -> np.ndarray | torch.Tensor:
        assert isinstance(
            x, list
        ), "expected list of symbolic state variables, [x0, x1, ...]"
        return np.eye(len(x))

    def fz_torch(self, x: np.ndarray | torch.Tensor, z: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        assert (
            len(x.shape) == 3
        ), "expected batched input with shape (batch_size, state_dim, 1)"
        assert (
            len(z.shape) == 3
        ), "expected batched input with shape (batch_size, uncertain_dim, 1)"
        assert (
            x.shape[0] == z.shape[0]
        ), "expected batched input with shape (batch_size, uncertain_dim, 1)"
        assert type(x) == type(z), f"expected same type for x and z, got {type(x)} and {type(z)}"

        fz = z  # simple additive uncertainty

        return fz

    def fz_smt(self, x: list, z: list) -> np.ndarray | torch.Tensor:
        assert isinstance(
            x, list
        ), "expected list of symbolic state variables, [x0, x1, ...]"
        assert isinstance(
            z, list
        ), "expected list of symbolic state variables, [z0, z1, ...]"
        return z

class SingleIntegratorAddBoundedUncertainty(UncertainControlAffineControllableDynamicalModel, SingleIntegrator):
    """
    Single integrator system with additive uncertainty.
    X=[x, y], U=[vx, vy], Z=[z_x, z_y]

    dX/dt = [vx, vy] + [z_x, z_y]
    """

    @property
    def n_uncertain(self) -> int:
        return 2

    def fz_torch(self, x: np.ndarray | torch.Tensor, z: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        assert (
            len(x.shape) == 3
        ), "expected batched input with shape (batch_size, state_dim, 1)"
        assert (
            len(z.shape) == 3
        ), "expected batched input with shape (batch_size, uncertain_dim, 1)"
        assert (
            x.shape[0] == z.shape[0]
        ), "expected batched input with shape (batch_size, uncertain_dim, 1)"
        assert type(x) == type(z), f"expected same type for x and z, got {type(x)} and {type(z)}"

        fz = z  # simple additive uncertainty

        return fz

    def fz_smt(self, x: list, z: list) -> np.ndarray | torch.Tensor:
        assert isinstance(
            x, list
        ), "expected list of symbolic state variables, [x0, x1, ...]"
        assert isinstance(
            z, list
        ), "expected list of symbolic state variables, [z0, z1, ...]"
        return z

    def gz_torch(self, x: np.ndarray | torch.Tensor, z: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if isinstance(x, np.ndarray):
            gx = np.zeros((self.n_vars, self.n_controls))[None].repeat(x.shape[0], axis=0)
        else:
            gx = torch.zeros((self.n_vars, self.n_controls))[None].repeat((x.shape[0], 1, 1))
        return gx

    def gz_smt(self, x: list, z: list) -> np.ndarray | torch.Tensor:
        assert isinstance(
            x, list
        ), "expected list of symbolic state variables, [x0, x1, ...]"
        assert isinstance(
            z, list
        ), "expected list of symbolic state variables, [z0, z1, ...]"
        return np.zeros((self.n_vars, self.n_controls))
