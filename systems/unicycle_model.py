import dreal
import numpy as np
import torch

from systems import ControlAffineDynamics
from systems.system import register


@register
class Unicycle(ControlAffineDynamics):
    """
    Unicycle system. X=[x, y, theta], U=[v, w]

    dx/dt = v * cos(theta)
    dy/dt = v * sin(theta)
    dtheta/dt = w
    """
    @property
    def id(self) -> str:
        return self.__class__.__name__

    @property
    def n_vars(self) -> int:
        # x, y, theta
        return 3

    @property
    def n_controls(self) -> int:
        # v, w
        return 2

    def fx_torch(self, x) -> np.ndarray | torch.Tensor:
        assert len(x.shape) == 3, f"expected batched input with shape (batch_size, state_dim, 1), got shape {x.shape}"
        if isinstance(x, np.ndarray):
            fx = np.zeros_like(x)
        else:
            fx = torch.zeros_like(x)
        return fx

    def fx_smt(self, x) -> np.ndarray | torch.Tensor:
        assert isinstance(
            x, list
        ), "expected list of symbolic state variables, [x0, x1, ...]"
        return np.zeros(len(x))

    def gx_torch(self, x) -> np.ndarray | torch.Tensor:
        assert (
                len(x.shape) == 3
        ), "expected batched input with shape (batch_size, state_dim, 1)"
        if isinstance(x, np.ndarray):
            cosx = np.cos(x[:, 2, :])
            sinx = np.sin(x[:, 2, :])
            # make a batch of matrices, each with
            # [[cos(theta_i), 0][sin(theta_), 0][0, 1]]
            gx = np.array([[[cosx[i][0], 0], [sinx[i][0], 0], [0, 1]] for i in range(x.shape[0])])
        else:
            cosx = torch.cos(x[:, 2, :])
            sinx = torch.sin(x[:, 2, :])
            gx = torch.tensor([[[cosx[i][0], 0], [sinx[i][0], 0], [0, 1]] for i in range(x.shape[0])])



        return gx

    def gx_smt(self, x) -> np.ndarray | torch.Tensor:
        assert isinstance(
            x, list
        ), "expected list of symbolic state variables, [x0, x1, ...]"
        Sin_ = dreal.sin
        Cos_ = dreal.cos

        return np.array([
            [Cos_(x[2]), 0.0],
            [Sin_(x[2]), 0.0],
            [0.0, 1.0]
        ])