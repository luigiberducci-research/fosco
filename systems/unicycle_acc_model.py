import dreal
import numpy as np
import torch

from systems import ControlAffineDynamics
from systems.system import register


@register
class UnicycleAcc(ControlAffineDynamics):
    """
    Unicycle system. X=[x, y, theta, v], U=[a, w]

    dx/dt = v * cos(theta)
    dy/dt = v * sin(theta)
    dv/dt = a
    dtheta/dt = w
    """
    @property
    def id(self) -> str:
        return self.__class__.__name__

    @property
    def n_vars(self) -> int:
        # x, y, v, theta
        return 4

    @property
    def n_controls(self) -> int:
        # a, w
        return 2

    def fx_torch(self, x) -> np.ndarray | torch.Tensor:
        assert len(x.shape) == 3, f"expected batched input with shape (batch_size, state_dim, 1), got shape {x.shape}"
        vx = x[:, 2, :]
        theta = x[:, 3, :]
        if isinstance(x, np.ndarray):
            fx = np.array([
                vx * np.cos(theta),
                vx * np.sin(theta),
                np.zeros_like(theta),
                np.zeros_like(vx)
            ])
            fx = np.moveaxis(fx, 0, 1)
        else:
            fx = torch.stack([
                vx * torch.cos(theta),
                vx * torch.sin(theta),
                torch.zeros_like(theta),
                torch.zeros_like(vx)
            ], dim=1)
        assert fx.shape == x.shape, f"expected output shape {x.shape}, got {fx.shape}"
        return fx

    def fx_smt(self, x) -> np.ndarray | torch.Tensor:
        assert isinstance(
            x, list
        ), "expected list of symbolic state variables, [x0, x1, ...]"
        raise NotImplementedError()
        return np.zeros(len(x))

    def gx_torch(self, x) -> np.ndarray | torch.Tensor:
        assert (
                len(x.shape) == 3
        ), "expected batched input with shape (batch_size, state_dim, 1)"
        if isinstance(x, np.ndarray):
            gx = np.array([
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0]
            ])
            gx = np.tile(gx, (x.shape[0], 1, 1))
        else:
            gx = torch.tensor([
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0]
            ]).repeat(x.shape[0], 1, 1)
        assert gx.shape == (x.shape[0], self.n_vars, self.n_controls), f"expected output shape {(x.shape[0], 4, 2)}, got {gx.shape}"
        return gx

    def gx_smt(self, x) -> np.ndarray | torch.Tensor:
        assert isinstance(
            x, list
        ), "expected list of symbolic state variables, [x0, x1, ...]"
        raise NotImplementedError()