import numpy as np
import torch

from fosco.common import domains
from fosco.common.consts import TimeDomain
from fosco.common.domains import Set

from fosco.systems import ControlAffineDynamics
from fosco.systems.core.system import register


@register
class DoubleIntegrator(ControlAffineDynamics):
    """
    Single integrator system. X=[x, y], U=[vx, vy]
    dX/dt = [vx, vy]
    """

    @property
    def id(self) -> str:
        return self.__class__.__name__

    @property
    def vars(self) -> tuple[str, ...]:
        return "x0", "x1", "x2", "x3"

    @property
    def controls(self) -> tuple[str, ...]:
        return "u0", "u1"

    @property
    def time_domain(self) -> TimeDomain:
        return TimeDomain.CONTINUOUS

    @property
    def state_domain(self) -> Set:
        return domains.Rectangle(
            vars=self.vars, lb=(-5.0,) * self.n_vars, ub=(5.0,) * self.n_vars
        )

    @property
    def input_domain(self) -> Set:
        return domains.Rectangle(
            vars=self.controls,
            lb=(-5.0,) * self.n_controls,
            ub=(5.0,) * self.n_controls,
        )

    @property
    def init_domain(self) -> Set:
        # return domains.Rectangle(
        #    vars=self.vars, lb=(-5.0,) * self.n_vars, ub=(-4.0, -4.0, 5.0, 5.0)
        # )

        init_unsafe = domains.Rectangle(
            vars=self.vars, lb=(-3.0, -3.0, -5.0, -5.0), ub=(3.0, 3.0, 5.0, 5.0),
        )
        return domains.Complement(set=init_unsafe, outer_set=self.state_domain)

    @property
    def unsafe_domain(self) -> Set:
        return domains.Rectangle(
            vars=self.vars, lb=(-1.0, -1.0, -5.0, -5.0), ub=(1.0, 1.0, 5.0, 5.0)
        )

    def fx_torch(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        assert (
            len(x.shape) == 3
        ), "expected batched input with shape (batch_size, state_dim, 1)"
        if isinstance(x, np.ndarray):
            vx, vy = x[:, 2, :], x[:, 3, :]
            fx = np.concatenate([vx, vy, np.zeros_like(vx), np.zeros_like(vy)], axis=1)
            fx = fx[:, :, None]
        else:
            vx, vy = x[:, 2, :], x[:, 3, :]
            fx = torch.cat([vx, vy, torch.zeros_like(vx), torch.zeros_like(vy)], dim=1)
            fx = fx[:, :, None]
        return fx

    def fx_smt(self, x: list) -> np.ndarray | torch.Tensor:
        assert isinstance(
            x, list
        ), "expected list of symbolic state variables, [x0, x1, ...]"
        vx, vy = x[2], x[3]
        return np.array([vx, vy, 0, 0])

    def gx_torch(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        assert (
            len(x.shape) == 3
        ), "expected batched input with shape (batch_size, state_dim, 1)"
        if isinstance(x, np.ndarray):
            gx = np.zeros((self.n_vars, self.n_controls))
            gx[2:, :] = np.eye(self.n_controls)
            gx = gx[None].repeat(x.shape[0], axis=0)
        else:
            gx = torch.zeros((self.n_vars, self.n_controls))
            gx[2:, :] = torch.eye(self.n_controls)
            gx = gx[None].repeat((x.shape[0], 1, 1))
        return gx

    def gx_smt(self, x: list) -> np.ndarray | torch.Tensor:
        assert isinstance(
            x, list
        ), "expected list of symbolic state variables, [x0, x1, ...]"
        gx = np.zeros((self.n_vars, self.n_controls))
        gx[2:, :] = np.eye(self.n_controls)
        return gx
