import numpy as np
import torch

from fosco.common import domains
from fosco.common.domains import Set
from fosco.systems import ControlAffineDynamics
from fosco.systems.system import register


@register
class SingleIntegrator(ControlAffineDynamics):
    """
    Single integrator system. X=[x, y], U=[vx, vy]
    dX/dt = [vx, vy]
    """

    @property
    def id(self) -> str:
        return self.__class__.__name__

    @property
    def vars(self) -> list[str]:
        return ["x0", "x1"]

    @property
    def controls(self) -> list[str]:
        return ["u0", "u1"]

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
        return domains.Rectangle(
            vars=self.vars, lb=(-5.0,) * self.n_vars, ub=(-4.0,) * self.n_vars
        )

    @property
    def unsafe_domain(self) -> Set:
        return domains.Sphere(
            vars=self.vars,
            center=(0.0,) * self.n_vars,
            radius=1.0,
            dim_select=[0, 1],
            include_boundary=False,
        )

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
            gx = gx.to(x.device)
        return gx

    def gx_smt(self, x: list) -> np.ndarray | torch.Tensor:
        assert isinstance(
            x, list
        ), "expected list of symbolic state variables, [x0, x1, ...]"
        return np.eye(len(x))

    def fz_torch(
        self, x: np.ndarray | torch.Tensor, z: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        assert (
            len(x.shape) == 3
        ), "expected batched input with shape (batch_size, state_dim, 1)"
        assert (
            len(z.shape) == 3
        ), "expected batched input with shape (batch_size, uncertain_dim, 1)"
        assert (
            x.shape[0] == z.shape[0]
        ), "expected batched input with shape (batch_size, uncertain_dim, 1)"
        assert isinstance(
            x, type(z)
        ), f"expected same type for x and z, got {type(x)} and {type(z)}"

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
