from abc import abstractmethod

import numpy as np
import torch
import z3
from matplotlib import pyplot as plt

from fosco.common.utils import contains_object


class ControlAffineControllableDynamicalModel:
    """
    Implements a controllable dynamical model with control-affine dynamics dx = f(x) + g(x) u
    """

    @property
    @abstractmethod
    def n_vars(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def n_controls(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def fx_torch(self, x) -> np.ndarray | torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def fx_smt(self, x) -> np.ndarray | torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def gx_torch(self, x) -> np.ndarray | torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def gx_smt(self, x) -> np.ndarray | torch.Tensor:
        raise NotImplementedError()

    def f(
            self, v: np.ndarray | torch.Tensor, u: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        if torch.is_tensor(v) or isinstance(v, np.ndarray):
            return self._f_torch(v, u)
        elif contains_object(v, z3.ArithRef):
            dvs = self.fx_smt(v) + self.gx_smt(v) @ u
            return [z3.simplify(dv) for dv in dvs]
        else:
            raise NotImplementedError(f"Unsupported type {type(v)}")

    def _f_torch(self, v: torch.Tensor, u: torch.Tensor) -> list:
        v = v.reshape(-1, self.n_vars, 1)
        u = u.reshape(-1, self.n_controls, 1)
        vdot = self.fx_torch(v) + self.gx_torch(v) @ u
        return vdot.reshape(-1, self.n_vars)

    def __call__(
            self, v: np.ndarray | torch.Tensor, u: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        return self.f(v, u)

    def plot(
            self,
            xrange: tuple,
            yrange: tuple,
            ctrl: callable,
            ax: plt.Axes = None,

    ):
        ax = ax or plt.gca()

        xx = np.linspace(xrange[0], xrange[1], 50)
        yy = np.linspace(yrange[0], yrange[1], 50)

        XX, YY = np.meshgrid(xx, yy)
        obs = torch.stack(
                    [torch.tensor(XX).ravel(), torch.tensor(YY).ravel()]
                ).T.float()
        uu = ctrl(obs)

        dx, dy = (
            self.f(v=obs, u=uu)
            .detach()
            .numpy()
            .T
        )
        # color = np.sqrt((np.hypot(dx, dy)))
        dx = dx.reshape(XX.shape)
        dy = dy.reshape(YY.shape)
        # color = color.reshape(XX.shape)
        ax.set_ylim(xrange)
        ax.set_xlim(yrange)
        plt.streamplot(
            XX,
            YY,
            dx,
            dy,
            linewidth=0.8,
            density=1.5,
            arrowstyle="fancy",
            arrowsize=1.5,
            color="tab:gray",
        )
        return ax


class UncertainControlAffineControllableDynamicalModel(ControlAffineControllableDynamicalModel):
    """
    Implements a controllable dynamical model with control-affine dynamics dx = f(x) + g(x) u
    """

    @property
    @abstractmethod
    def n_uncertain(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def fz_torch(self, x, z) -> np.ndarray | torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def fz_smt(self, x, z) -> np.ndarray | torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def gz_torch(self, x, z) -> np.ndarray | torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def gz_smt(self, x, z) -> np.ndarray | torch.Tensor:
        raise NotImplementedError()

    def f(
            self, v: np.ndarray | torch.Tensor, u: np.ndarray | torch.Tensor, z: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        if torch.is_tensor(v) or isinstance(v, np.ndarray):
            return self._f_torch(v, u, z)
        elif contains_object(v, z3.ArithRef):
            dvs = self.fx_smt(v) + self.gx_smt(v) @ u + self.fz_smt(v, z) + self.gz_smt(v, z) @ u
            return [z3.simplify(dv) for dv in dvs]
        else:
            raise NotImplementedError(f"Unsupported type {type(v)}")

    def _f_torch(self, v: torch.Tensor, u: torch.Tensor, z: torch.Tensor) -> list:
        v = v.reshape(-1, self.n_vars, 1)
        u = u.reshape(-1, self.n_controls, 1)
        z = z.reshape(-1, self.n_uncertain, 1)
        vdot = self.fx_torch(v) + self.gx_torch(v) @ u + self.fz_torch(v, z) + self.gz_torch(v, z) @ u
        return vdot.reshape(-1, self.n_vars)

    def __call__(
            self, v: np.ndarray | torch.Tensor, u: np.ndarray | torch.Tensor, z: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        return self.f(v, u, z)

    def plot(
            self,
            xrange: tuple,
            yrange: tuple,
            ctrl: callable,
            zmodel: callable = None,
            ax: plt.Axes = None,
    ):
        ax = ax or plt.gca()

        xx = np.linspace(xrange[0], xrange[1], 50)
        yy = np.linspace(yrange[0], yrange[1], 50)

        XX, YY = np.meshgrid(xx, yy)
        obs = torch.stack(
                    [torch.tensor(XX).ravel(), torch.tensor(YY).ravel()]
                ).T.float()
        uu = ctrl(obs)

        if zmodel is not None:
            zz = zmodel(obs)
        else:
            zz = torch.zeros(obs.shape[0], self.n_uncertain)

        dx, dy = (
            self.f(v=obs, u=uu, z=zz)
            .detach()
            .numpy()
            .T
        )
        # color = np.sqrt((np.hypot(dx, dy)))
        dx = dx.reshape(XX.shape)
        dy = dy.reshape(YY.shape)
        # color = color.reshape(XX.shape)
        ax.set_ylim(xrange)
        ax.set_xlim(yrange)
        plt.streamplot(
            XX,
            YY,
            dx,
            dy,
            linewidth=0.8,
            density=1.5,
            arrowstyle="fancy",
            arrowsize=1.5,
            color="tab:gray",
        )
        return ax
