from typing import Callable

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from cvxpylayers.torch import CvxpyLayer
import cvxpy as cp

from systems import ControlAffineDynamics


def assert_input(x):
    assert isinstance(x, torch.Tensor), f"Expected torch.Tensor, got {type(x)}"
    assert len(x.shape) == 2, f"Expected (batch, dim) got {x.shape}"


def assert_output(x, u):
    assert isinstance(x, torch.Tensor), f"Expected torch.Tensor, got {type(x)}"
    assert isinstance(u, torch.Tensor), f"Expected torch.Tensor, got {type(u)}"
    assert len(x.shape) == len(u.shape), f"Expected same dimensions for x and u, got {x.shape} and {u.shape}"
    assert x.shape[0] == u.shape[0], f"Expected return same batch as input, got {x.shape[0]} != {u.shape[0]}"


class BarrierPolicy(nn.Module):
    def __init__(
        self,
        system: ControlAffineDynamics,
        barrier: Callable,
        nHidden1=64,
        nHidden21=32,
        nHidden22=32,
        mean=None,
        std=None,
        device=None,
        bn=False,
    ):
        super().__init__()
        self.nFeatures = system.n_vars
        self.nHidden1 = nHidden1
        self.nHidden21 = nHidden21
        self.nHidden22 = nHidden22
        self.bn = bn
        self.nCls = system.n_controls

        mean = mean or np.zeros(self.nFeatures)
        std = std or np.ones(self.nFeatures)

        self.mean = torch.from_numpy(mean).to(device)
        self.std = torch.from_numpy(std).to(device)
        self.device = device

        # system specific todo: remove
        self.barrier = barrier
        self.fx = system.fx_torch
        self.gx = system.gx_torch

        # Normal BN/FC layers.
        if bn:
            self.bn1 = nn.BatchNorm1d(nHidden1)
            self.bn21 = nn.BatchNorm1d(nHidden21)
            self.bn22 = nn.BatchNorm1d(nHidden22)

        self.fc1 = nn.Linear(self.nFeatures, nHidden1)
        self.fc21 = nn.Linear(nHidden1, nHidden21)
        self.fc22 = nn.Linear(nHidden1, nHidden22)
        self.fc31 = nn.Linear(nHidden21, self.nCls)
        self.fc32 = nn.Linear(nHidden22, 1)

        self.umin = -torch.ones(self.nCls)
        self.umax = torch.ones(self.nCls)
        self._safety_layer = self._construct_cbf_problem()

    def forward(self, x):
        assert_input(x)
        nBatch = x.size(0)

        # Normal FC network.
        x = x.view(nBatch, -1)
        x0 = (x * self.std + self.mean).float()
        x = F.relu(self.fc1(x))
        if self.bn:
            x = self.bn1(x)

        x21 = F.relu(self.fc21(x))
        if self.bn:
            x21 = self.bn21(x21)
        x22 = F.relu(self.fc22(x))
        if self.bn:
            x22 = self.bn22(x22)

        px = self.fc31(x21)
        x32 = self.fc32(x22)
        alphax = 4 * nn.Sigmoid()(x32)  # ensure CBF parameters are positive

        # BarrierNet
        hx = self.barrier(x0).view(nBatch, 1)
        dhdx = self.barrier.gradient(x0)
        dhdx = dhdx.view(nBatch, 1, self.nCls)

        fx = self.fx(x0.view(-1, self.nFeatures, 1))
        gx = self.gx(x0.view(-1, self.nFeatures, 1))

        Lfhx = (dhdx @ fx).view(nBatch, 1)
        Lghx = (dhdx @ gx).view(nBatch, self.nCls)
        alphahx = (alphax * hx).view(nBatch, 1)
        (u,) = self._safety_layer(
            px,
            Lfhx,
            Lghx,
            alphahx
        )

        self.hx = hx
        self.px = px

        assert_output(x, u)

        return u

    def _construct_cbf_problem(self) -> CvxpyLayer:
        """
        Constructs a CvxpyLayer to implement the CBF problem:
            argmin_u u.T Q u + px.T u
            s.t. dh(x,u) + alpha(h(x)) >= 0
        """
        Q = torch.eye(self.nCls)
        px = cp.Parameter(self.nCls)

        Lfhx = cp.Parameter(1)
        Lghx = cp.Parameter(self.nCls)
        alphahx = cp.Parameter(1)

        u = cp.Variable(self.nCls)

        constraints = []
        # input constraint: u in U
        constraints += [u >= self.umin, u <= self.umax]

        # constraint: hdot(x,u) + alpha(h(x)) >= 0
        constraints += [Lfhx + Lghx @ u + alphahx >= 0.0]

        # objective: u.T Q u + p.T u
        objective = 1 / 2 * cp.quad_form(u, Q) + px.T @ u
        problem = cp.Problem(cp.Minimize(objective), constraints)

        return CvxpyLayer(problem, variables=[u], parameters=[px, Lfhx, Lghx, alphahx])
