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

        self.device = device or torch.device("cpu")
        self.mean = torch.from_numpy(mean).to(self.device)
        self.std = torch.from_numpy(std).to(self.device)

        # system specific
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


import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from models.utils import layer_init


class CBFActorCriticAgent(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.critic = nn.Sequential(
            layer_init(nn.Linear(input_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_backbone = nn.Sequential(
            layer_init(nn.Linear(input_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(64, output_size), std=0.01),
        )
        self.class_k_mean = nn.Sequential(
            layer_init(nn.Linear(64, 1), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, output_size))
        self.class_k_logstd = nn.Parameter(torch.zeros(1, 1))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(self.actor_backbone(x))
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        action_probs = Normal(action_mean, action_std)

        class_k_mean = self.class_k_mean(self.actor_backbone(x))
        class_k_logstd = self.class_k_logstd.expand_as(class_k_mean)
        class_k_std = torch.exp(class_k_logstd)
        class_k_probs = Normal(class_k_mean, class_k_std)

        if action is None:
            action = (action_probs.sample(), class_k_probs.sample())

        logprobs = (action_probs.log_prob(action[0]).sum(1), class_k_probs.log_prob(action[1]).sum(1))
        entropy = (action_probs.entropy().sum(1), class_k_probs.entropy().sum(1))

        return action, logprobs, entropy, self.critic(x)

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