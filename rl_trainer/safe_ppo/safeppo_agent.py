from typing import Callable

import gymnasium
import numpy as np
import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from rl_trainer.ppo.ppo_agent import ActorCriticAgent
from fosco.models import TorchSymDiffModel
from fosco.models.utils import layer_init
from fosco.systems import ControlAffineDynamics
from fosco.systems.system_env import SystemEnv


class SafeActorCriticAgent(ActorCriticAgent):
    def __init__(
        self, envs: SystemEnv, barrier: TorchSymDiffModel | Callable,
    ):
        super().__init__(envs=envs)
        self.classk_size = 1

        # override actor model
        self.actor_backbone = nn.Sequential(
            layer_init(nn.Linear(self.input_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(64, self.output_size), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, self.output_size))

        self.actor_k_mean = nn.Sequential(
            layer_init(nn.Linear(64, self.classk_size), std=0.01),
        )
        self.actor_k_logstd = nn.Parameter(torch.zeros(1, self.classk_size))

        # safety layer
        if not isinstance(envs.action_space, gymnasium.spaces.Box):
            raise TypeError(
                f"This agent only supports continuous actions as Box type, got {envs.action_space}"
            )
        self.umin = envs.action_space.low
        self.umax = envs.action_space.high
        self.barrier = barrier
        self.safety_layer = self._make_barrier_layer()
        self.fx = envs.system.fx_torch
        self.gx = envs.system.gx_torch

    def _make_barrier_layer(self) -> CvxpyLayer:
        """
        Constructs a CvxpyLayer to implement the CBF problem:
            argmin_u u.T Q u + px.T u
            s.t. dh(x,u) + alpha(h(x)) >= 0
        """
        Q = torch.eye(self.output_size)
        px = cp.Parameter(self.output_size)

        Lfhx = cp.Parameter(1)
        Lghx = cp.Parameter(self.output_size)
        alphahx = cp.Parameter(1)

        u = cp.Variable(self.output_size)
        slack = cp.Variable(1)

        constraints = []
        # input constraint: u in U
        constraints += [u >= self.umin, u <= self.umax]

        # constraint: hdot(x,u) + alpha(h(x)) >= 0
        constraints += [Lfhx + Lghx @ u + alphahx + slack >= 0.0]

        # objective: u.T Q u + p.T u
        objective = 1 / 2 * cp.quad_form(u, Q) + px.T @ u + 1000 * slack ** 2
        problem = cp.Problem(cp.Minimize(objective), constraints)

        return CvxpyLayer(
            problem, variables=[u, slack], parameters=[px, Lfhx, Lghx, alphahx]
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(
        self, x, action=None, action_k=None, use_safety_layer: bool = True
    ):
        # assert input is ok
        assert isinstance(x, torch.Tensor), f"Expected torch.Tensor, got {type(x)}"
        assert len(x.shape) == 2, f"Expected (batch, dim) got {x.shape}"

        # denormalize
        x0 = x

        # normalize
        # todo
        action_z = self.actor_backbone(x)

        # action mean here is not the diag in the Q matrix
        action_mean = self.actor_mean(action_z)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        action_k_mean = self.actor_k_mean(action_z)
        action_k_logstd = self.actor_k_logstd.expand_as(action_k_mean)
        action_k_std = torch.exp(action_k_logstd)
        probs_k = Normal(action_k_mean, action_k_std)

        if action is None:
            action = probs.sample()

        if action_k is None:
            action_k = probs_k.sample()

        # safety layer
        safe_action = action
        if use_safety_layer:
            n_batch = x.size(0)
            hx = self.barrier(x0).view(n_batch, 1)
            dhdx = self.barrier.gradient(x0).view(n_batch, 1, self.output_size)

            fx = self.fx(x0.view(-1, self.input_size, 1))
            gx = self.gx(x0.view(-1, self.input_size, 1))

            Lfhx = (dhdx @ fx).view(n_batch, 1)
            Lghx = (dhdx @ gx).view(n_batch, self.output_size)
            alpha = 4 * nn.Sigmoid()(action_k)
            alphahx = (alpha * hx).view(n_batch, 1)

            # note: no kwargs to cvxpylayer
            (safe_action, slack) = self.safety_layer(action, Lfhx, Lghx, alphahx)

        log_probs = probs.log_prob(action).sum(1)
        entropy = probs.entropy().sum(1)
        class_k_log_probs = probs_k.log_prob(action_k).sum(1)
        class_k_entropy = probs_k.entropy().sum(1)
        value = self.critic(x)

        results = {
            "safe_action": safe_action,
            "action": action,  # this is px
            "logprob": log_probs,  # this is logprob(px)
            "entropy": entropy,  # entropy of px
            "classk": action_k,
            "classk_logprob": class_k_log_probs,
            "classk_entropy": class_k_entropy,
            "value": value,
        }

        batch_sz = x.shape[0]
        for k, batch in results.items():
            assert batch.shape[0] == batch_sz, f"wrong {k} shape: {batch.shape}"

        return results


class BarrierPolicy(nn.Module):
    def __init__(
        self,
        system: ControlAffineDynamics,
        barrier: TorchSymDiffModel | Callable,
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
        self._assert_input(x)
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
        (u,) = self._safety_layer(px, Lfhx, Lghx, alphahx)

        self.hx = hx
        self.px = px

        self._assert_output(x, u)

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

    def _assert_input(self, x):
        assert isinstance(x, torch.Tensor), f"Expected torch.Tensor, got {type(x)}"
        assert len(x.shape) == 2, f"Expected (batch, dim) got {x.shape}"

    def _assert_output(self, x, u):
        assert isinstance(x, torch.Tensor), f"Expected torch.Tensor, got {type(x)}"
        assert isinstance(u, torch.Tensor), f"Expected torch.Tensor, got {type(u)}"
        assert len(x.shape) == len(
            u.shape
        ), f"Expected same dimensions for x and u, got {x.shape} and {u.shape}"
        assert (
            x.shape[0] == u.shape[0]
        ), f"Expected return same batch as input, got {x.shape[0]} != {u.shape[0]}"
