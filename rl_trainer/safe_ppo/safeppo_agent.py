from typing import Callable, Optional

import gymnasium
import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from torch import nn
from torch.distributions import Normal

from rl_trainer.ppo.ppo_agent import ActorCriticAgent
from fosco.models import TorchSymDiffModel
from fosco.models.utils import layer_init
from fosco.systems import ControlAffineDynamics, EulerDTSystem
from fosco.systems.gym_env.system_env import SystemEnv


class SafeActorCriticAgent(ActorCriticAgent):
    def __init__(
        self,
        envs: SystemEnv,
        barrier: TorchSymDiffModel | Callable,
        compensator: TorchSymDiffModel | Callable = None,
        device: Optional[torch.device | str] = None,
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
        self.xsigma = compensator
        self.safety_layer = self._make_barrier_layer()

        # extract continuous dynamics
        assert isinstance(envs.system, EulerDTSystem) and not isinstance(
            envs.unwrapped.system.system, EulerDTSystem
        )
        self.fx = envs.unwrapped.system.system.fx_torch
        self.gx = envs.unwrapped.system.system.gx_torch

        # device
        self.device = device or torch.device("cpu")
        self.to(self.device)

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
            dhdx = self.barrier.gradient(x0).view(n_batch, 1, self.input_size)

            fx = self.fx(x0.view(-1, self.input_size, 1))
            gx = self.gx(x0.view(-1, self.input_size, 1))

            Lfhx = (dhdx @ fx).view(n_batch, 1)
            Lghx = (dhdx @ gx).view(n_batch, self.output_size)
            alpha = 4 * nn.Sigmoid()(action_k)
            alphahx = (alpha * hx).view(n_batch, 1)

            # add compensation term if available
            if self.xsigma is not None:
                sigmax = self.xsigma(x0).view(n_batch, 1)
                Lfhx = Lfhx - sigmax

            # note: no kwargs to cvxpylayer
            (safe_action, slack) = self.safety_layer(action, Lfhx, Lghx, alphahx)

        log_probs = probs.log_prob(action).sum(1)
        entropy = probs.entropy().sum(1)
        class_k_log_probs = probs_k.log_prob(action_k).sum(1)
        class_k_entropy = probs_k.entropy().sum(1)
        value = self.critic(x)[..., 0]

        results = {
            "action": safe_action,
            "unsafe_action": action,  # this is px
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
