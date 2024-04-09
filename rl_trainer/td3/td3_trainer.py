from argparse import Namespace
from typing import Optional

import gymnasium
import torch
from torch import optim
import torch.nn.functional as F

from rl_trainer.td3.td3_agent import DDPGActor, QNetwork
from rl_trainer.trainer import RLTrainer


class TD3Trainer(RLTrainer):
    def __init__(
        self,
        envs: gymnasium.Env,
        args: Namespace,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cpu")
        self.args = args

        self.obs_space = (
            envs.single_observation_space
            if hasattr(envs, "single_observation_space")
            else envs.observation_space
        )
        self.act_space = (
            envs.single_action_space
            if hasattr(envs, "single_action_space")
            else envs.action_space
        )

        self.actor = DDPGActor(
            observation_space=self.obs_space, action_space=self.act_space
        ).to(device)
        self.qf1 = QNetwork(
            observation_space=self.obs_space, action_space=self.act_space
        ).to(device)
        self.qf2 = QNetwork(
            observation_space=self.obs_space, action_space=self.act_space
        ).to(device)
        self.qf1_target = QNetwork(
            observation_space=self.obs_space, action_space=self.act_space
        ).to(device)
        self.qf2_target = QNetwork(
            observation_space=self.obs_space, action_space=self.act_space
        ).to(device)
        self.target_actor = DDPGActor(
            observation_space=self.obs_space, action_space=self.act_space
        ).to(device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        self.q_optimizer = optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()),
            lr=args.learning_rate,
        )
        self.actor_optimizer = optim.Adam(
            list(self.actor.parameters()), lr=args.learning_rate
        )

        self.iteration = 0

    def train(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_obs: torch.Tensor,
        next_done: torch.Tensor = None,
        values: torch.Tensor = None,
        logprobs: torch.Tensor = None,
        global_step: Optional[int] = None,
    ) -> dict[str, float]:
        global_step = obs.shape[0]
        self.iteration += 1

        # train
        args = self.args
        with torch.no_grad():
            clipped_noise = (
                torch.randn_like(actions, device=self.device) * args.policy_noise
            ).clamp(-args.noise_clip, args.noise_clip) * self.target_actor.action_scale

            next_state_actions = (self.target_actor(next_obs) + clipped_noise).clamp(
                self.act_space.low[0], self.act_space.high[0]
            )
            qf1_next_target = self.qf1_target(next_obs, next_state_actions)
            qf2_next_target = self.qf2_target(next_obs, next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            next_q_value = rewards.flatten() + (1 - dones.flatten()) * args.gamma * (
                min_qf_next_target
            ).view(-1)

        qf1_a_values = self.qf1(obs, actions).view(-1)
        qf2_a_values = self.qf2(obs, actions).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        # optimize the model
        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        if global_step % args.policy_frequency == 0:
            actor_loss = -self.qf1(obs, self.actor(obs)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # update the target network
            for param, target_param in zip(
                self.actor.parameters(), self.target_actor.parameters()
            ):
                target_param.data.copy_(
                    args.tau * param.data + (1 - args.tau) * target_param.data
                )
            for param, target_param in zip(
                self.qf1.parameters(), self.qf1_target.parameters()
            ):
                target_param.data.copy_(
                    args.tau * param.data + (1 - args.tau) * target_param.data
                )
            for param, target_param in zip(
                self.qf2.parameters(), self.qf2_target.parameters()
            ):
                target_param.data.copy_(
                    args.tau * param.data + (1 - args.tau) * target_param.data
                )

        return {
            "losses/qf1_values": qf1_a_values.mean().item(),
            "losses/qf2_values": qf2_a_values.mean().item(),
            "losses/qf1_loss": qf1_loss.item(),
            "losses/qf2_loss": qf2_loss.item(),
            "losses/qf_loss": qf_loss.item() / 2.0,
            "losses/actor_loss": actor_loss.item(),
        }

    def get_actor(self) -> torch.nn.Module:
        return self.actor
