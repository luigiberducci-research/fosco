import time
from argparse import Namespace
from typing import Optional

import gymnasium
import numpy as np
import torch
from torch import optim, nn

from models.ppo_agent import ActorCriticAgent
from rl_algorithms.rl_trainer import RLTrainer


class PPOTrainer(RLTrainer):
    def __init__(
            self,
            envs: gymnasium.Env,
            args: Namespace,
            device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.args = args

        obs_space = envs.single_observation_space if hasattr(envs,
                                                             "single_observation_space") else envs.observation_space
        act_space = envs.single_action_space if hasattr(envs, "single_action_space") else envs.action_space
        input_size = np.array(obs_space.shape).prod()
        output_size = np.array(act_space.shape).prod()
        self.agent = ActorCriticAgent(input_size=input_size, output_size=output_size).to(device)

        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.args.learning_rate, eps=1e-5)
        self.iteration = 0

    def train(
            self,
            obs: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            dones: torch.Tensor,
            next_obs: torch.Tensor,
            next_done: Optional[torch.Tensor] = None,
            values: Optional[torch.Tensor] = None,
            logprobs: Optional[torch.Tensor] = None,
            global_step: Optional[int] = None
    ) -> dict[str, float]:
        self.iteration += 1

        # Annealing the rate if instructed to do so.
        if self.args.anneal_lr:
            frac = 1.0 - (self.iteration - 1.0) / self.args.num_iterations
            lrnow = frac * self.args.learning_rate
            self.optimizer.param_groups[0]["lr"] = lrnow

        # advantage estimation
        advantages, returns = self._advantage_estimation(obs, actions, rewards, dones, next_obs, next_done, values)

        # flatten the batch
        b_obs = obs.reshape((-1,) + (self.agent.input_size,))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + (self.agent.output_size,))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # update policy and value
        # Optimizing the policy and value network
        b_inds = np.arange(self.args.batch_size)
        clipfracs = []
        for epoch in range(self.args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.args.batch_size, self.args.minibatch_size):
                end = start + self.args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if self.args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.args.clip_coef,
                        self.args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.args.ent_coef * entropy_loss + v_loss * self.args.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                self.optimizer.step()

            if self.args.target_kl is not None and approx_kl > self.args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # return logging infos
        return {
            "charts/learning_rate": self.optimizer.param_groups[0]["lr"],
            "losses/value_loss": v_loss.item(),
            "losses/policy_loss": pg_loss.item(),
            "losses/entropy": entropy_loss.item(),
            "losses/old_approx_kl": old_approx_kl.item(),
            "losses/approx_kl": approx_kl.item(),
            "losses/clipfrac": np.mean(clipfracs),
            "losses/explained_variance": explained_var,
        }

    def _advantage_estimation(self, obs, actions, rewards, dones, next_obs, next_done, values):
        # bootstrap value if not done
        with torch.no_grad():
            next_value = self.agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(self.args.num_steps)):
                if t == self.args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + self.args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[
                    t] = lastgaelam = delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        return advantages, returns

    def get_actor(self) -> ActorCriticAgent:
        return self.agent


