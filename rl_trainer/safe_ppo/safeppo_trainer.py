from functools import partial
from typing import Optional

import gymnasium
import numpy as np
import torch
from torch import nn

from rl_trainer.ppo.ppo_config import PPOConfig
from rl_trainer.safe_ppo.safeppo_agent import SafeActorCriticAgent
from rl_trainer.common.buffer import CyclicBuffer
from rl_trainer.ppo.ppo_trainer import PPOTrainer


class SafePPOTrainer(PPOTrainer):
    def __init__(
        self,
        envs: gymnasium.Env,
        config: PPOConfig,
        barrier: nn.Module,
        compensator: nn.Module = None,
        device: Optional[torch.device] = None,
    ) -> None:
        agent_cls = partial(SafeActorCriticAgent, barrier=barrier, compensator=compensator)
        super().__init__(
            envs=envs,
            config=config,
            agent_cls=agent_cls,
            device=device,
        )

        buffer_shapes = {
            "obs": (self.cfg.num_steps, self.cfg.num_envs)
            + envs.single_observation_space.shape,
            "action": (self.cfg.num_steps, self.cfg.num_envs)
            + envs.single_action_space.shape,
            "unsafe_action": (self.cfg.num_steps, self.cfg.num_envs)
            + envs.single_action_space.shape,
            "classk": (self.cfg.num_steps, self.cfg.num_envs) + (1,),
            "logprob": (self.cfg.num_steps, self.cfg.num_envs),
            "classk_logprob": (self.cfg.num_steps, self.cfg.num_envs),
            "reward": (self.cfg.num_steps, self.cfg.num_envs),
            "cost": (self.cfg.num_steps, self.cfg.num_envs),
            "done": (self.cfg.num_steps, self.cfg.num_envs),
            "value": (self.cfg.num_steps, self.cfg.num_envs),
        }
        self.buffer = CyclicBuffer(
            capacity=self.cfg.num_steps,
            feature_shapes=buffer_shapes,
            device=self.device,
        )

    def _update(
        self,
        next_obs: torch.Tensor,
        next_done: Optional[torch.Tensor] = None,
    ) -> dict[str, float]:
        data = self.buffer.sample()
        obs = data["obs"]
        logprobs = data["logprob"]
        classk_logprobs = data["classk_logprob"]
        actions = data["unsafe_action"]
        classks = data["classk"]
        rewards = data["reward"]
        dones = data["done"]
        values = data["value"]

        self.iteration += 1

        # Annealing the rate if instructed to do so.
        if self.cfg.anneal_lr:
            frac = 1.0 - (self.iteration - 1.0) / self.cfg.num_iterations
            lrnow = frac * self.cfg.learning_rate
            self.optimizer.param_groups[0]["lr"] = lrnow

        # advantage estimation
        advantages, returns = self._advantage_estimation(
            obs, actions, rewards, dones, next_obs, next_done, values
        )

        # flatten the batch
        b_obs = obs.reshape((-1,) + (self.agent.input_size,))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + (self.agent.output_size,))
        b_classks = classks.reshape((-1,) + (self.agent.classk_size,))
        b_classk_logprobs = classk_logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # update policy and value
        # Optimizing the policy and value network
        b_inds = np.arange(self.cfg.batch_size)
        clipfracs = []
        for epoch in range(self.cfg.update_epochs):
            self._logger.debug(f"epoch {epoch}")
            np.random.shuffle(b_inds)
            for start in range(0, self.cfg.batch_size, self.cfg.minibatch_size):
                end = start + self.cfg.minibatch_size
                mb_inds = b_inds[start:end]

                # note: logprob, entropy, newvalue do not depende on safe_action -> disable it
                results = self.agent.get_action_and_value(
                    x=b_obs[mb_inds],
                    action=b_actions[mb_inds],
                    action_k=b_classks[mb_inds],
                    use_safety_layer=False,
                )
                newlogprob = results["logprob"]
                newclassklogprob = results["classk_logprob"]
                entropy = results["entropy"]
                classkentropy = results["classk_entropy"]
                newvalue = results["value"]

                logratio = (
                    newlogprob
                    + newclassklogprob
                    - b_logprobs[mb_inds]
                    - b_classk_logprobs[mb_inds]
                )
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > self.cfg.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if self.cfg.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - self.cfg.clip_coef, 1 + self.cfg.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.cfg.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.cfg.clip_coef,
                        self.cfg.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = 0.5 * (entropy.mean() + classkentropy.mean())
                loss = (
                    pg_loss
                    - self.cfg.ent_coef * entropy_loss
                    + v_loss * self.cfg.vf_coef
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.agent.parameters(), self.cfg.max_grad_norm
                )
                self.optimizer.step()

            if self.cfg.target_kl is not None and approx_kl > self.cfg.target_kl:
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
