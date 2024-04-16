import logging
import time
from argparse import Namespace
from typing import Optional, Type

import gymnasium
import numpy as np
import torch
from torch import optim, nn

from fosco.logger import LOGGING_LEVELS
from rl_trainer.ppo.ppo_agent import ActorCriticAgent
from rl_trainer.common.buffer import CyclicBuffer
from rl_trainer.ppo.ppo_config import PPOConfig
from rl_trainer.trainer import RLTrainer


class PPOTrainer(RLTrainer):
    def __init__(
            self,
            envs: gymnasium.Env,
            config: PPOConfig,
            agent_cls: Optional[Type[ActorCriticAgent]] = None,
            device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device("cpu")

        self.cfg = config
        self.cfg.batch_size = int(self.cfg.num_envs * self.cfg.num_steps)
        self.cfg.minibatch_size = int(self.cfg.batch_size // self.cfg.num_minibatches)
        self.cfg.num_iterations = self.cfg.total_timesteps // self.cfg.batch_size

        if envs.unwrapped.is_vector_env:
            single_env = envs.envs[0]

        agent_cls = agent_cls or ActorCriticAgent
        self.agent = agent_cls(envs=single_env).to(device)

        buffer_shapes = {
            "obs": (self.cfg.num_steps, self.cfg.num_envs) + single_env.observation_space.shape,
            "action": (self.cfg.num_steps, self.cfg.num_envs) + single_env.action_space.shape,
            "logprob": (self.cfg.num_steps, self.cfg.num_envs),
            "reward": (self.cfg.num_steps, self.cfg.num_envs),
            "cost": (self.cfg.num_steps, self.cfg.num_envs),
            "done": (self.cfg.num_steps, self.cfg.num_envs),
            "value": (self.cfg.num_steps, self.cfg.num_envs),
        }
        self.buffer = CyclicBuffer(
            capacity=self.cfg.num_steps, feature_shapes=buffer_shapes, device=self.device
        )

        self.optimizer = optim.Adam(
            self.agent.parameters(), lr=self.cfg.learning_rate, eps=1e-5
        )
        self.iteration = 0

        self._logger = logging.getLogger(__name__)

    def train(self, envs, writer=None, verbose=1) -> dict[str, np.ndarray]:
        device = self.device

        # verbosity
        verbose = min(max(verbose, 0), len(LOGGING_LEVELS) - 1)
        self._logger.setLevel(LOGGING_LEVELS[verbose])
        self._logger.debug("Starting training...")

        # statistics
        train_steps = []
        train_lengths = []
        train_return = []
        train_cost = []

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        next_obs, _ = envs.reset(seed=self.cfg.seed)
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(self.cfg.num_envs).to(device)

        for iteration in range(1, self.cfg.num_iterations + 1):
            self._logger.info(
                f"iteration {iteration}/{self.cfg.num_iterations} \n"
                f"\tglobal step: {global_step}/{self.cfg.total_timesteps} \n"
                f"\tepisodic returns: {np.mean(train_return[-10:]):.2f} +/- {np.std(train_return[-10:]):.2f} \n"
                f"\tepisodic costs: {np.mean(train_cost[-10:]):.2f} +/- {np.std(train_cost[-10:]):.2f} \n"
                f"\tepisodic lengths: {np.mean(train_lengths[-10:]):.2f} +/- {np.std(train_lengths[-10:]):.2f} \n"
            )

            agent = self.get_actor()

            # data collection
            for step in range(0, self.cfg.num_steps):
                global_step += self.cfg.num_envs
                cur_obs = torch.clone(next_obs)

                # ALGO LOGIC: action logic
                # note: we dont want to keep gradients of inference but we cannot use torch.nograd because we need
                # autograd for the barrier derivative
                # with torch.no_grad():
                results = agent.get_action_and_value(next_obs)
                results = {
                    k: v.detach() for k, v in results.items()
                }  # solution: detach all returned values
                action = (
                    results["safe_action"]
                    if "safe_action" in results
                    else results["action"]
                )

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = envs.step(
                    action.cpu().numpy()
                )
                next_done = np.logical_or(terminations, truncations)
                next_obs, next_done = (
                    torch.Tensor(next_obs).to(device),
                    torch.Tensor(next_done).to(device),
                )
                envs.envs[0].render()

                if "final_info" not in infos:
                    cost = (
                        infos["costs"][0]
                        if "costs" in infos
                        else np.zeros(self.cfg.num_envs, dtype=np.float32)
                    )
                else:
                    cost = []
                    for i, info in enumerate(infos["final_info"]):
                        if info is None:
                            c = (
                                infos["costs"][i]
                                if "costs" in infos
                                else np.zeros(self.cfg.num_envs, dtype=np.float32)
                            )
                            cost.append(c)
                        else:
                            c = (
                                info["costs"]
                                if "costs" in infos
                                else np.zeros(self.cfg.num_envs, dtype=np.float32)
                            )
                            cost.append(c)
                    cost = np.array(cost)

                self.buffer.push(
                    obs=cur_obs,
                    done=next_done,
                    reward=torch.tensor(reward).to(device).view(-1),
                    cost=torch.tensor(cost).to(device).view(-1),
                    **results,
                )

                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            self._logger.debug(
                                f"global_step={global_step}, episodic_return={info['episode']['r']}, episodic_cost={info['episode']['c']}"
                            )

                            train_steps.append(int(global_step))
                            train_lengths.append(float(info["episode"]["l"]))
                            train_return.append(float(info["episode"]["r"]))
                            train_cost.append(float(info["episode"]["c"]))

                            if writer:
                                writer.add_scalar(
                                    "charts/episodic_return", train_return[-1], train_steps[-1]
                                )
                                writer.add_scalar(
                                    "charts/episodic_cost", train_cost[-1], train_steps[-1]
                                )
                                writer.add_scalar(
                                    "charts/episodic_length", train_lengths[-1], train_steps[-1]
                                )

            # update agent
            train_infos = self._update(next_obs=next_obs, next_done=next_done, )

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            sps = int(global_step / (time.time() - start_time))
            self._logger.info(f"SPS: {sps}")
            if writer:
                for k, v in train_infos.items():
                    writer.add_scalar(k, v, global_step)
                writer.add_scalar(
                    "charts/SPS", int(global_step / (time.time() - start_time)), global_step
                )

        return {
            "train_steps": np.array(train_steps),
            "train_lengths": np.array(train_lengths),
            "train_returns": np.array(train_return),
            "train_costs": np.array(train_cost)
        }

    def _update(self, next_obs, next_done, ) -> dict[str, float]:
        data = self.buffer.sample()
        obs = data["obs"]
        logprobs = data["logprob"]
        actions = data["action"]
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

                results = self.agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                newlogprob = results["logprob"]
                entropy = results["entropy"]
                newvalue = results["value"]

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > self.cfg.clip_coef)
                        .float()
                        .mean()
                        .item()
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

                entropy_loss = entropy.mean()
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

    def _advantage_estimation(
            self, obs, actions, rewards, dones, next_obs, next_done, values
    ):
        # bootstrap value if not done
        with torch.no_grad():
            next_value = self.agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(self.cfg.num_steps)):
                if t == self.cfg.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                        rewards[t]
                        + self.cfg.gamma * nextvalues * nextnonterminal
                        - values[t]
                )
                advantages[t] = lastgaelam = (
                        delta
                        + self.cfg.gamma
                        * self.cfg.gae_lambda
                        * nextnonterminal
                        * lastgaelam
                )
            returns = advantages + values

        return advantages, returns

    def get_actor(self) -> ActorCriticAgent:
        return self.agent
