import pathlib
from argparse import Namespace
from functools import partial
from typing import Optional

import gymnasium
import numpy as np
import torch
from torch import nn

from barriers import make_barrier
from rl_trainer.safe_ppo.safeppo_agent import SafeActorCriticAgent
from rl_trainer.common.buffer import CyclicBuffer
from rl_trainer.ppo.ppo_trainer import PPOTrainer
from fosco.systems.system_env import SystemEnv
from fosco.learner import make_learner

class SafePPOTrainer(PPOTrainer):
    def __init__(
            self,
            envs: gymnasium.Env,
            args: Namespace,
            device: Optional[torch.device] = None,
    ) -> None:
        if not args.use_true_barrier and not args.barrier_path:
            raise TypeError("safe ppo needs a cbf, either known or learned")

        if args.barrier_path:
            single_env = envs.envs[0] if envs.unwrapped.is_vector_env else envs
            system = single_env.system

            if pathlib.Path(args.barrier_path).exists():
                raise NotImplementedError
            else:
                # load model from logs
                from aim import Run
                from fosco.common.consts import TimeDomain, ActivationType

                aim_run = Run(run_hash=args.barrier_path)
                config = aim_run["config"]

                timedomain = eval(config["TIME_DOMAIN"])
                learner_type = make_learner(system=system, time_domain=timedomain)
                # todo rewrite logs in primitive types to make loading easier
                learner = learner_type(
                    state_size=system.n_vars,
                    learn_method=None,
                    hidden_sizes=eval(config["N_HIDDEN_NEURONS"]),  # todo: load from cfg instead of hardcoding
                    activation=eval(config["ACTIVATION"]),  # todo: load from cfg
                    optimizer=eval(config["OPTIMIZER"]),
                    lr=eval(config["LEARNING_RATE"]),
                    weight_decay=eval(config["WEIGHT_DECAY"]),
                )

                model_path = pathlib.Path(config["MODEL_DIR"]) / config["EXP_NAME"]
                model_path = [p for p in model_path.glob("*pt")]

                if len(model_path) == 0:
                    raise FileNotFoundError(f"no model found in {model_path}")

                model_path = model_path[0]  # pick first

            learner.load(model_path=model_path)
            barrier = learner.net
            barrier.train()
        else:
            single_env = envs.envs[0] if envs.unwrapped.is_vector_env else envs
            if not isinstance(single_env.unwrapped, SystemEnv):
                raise TypeError(
                    f"This trainer runs only in SystemEnv because it relies on the dynamics, got {single_env.unwrapped}"
                )
            system = single_env.system
            barrier = make_barrier(system=system)["barrier"]

        agent_cls = partial(SafeActorCriticAgent, barrier=barrier)
        super().__init__(
            envs=envs,
            args=args,
            agent_cls=agent_cls,
            device=device,
        )

        buffer_shapes = {
            "obs": (args.num_steps, args.num_envs) + envs.single_observation_space.shape,
            "action": (args.num_steps, args.num_envs) + envs.single_action_space.shape,
            "classk": (args.num_steps, args.num_envs) + (1,),
            "logprob": (args.num_steps, args.num_envs),
            "classk_logprob": (args.num_steps, args.num_envs),
            "reward": (args.num_steps, args.num_envs),
            "done": (args.num_steps, args.num_envs),
            "value": (args.num_steps, args.num_envs),
        }
        self.buffer = CyclicBuffer(
            capacity=args.num_steps,
            feature_shapes=buffer_shapes,
            device=self.device
        )

    def train(
            self,
            next_obs: torch.Tensor,
            next_done: Optional[torch.Tensor] = None,
    ) -> dict[str, float]:
        data = self.buffer.sample()
        obs = data["obs"]
        logprobs = data["logprob"]
        classk_logprobs = data["classk_logprob"]
        actions = data["action"]
        classks = data["classk"]
        rewards = data["reward"]
        dones = data["done"]
        values = data["value"]

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
        b_classks = classks.reshape((-1,) + (self.agent.classk_size,))
        b_classk_logprobs = classk_logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # update policy and value
        # Optimizing the policy and value network
        b_inds = np.arange(self.args.batch_size)
        clipfracs = []
        for epoch in range(self.args.update_epochs):
            print(f"epoch {epoch}")
            np.random.shuffle(b_inds)
            for start in range(0, self.args.batch_size, self.args.minibatch_size):
                end = start + self.args.minibatch_size
                mb_inds = b_inds[start:end]

                # note: logprob, entropy, newvalue do not depende on safe_action -> disable it
                results = self.agent.get_action_and_value(x=b_obs[mb_inds],
                                                          action=b_actions[mb_inds],
                                                          action_k=b_classks[mb_inds],
                                                          use_safety_layer=False)
                newlogprob = results["logprob"]
                newclassklogprob = results["classk_logprob"]
                entropy = results["entropy"]
                classkentropy = results["classk_entropy"]
                newvalue = results["value"]

                logratio = newlogprob + newclassklogprob - b_logprobs[mb_inds] - b_classk_logprobs[mb_inds]
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

                entropy_loss = 0.5 * (entropy.mean() + classkentropy.mean())
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
