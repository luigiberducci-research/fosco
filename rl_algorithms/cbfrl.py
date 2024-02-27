# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import tyro
from torch.utils.tensorboard import SummaryWriter

from models.cbf_agent import SafeActorCriticAgent
from rl_algorithms.cbfagent_trainer import SafePPOTrainer
from rl_algorithms.ppo import Args, evaluate
from rl_algorithms.utils import make_env


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id=args.env_id, seed=args.seed, idx=i, capture_video=args.capture_video, run_name=run_name,
                  gamma=args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    trainer = SafePPOTrainer(envs=envs, args=args, device=device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        agent = trainer.get_actor()

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            cur_obs = torch.clone(next_obs)

            # ALGO LOGIC: action logic
            with torch.no_grad():
                results = agent.get_action_and_value(next_obs)
                action = results["action"]
                classk = results["class_k"]
                logprob = results["log_prob"]
                classk_logprob = results["class_k_log_prob"]
                value = results["value"]


            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            trainer.buffer.push(
                obs=cur_obs,
                dones=next_done,
                actions=action,
                classks=classk,
                logprobs=logprob,
                classk_logprobs=classk_logprob,
                values=value.flatten(),
                rewards=torch.tensor(reward).to(device).view(-1)
            )

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # update agent
        train_infos = trainer.train(
            next_obs=next_obs,
            next_done=next_done,
        )

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for k, v in train_infos.items():
            writer.add_scalar(k, v, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=SafeActorCriticAgent,
            device=device,
            gamma=args.gamma,
        )
        print(f"eval episodic returns: {np.mean(episodic_returns)} +/- {np.std(episodic_returns)}")
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

    envs.close()
    writer.close()
