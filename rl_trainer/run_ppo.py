import logging
import os
import pathlib
import random
import time
import warnings
from dataclasses import dataclass
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import tyro
from torch.utils.tensorboard import SummaryWriter

from fosco.systems.gym_env.system_env import SystemEnv
from rl_trainer.ppo.ppo_config import PPOConfig
from rl_trainer.safe_ppo.safeppo_trainer import SafePPOTrainer
from rl_trainer.ppo.ppo_trainer import PPOTrainer
from rl_trainer.common.utils import make_env


@dataclass
class Args(PPOConfig):
    exp_name: str = pathlib.Path(__file__).stem
    """the name of this experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    capture_video_eval: bool = True
    """whether to capture videos during evaluation"""
    render_mode: str = None
    """render mode during training if no capture video"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    logdir: str = f"{pathlib.Path(__file__).parent.parent}/runs"
    """the directory to save the logs"""
    env_id: str = "Hopper-v4"  # "systems:SingleIntegrator-GoToUnsafeReward-v0"
    """the id of the environment"""
    trainer_id: str = "ppo"
    """the id of the rl trainer"""


def evaluate(
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    logdir: str,
    agent: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = True,
    gamma: float = 0.99,
):
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                env_id=env_id,
                seed=None,
                idx=0,
                capture_video=capture_video,
                logdir=f"{logdir}/eval",
                gamma=gamma,
            )
        ]
    )
    agent.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    episodic_costs = []
    while len(episodic_returns) < eval_episodes:
        results = agent.get_action_and_value(torch.Tensor(obs).to(device))
        actions = results["action"]
        next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(
                    f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}, episodic_cost={info['episode']['c']}"
                )
                episodic_returns += [info["episode"]["r"]]
                episodic_costs += [info["episode"]["c"]]
        obs = next_obs

    return episodic_returns, episodic_costs


def run(args):
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    env_id = args.env_id if isinstance(args.env_id, str) else "env"
    run_name = f"{env_id}__{args.trainer_id}__{args.seed}__{int(time.time())}"

    logdir = None
    if args.logdir:
        logdir = f"{args.logdir}/{run_name}"

    if args.num_iterations == 0:
        raise ValueError(
            "Number of iterations = 0, maybe total timestep <= numenvs*numsteps?"
        )

    writer = None
    if logdir is not None:
        writer = SummaryWriter(logdir)
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
    logging.info(f"Logdir {logdir}")

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    logging.info(f"Running on device {device}")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                env_id=args.env_id,
                seed=args.seed,
                idx=i,
                capture_video=args.capture_video,
                logdir=logdir,
                gamma=args.gamma,
                render_mode=args.render_mode,
            )
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    if args.trainer_id == "ppo":
        trainer = PPOTrainer(envs=envs, config=args, device=device)
    elif args.trainer_id == "safe-ppo":
        if not isinstance(envs.envs[0].unwrapped, SystemEnv):
            raise TypeError(f"SafePPO only supports SystemEnv, got {envs.envs[0].unwrapped}")
        from barriers import make_barrier
        system = envs.envs[0].unwrapped.system
        barrier = make_barrier(system=system)
        trainer = SafePPOTrainer(envs=envs, barrier=barrier, config=args, device=device)
    else:
        raise NotImplementedError(f"No trainer implemented for id={args.trainer_id}")

    results = trainer.train(envs=envs, writer=writer)

    if logdir and args.save_model:
        raise warnings.warn("saving model is not tested yet")
        checkpoint_dir = pathlib.Path(f"{logdir}/checkpoints/")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        model_path = checkpoint_dir / f"model_{global_step}.pt"
        torch.save(agent.state_dict(), str(model_path))
        print(f"model saved to {model_path}")

        del agent
        agent = trainer.get_actor()
        agent.load(model_path=model_path, device=device)

    if args.num_eval_episodes > 0:
        episodic_returns, episodic_costs = evaluate(
            make_env=make_env,
            env_id=args.env_id,
            eval_episodes=args.num_eval_episodes,
            logdir=logdir,
            capture_video=args.capture_video_eval,
            agent=agent,
            device=device,
            gamma=args.gamma,
        )
        print(
            f"eval episodic returns: {np.mean(episodic_returns)} +/- {np.std(episodic_returns)}"
        )
        print(
            f"eval episodic costs: {np.mean(episodic_costs)} +/- {np.std(episodic_costs)}"
        )

        if writer:
            for idx, (episodic_return, episodic_cost) in enumerate(
                zip(episodic_returns, episodic_costs)
            ):
                writer.add_scalar("eval/episodic_return", episodic_return, idx)
                writer.add_scalar("eval/episodic_cost", episodic_cost, idx)

    envs.close()

    if writer:
        writer.close()

    return logdir


if __name__ == "__main__":
    args = tyro.cli(Args)
    run(args=args)
