import logging
import os
import pathlib
import random
import time
from dataclasses import dataclass
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import tyro
from torch.utils.tensorboard import SummaryWriter

from rl_trainer.safe_ppo.safeppo_trainer import SafePPOTrainer
from rl_trainer.ppo.ppo_trainer import PPOTrainer
from rl_trainer.common.utils import make_env


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
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
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""
    logdir: str = f"{pathlib.Path(__file__).parent.parent}/runs"
    """the directory to save the logs"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4" #"systems:SingleIntegrator-GoToUnsafeReward-v0"
    """the id of the environment"""
    trainer_id: str = "ppo"
    """the id of the rl trainer"""
    use_true_barrier: bool = False
    """toggle the use of grount-truth barrier"""
    barrier_path: str = None
    """barrier model or hash of training run from which load the barrier model"""
    total_timesteps: int = 50000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    num_eval_episodes: int = 10
    """the number of episodes to evaluate the policy at the end of training"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


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
    envs = gym.vector.SyncVectorEnv([make_env(env_id=env_id, seed=None, idx=0,
                                              capture_video=capture_video, logdir=f"{logdir}/eval",
                                              gamma=gamma)])
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
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}, episodic_cost={info['episode']['c']}")
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
    logdir = f"{args.logdir}/{run_name}"

    if args.num_iterations == 0:
        raise ValueError("Number of iterations = 0, maybe total timestep <= numenvs*numsteps?")

    writer = SummaryWriter(logdir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
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
        [make_env(env_id=args.env_id, seed=args.seed, idx=i, capture_video=args.capture_video, logdir=logdir,
                  gamma=args.gamma, render_mode=args.render_mode) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    if args.trainer_id == "ppo":
        trainer = PPOTrainer(envs=envs, args=args, device=device)
    elif args.trainer_id == "safe-ppo":
        trainer = SafePPOTrainer(envs=envs, args=args, device=device)
    else:
        raise NotImplementedError(f"No trainer implemented for id={args.trainer_id}")

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        agent = trainer.get_actor()

        # data collection
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            cur_obs = torch.clone(next_obs)

            # ALGO LOGIC: action logic
            # note: we dont want to keep gradients of inference but we cannot use torch.nograd because we need
            # autograd for the barrier derivative
            #with torch.no_grad():
            results = agent.get_action_and_value(next_obs)
            results = {k: v.detach() for k, v in results.items()}   # solution: detach all returned values
            action = results["safe_action"] if "safe_action" in results else results["action"]

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            envs.envs[0].render()

            if "final_info" not in infos:
                cost = infos["costs"][0] if "costs" in infos else np.zeros(args.num_envs, dtype=np.float32)
            else:
                cost = []
                for i, info in enumerate(infos["final_info"]):
                    if info is None:
                        c = infos["costs"][i] if "costs" in infos else np.zeros(args.num_envs, dtype=np.float32)
                        cost.append(c)
                    else:
                        c = info["costs"] if "costs" in infos else np.zeros(args.num_envs, dtype=np.float32)
                        cost.append(c)


            trainer.buffer.push(
                obs=cur_obs,
                done=next_done,
                reward=torch.tensor(reward).to(device).view(-1),
                cost=torch.tensor(cost).to(device).view(-1),
                **results
            )

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}, episodic_cost={info['episode']['c']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_cost", info["episode"]["c"], global_step)
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
        checkpoint_dir = pathlib.Path(f"{logdir}/checkpoints/")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        model_path = checkpoint_dir / f"model_{global_step}.pt"
        torch.save(agent.state_dict(), str(model_path))
        print(f"model saved to {model_path}")

        del agent
        agent = trainer.get_actor()
        agent.load(model_path=model_path, device=device)

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
        print(f"eval episodic returns: {np.mean(episodic_returns)} +/- {np.std(episodic_returns)}")
        print(f"eval episodic costs: {np.mean(episodic_costs)} +/- {np.std(episodic_costs)}")
        for idx, (episodic_return, episodic_cost) in enumerate(zip(episodic_returns, episodic_costs)):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)
            writer.add_scalar("eval/episodic_cost", episodic_cost, idx)

    envs.close()
    writer.close()

    return logdir


if __name__ == "__main__":
    args = tyro.cli(Args)
    run(args=args)
