import logging

import tyro

from rl_trainer.run_ppo import Args
from rl_trainer import run_ppo

def main(args):
    logging.basicConfig(level=logging.INFO)
    logdir = run_ppo.run(args=args)
    logging.info(f"Done. Results stored in {logdir}")

if __name__=="__main__":
    args_ = Args
    args = tyro.cli(args_)
    args.env_id = "systems:SingleIntegrator-GoToUnsafeReward-v0"
    main(args=args)