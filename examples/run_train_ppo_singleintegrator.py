import logging
import time

import tyro

from rl_trainer.run_ppo import Args
from rl_trainer import run_ppo

def main(args):
    logging.basicConfig(level=logging.INFO)
    t0 = time.time()
    logdir = run_ppo.run(args=args)
    logging.info(f"Done in {time.time()-t0} sec")
    logging.info(f"Results stored in {logdir}")

if __name__=="__main__":
    args_ = Args
    args = tyro.cli(args_)
    args.env_id = "systems:SingleIntegrator-GoToUnsafeReward-v0"
    main(args=args)