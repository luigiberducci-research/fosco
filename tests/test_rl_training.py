import unittest



class TestRLTraining(unittest.TestCase):


    def test_ppo_hopper(self):
        from rl_trainer.run_ppo import Args, run
        from rl_trainer.common.utils import tflog2pandas
        import scipy

        args = Args
        args.seed = 0
        args.env_id = "Hopper-v4"
        args.trainer_id = "ppo"
        args.total_timesteps = 7500

        logdir = run(args=args)
        df = tflog2pandas(logdir)

        returns = df[df["metric"] == "charts/episodic_return"]["value"]
        steps = df[df["metric"] == "charts/episodic_return"]["step"]

        linreg = scipy.stats.linregress(steps, returns)
        print(linreg)
        slope = linreg.slope
        self.assertTrue(slope > 0, f"We expect a positive trend in return (learning something), got slope={slope}")


    def test_safeppo_hopper(self):
        from rl_trainer.run_ppo import Args, run
        from rl_trainer.common.utils import tflog2pandas
        import scipy

        args = Args
        args.seed = 0
        args.env_id = "Hopper-v4"
        args.trainer_id = "safe-ppo"
        args.total_timesteps = 7500

        logdir = run(args=args)
        df = tflog2pandas(logdir)

        returns = df[df["metric"] == "charts/episodic_return"]["value"]
        steps = df[df["metric"] == "charts/episodic_return"]["step"]

        linreg = scipy.stats.linregress(steps, returns)
        print(linreg)
        slope = linreg.slope
        self.assertTrue(slope > 0, f"We expect a positive trend in return (learning something), got slope={slope}")
