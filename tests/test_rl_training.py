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
        self.assertTrue(
            slope > 0,
            f"We expect a positive trend in return (learning something), got slope={slope}",
        )

    def test_safeppo_hopper(self):
        """
        Try to run safe-ppo in the mujoco hopper environment.
        Expected to raise an exception because not SystemEnv.
        """
        from rl_trainer.run_ppo import Args, run

        args = Args
        args.seed = 0
        args.env_id = "Hopper-v4"
        args.trainer_id = "safe-ppo"
        args.total_timesteps = 7500

        with self.assertRaises(TypeError):
            logdir = run(args=args)

    def test_safeppo_single_integrator(self):
        """
        Try to run safe-ppo in the mujoco hopper environment.
        Expected to run fine.
        """
        from rl_trainer.run_ppo import Args, run
        from rl_trainer.common.utils import tflog2pandas
        import scipy

        args = Args
        args.seed = 0
        args.env_id = "fosco.systems:SingleIntegrator-GoToUnsafeReward-v0"
        args.trainer_id = "safe-ppo"
        args.total_timesteps = 7500

        logdir = run(args=args)
        df = tflog2pandas(logdir)

        returns = df[df["metric"] == "charts/episodic_return"]["value"]
        steps = df[df["metric"] == "charts/episodic_return"]["step"]

        linreg = scipy.stats.linregress(steps, returns)
        print(linreg)
        slope = linreg.slope
        self.assertTrue(
            slope > 0,
            f"We expect a positive trend in return (learning something), got slope={slope}",
        )
