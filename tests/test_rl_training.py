import unittest
import shutil


class TestRLTraining(unittest.TestCase):
    def _test_ppo_hopper(self):
        """
        Try to run ppo in the mujoco hopper environment.
        Expected to run fine.
        """
        from rl_trainer.run_ppo import Args, run
        from rl_trainer.common.utils import tflog2pandas
        import scipy

        args = Args
        args.seed = 0
        args.env_id = "Hopper-v4"
        args.trainer_id = "ppo"
        args.total_timesteps = 7500
        args.update_epochs = 10
        args.capture_video = args.capture_video_eval = False  # to make it faster
        args.render_mode = "rgb_array"

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

        # delete logdir once test is done
        shutil.rmtree(logdir)

    def _test_safeppo_hopper(self):
        """
        Try to run safe-ppo in the mujoco hopper environment.
        Expected to raise an exception because not SystemEnv.
        """
        from rl_trainer.run_ppo import Args, run

        args = Args
        args.seed = 0
        args.env_id = "Hopper-v4"
        args.update_epochs = 10
        args.trainer_id = "safe-ppo"
        args.total_timesteps = 7500
        args.capture_video = args.capture_video_eval = False  # to make it faster
        args.render_mode = "rgb_array"

        with self.assertRaises(TypeError):
            logdir = run(args=args)

            # delete logdir once test is done
            shutil.rmtree(logdir)

    def _test_ppo_single_integrator(self):
        """
        Try to run ppo in the single integrator environment.
        Expected to learn to go to the goal, with some cost violations.
        """
        from rl_trainer.run_ppo import Args, run
        from rl_trainer.common.utils import tflog2pandas
        import scipy

        args = Args
        args.seed = 0
        args.env_id = "fosco.systems:SingleIntegrator-GoToUnsafeReward-v0"
        args.trainer_id = "ppo"
        args.use_true_barrier = False
        args.update_epochs = 10
        args.num_steps = 64
        args.total_timesteps = 2000
        args.capture_video = args.capture_video_eval = False
        args.render_mode = None

        logdir = run(args=args)
        df = tflog2pandas(logdir)

        returns = df[df["metric"] == "charts/episodic_return"]["value"]
        costs = df[df["metric"] == "charts/episodic_cost"]["value"]
        steps = df[df["metric"] == "charts/episodic_return"]["step"]

        linreg = scipy.stats.linregress(steps, returns)
        print(linreg)
        slope = linreg.slope
        self.assertTrue(
            slope > 0,
            f"We expect a positive trend in return (learning something), got slope={slope}",
        )

        self.assertTrue(
            any(costs > 0), f"We expect some costs to be >0, got {costs}",
        )

        # delete logdir once test is done
        shutil.rmtree(logdir)

    def _test_safeppo_single_integrator(self):
        """
        Try to run ppo in the single integrator environment.
        Expected to learn to go to the goal, with no cost violations.
        """
        from rl_trainer.run_ppo import Args, run
        from rl_trainer.common.utils import tflog2pandas
        import scipy

        args = Args
        args.seed = 0
        args.env_id = "fosco.systems:SingleIntegrator-GoToUnsafeReward-v0"
        args.trainer_id = "safe-ppo"
        args.use_true_barrier = True
        args.update_epochs = 10
        args.num_steps = 64
        args.total_timesteps = 2000
        args.capture_video = args.capture_video_eval = False
        args.render_mode = None

        logdir = run(args=args)
        df = tflog2pandas(logdir)

        returns = df[df["metric"] == "charts/episodic_return"]["value"]
        costs = df[df["metric"] == "charts/episodic_cost"]["value"]
        steps = df[df["metric"] == "charts/episodic_return"]["step"]

        linreg = scipy.stats.linregress(steps, returns)
        print(linreg)
        slope = linreg.slope
        self.assertTrue(
            slope > 0,
            f"We expect a positive trend in return (learning something), got slope={slope}",
        )

        self.assertTrue(
            all(costs <= 0), f"We expect all costs to be 0, got {costs}",
        )

        # delete logdir once test is done
        shutil.rmtree(logdir)

    def test_evaluate_fn(self):
        """
        Test the evaluate function from the ppo trainer and check logs returns and costs.
        Expected to run fine.
        """
        from rl_trainer.run_ppo import Args, run
        from rl_trainer.common.utils import tflog2pandas

        args = Args
        args.env_id = "fosco.systems:SingleIntegrator-GoToUnsafeReward-v0"
        args.trainer_id = "ppo"
        args.total_timesteps = args.num_steps = 64  # to make it faster
        args.update_epochs = 1  # to make it faster
        args.num_eval_episodes = 1  # to make it faster
        args.capture_video = args.capture_video_eval = False  # to make it faster
        args.render_mode = None

        logdir = run(args=args)
        self.assertTrue(
            logdir is not None,
            f"Expected to have a logdir where to store metrics, got {logdir}",
        )

        # tboard to df
        df = tflog2pandas(logdir)
        returns = df[df["metric"] == "eval/episodic_return"]["value"]
        costs = df[df["metric"] == "eval/episodic_cost"]["value"]

        self.assertTrue(
            len(returns) > 0, f"We expect to have some returns, got {len(returns)}",
        )
        self.assertTrue(
            len(costs) > 0, f"We expect to have some costs, got {len(costs)}",
        )

        # delete logdir once test is done
        shutil.rmtree(logdir)
