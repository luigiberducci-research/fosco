import gymnasium as gym
import numpy as np
import pandas as pd


def make_env(env_id, seed, idx, capture_video, logdir, gamma, render_mode=None):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"{logdir}/videos")
        else:
            env = gym.make(env_id, render_mode=render_mode)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        #env = gym.wrappers.ClipAction(env)
        #env = gym.wrappers.NormalizeObservation(env)
        #env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def tflog2pandas(path: str) -> pd.DataFrame:
    """
    Convert a tensorboard log file to pandas dataframe.
    """
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        event_acc = EventAccumulator(path)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        import traceback
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data