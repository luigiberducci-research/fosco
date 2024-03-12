import numpy as np
import torch
from matplotlib import pyplot as plt

from fosco.systems import make_system
from fosco.systems import SystemEnv


def dummy_term_fn(actions, next_obss):
    return torch.zeros((actions.shape[0],), dtype=torch.bool)


def dummy_reward_fn(actions, next_obss):
    return torch.ones((next_obss.shape[0],), dtype=torch.float32)


def main():
    system_id = "DoubleIntegrator"
    max_steps = 100
    dt = 0.1
    batch_size = 1

    system = make_system(system_id=system_id)()
    env = SystemEnv(
        system=system,
        termination_fn=dummy_term_fn,
        reward_fn=dummy_reward_fn,
        max_steps=max_steps,
        dt=dt,
        render_mode="human"
    )

    # plotting
    xs, ys = [], []

    obss, infos = env.reset(options={"batch_size": batch_size})
    terminations = truncations = np.zeros(batch_size, dtype=bool)
    while not any(terminations) and not any(truncations):
        print(obss)
        actions = np.stack([np.ones(2, dtype=np.float32) for _ in range(batch_size)])

        obs, rewards, terminations, truncations, infos = env.step(actions)
        env.render()

        xs.append(obs[:, 0])
        ys.append(obs[:, 1])

    env.close()

    # plotting
    xs = np.array(xs)
    ys = np.array(ys)

    plt.plot(xs, ys)
    plt.show()



if __name__ == "__main__":
    main()
