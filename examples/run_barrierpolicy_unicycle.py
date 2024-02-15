import numpy as np
import torch

from models.policy import BarrierPolicy
from systems import make_system


def main():
    seed = 0
    system_id = "UnicycleAcc"
    n_episodes = 100
    T, dt = 10.0, 0.1
    obs_x, obs_y, R = 40, 15, 6
    hidden_sizes = (128, 32, 32)
    device = "cpu"

    # seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    sys = make_system(system_id=system_id)()
    policy = BarrierPolicy(
        nFeatures=sys.n_vars,
        nHidden1=hidden_sizes[0],
        nHidden21=hidden_sizes[1],
        nHidden22=hidden_sizes[2],
        nCls=sys.n_controls,
        mean=np.zeros(sys.n_vars),
        std=np.ones(sys.n_vars),
        device=device,
        bn=False,
        obs_x=obs_x,
        obs_y=obs_y,
        R=R,
    )
    policy.eval()


    # reset
    state = torch.zeros((n_episodes, sys.n_vars))
    t = 0.0
    traj = [state.numpy().copy()]

    while t < T:
        with torch.no_grad():
            ctrl = policy(state, 1).float()
        dstate = sys.f(state, ctrl)
        state += dstate * dt
        traj.append(state.numpy().copy())
        t += dt


    # plot
    import matplotlib.pyplot as plt

    plt.scatter(obs_x, obs_y, s=100, c="r", marker="x")
    plt.plot(obs_x + R * np.cos(np.linspace(0, 2 * np.pi, 100)),
             obs_y + R * np.sin(np.linspace(0, 2 * np.pi, 100)),
             "r")

    all_trajs = np.stack(traj, 1)
    for traj in all_trajs:
        plt.plot(traj[:, 0], traj[:, 1])
    plt.show()




if __name__=="__main__":
    main()