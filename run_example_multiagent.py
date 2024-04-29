import gymnasium
import numpy as np

from fosco.systems import make_system, MultiParticle
from fosco.systems.gym_env.system_env import SystemEnv


def main():
    single_sys = make_system("SingleIntegrator")()
    multi_sys = MultiParticle(single_agent_dynamics=single_sys, n_agents=3)

    env = SystemEnv(
        system=multi_sys,
        dt=0.1,
        max_steps=100,
        render_mode="human"
    )

    env.reset()
    env.render()
    done = False

    while not done:
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        env.render()

    env.close()



if __name__ == '__main__':
    main()