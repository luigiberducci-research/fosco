from functools import partial

import numpy as np
import torch

from fosco.common import domains
from fosco.common.consts import TimeDomain
from fosco.common.domains import Set, Rectangle
from fosco.systems import SingleIntegrator, make_system
from fosco.systems.core.system import register, ControlAffineDynamics
from fosco.systems.gym_env.system_env import RenderObject, SystemEnv


class MultiParticle(ControlAffineDynamics):
    """
    Multi-particle system with state
    [dx1_0, dx1_1, ..., dx2_0, dx2_1, ...] relative to the first agent (ego),
    the control inputs [u0_0, u0_1, ...] are for the first agent (ego).

    dX/dt = 0 - ID u
    """



    def __init__(
            self,
            n_agents: int,
            single_agent_dynamics: ControlAffineDynamics,
            initial_distance: float = 3.0,
            collision_distance: float = 1.0,
    ):
        super().__init__()
        assert n_agents > 1, f"expected at least two agents, got {n_agents}"
        self._n_agents = n_agents
        self._single_agent_dynamics = single_agent_dynamics

        self._initial_distance = initial_distance
        self._collision_distance = collision_distance

    @property
    def id(self) -> str:
        return self.__class__.__name__ + self._single_agent_dynamics.id + f"{self._n_agents}"

    @property
    def vars(self) -> tuple[str, ...]:
        variables = []
        # _aid suffix for the agent id but 0 for the first agent (ego)
        for aid in range(1, self._n_agents):
            variables.extend([f"d{var}_{aid}" for var in self._single_agent_dynamics.vars])
        return tuple(variables)

    @property
    def controls(self) -> tuple[str, ...]:
        # _0 suffix for the first agent (ego)
        return tuple([f"{uvar}_0" for uvar in self._single_agent_dynamics.controls])

    @property
    def time_domain(self) -> TimeDomain:
        return self._single_agent_dynamics.time_domain

    @property
    def state_domain(self) -> Rectangle:
        # assume the state domain is the same for all agents, replicate it for all agents
        lb = np.array(self._single_agent_dynamics.state_domain.lower_bounds * (self._n_agents - 1))
        ub = np.array(self._single_agent_dynamics.state_domain.upper_bounds * (self._n_agents - 1))
        return domains.Rectangle(
            vars=self.vars,
            lb=lb,
            ub=ub,
        )


    @property
    def input_domain(self) -> Set:
        # the input domain of the ego agent
        return self._single_agent_dynamics.input_domain

    @property
    def init_domain(self) -> Set:
        """
        For each pair of agents, the initial domain is the complement of an unsafe box with halfwidth `initial_distance`
        The joint initial domain is the intersection of all these boxes
        """
        assert isinstance(self.state_domain, domains.Rectangle), f"expected Rectangle, got {self.state_domain}"
        lowerbound = self.state_domain.lower_bounds
        upperbound = self.state_domain.upper_bounds
        init_domains = []
        for aid in range(1, self._n_agents):
            lb = np.array(lowerbound)
            ub = np.array(upperbound)

            # set the initial range for the agent aid
            state_i = aid - 1
            n_vars_single = self._single_agent_dynamics.n_vars
            lb[n_vars_single * state_i: n_vars_single * (state_i + 1)] = -self._initial_distance
            ub[n_vars_single * state_i: n_vars_single * (state_i + 1)] = self._initial_distance

            complement_box_i = domains.Complement(
                domains.Rectangle(
                    vars=self.vars,
                    lb=lb,
                    ub=ub,
                ),
                outer_set=self.state_domain
            )
            init_domains.append(complement_box_i)

        return domains.Intersection(init_domains)

    @property
    def unsafe_domain(self) -> Set:
        """
        for simplicity in higher dimensions, we use boxes instead of spheres
        among each pair of agents, the unsafe domain is a box with halfwidth `collision_distance`
        the joint unsafe domain is the union of all these spheres
        """
        assert isinstance(self.state_domain, domains.Rectangle), f"expected Rectangle, got {self.state_domain}"
        lowerbound = self.state_domain.lower_bounds
        upperbound = self.state_domain.upper_bounds
        unsafe_domains = []
        for aid in range(1, self._n_agents):
            lb = np.array(lowerbound)
            ub = np.array(upperbound)

            # set the collision range for the agent aid
            state_i = aid - 1
            n_vars_single = self._single_agent_dynamics.n_vars
            lb[n_vars_single * state_i: n_vars_single * (state_i + 1)] = -self._collision_distance
            ub[n_vars_single * state_i: n_vars_single * (state_i + 1)] = self._collision_distance

            box_i = domains.Rectangle(
                vars=self.vars,
                lb=lb,
                ub=ub,
            )
            unsafe_domains.append(box_i)

        return domains.Union(unsafe_domains)

    def fx_torch(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        assert (
                len(x.shape) == 3
        ), f"expected batched input with shape (batch_size, state_dim, 1), got shape {x.shape}"
        if isinstance(x, np.ndarray):
            fx = np.zeros_like(x)
        else:
            fx = torch.zeros_like(x)
        return fx

    def fx_smt(self, x: list) -> np.ndarray | torch.Tensor:
        assert isinstance(
            x, list
        ), "expected list of symbolic state variables, [x0, x1, ...]"
        return np.zeros(len(x))

    def gx_torch(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        assert (
                len(x.shape) == 3
        ), "expected batched input with shape (batch_size, state_dim, 1)"
        if isinstance(x, np.ndarray):
            gx = np.tile(- np.eye(self.n_controls), (self._n_agents - 1, 1))
            gx_batch = np.tile(gx, (x.shape[0], 1, 1))
        else:
            gx = torch.eye(self.n_controls).repeat((self._n_agents - 1, 1))
            gx_batch = gx[None].repeat(x.shape[0], 1, 1).to(x.device)
        return gx_batch

    def gx_smt(self, x: list) -> np.ndarray | torch.Tensor:
        assert isinstance(
            x, list
        ), "expected list of symbolic state variables, [x0, x1, ...]"
        return - np.tile(np.eye(self.n_controls), (self._n_agents - 1, 1))

    def render_state_with_objects(self, obs: np.ndarray) -> list[RenderObject]:
        if len(obs.shape) == 2 and obs.shape[0] > 1:
            return []


        # ego agent
        ego_obj = RenderObject(
            type="circle",
            position=np.zeros(2,),
            size=self._collision_distance / 2,
            color=[0, 0, 200]
        )

        # other agents
        n_vars_single = self._single_agent_dynamics.n_vars
        objects = []
        for aid in range(1, self._n_agents):
            state_i = aid - 1
            x = obs.squeeze()[n_vars_single * state_i: n_vars_single * (state_i + 1)]
            objects.append(
                RenderObject(
                    type="circle",
                    position=x[:2],
                    size=self._collision_distance / 2,
                    color=[200, 0, 0]
                )
            )

        return [ego_obj] + objects

register(
    name="MultiParticleSingleIntegrator",
    entrypoint=partial(MultiParticle, single_agent_dynamics=SingleIntegrator()),
)

if __name__=="__main__":
    system = make_system("MultiParticleSingleIntegrator")(n_agents=4)
    env = SystemEnv(system=system, max_steps=100, render_mode="human")
    obs = env.reset()
    env.render()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, term, truncated, info = env.step(action)
        done = term or truncated
        env.render()
        print(f"obs: {obs}, reward: {reward}, done: {done}")
    env.close()
    print("done")