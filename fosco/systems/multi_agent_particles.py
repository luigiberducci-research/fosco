from functools import partial

import numpy as np
import torch

from fosco.common import domains
from fosco.common.consts import TimeDomain
from fosco.common.domains import Set
from fosco.systems import SingleIntegrator
from fosco.systems.core.system import register, ControlAffineDynamics


class MultiParticle(ControlAffineDynamics):
    """
    Multi-particle system with state [dx1_0, dx1_1, ..., dx2_0, dx2_1, ...] is relative to the first agent (ego),
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
