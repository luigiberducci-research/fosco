from typing import Any, SupportsFloat

import gymnasium
import numpy as np
from gymnasium.core import ObsType, ActType

from fosco.common.domains import Set, Rectangle
from systems import ControlAffineControllableDynamicalModel


class SystemEnv(gymnasium.Env):
    """
    Wrap a system as a gym environment, implements the necessary api.
    """

    def __init__(
        self,
        system: ControlAffineControllableDynamicalModel,
        domains: dict[str, Set],
        dt: float = 0.1,
        max_time: float = 100.0,
    ) -> None:
        super().__init__()

        self._assert_input(system=system, domains=domains)
        self.system = system
        self.domains = domains
        self.dt = dt
        self.max_time = max_time

        input_domain: Rectangle = domains["input"]
        self.action_space = gymnasium.spaces.Box(
            low=np.array(input_domain.lower_bounds),
            high=np.array(input_domain.upper_bounds),
            shape=(self.system.n_controls,),
            dtype=np.float32,
        )

        # todo: restrict observation space to lie domain?
        self.observation_space = gymnasium.spaces.Box(
            low=np.array([-np.inf] * self.system.n_vars)[None],
            high=np.array([np.inf] * self.system.n_vars)[None],
            dtype=np.float32,
        )

        self.state = None
        self.time = None

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed, options=options)

        self.state = self.domains["init"].generate_data(1).detach().numpy()
        self.time = 0.0

        return self.state, {}

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        assert self.state is not None, "must call reset() before step()"

        dstate = self.system.f(v=self.state, u=action)
        self.state += dstate * self.dt
        self.time += self.dt

        # reward
        reward = 0.0

        # check termination
        terminated = bool(self.time >= self.max_time)
        truncated = bool(self.domains["unsafe"].check_containment(self.state))

        return self.state, reward, terminated, truncated, {}

    def render(
        self, mode: str = "human", *, options: dict[str, Any] | None = None,
    ) -> Any:
        raise NotImplementedError()

    def _assert_input(self, system, domains):
        assert isinstance(
            system, ControlAffineControllableDynamicalModel
        ), f"system must be a ControlAffineControllableDynamicalModel, got {type(system)}"
        assert isinstance(domains, dict), f"domains must be a dict, got {type(domains)}"

        assert "init" in domains, "must specify init domain"
        assert "unsafe" in domains, "must specify unsafe domain"
        assert "input" in domains, "must specify input domain"

        assert isinstance(
            domains["init"], Set
        ), f"init domain must be a Set, got {type(domains['init'])}"
        assert isinstance(
            domains["unsafe"], Set
        ), f"unsafe domain must be a Set, got {type(domains['unsafe'])}"
        assert isinstance(
            domains["input"], Set
        ), f"input domain must be a Set, got {type(domains['input'])}"

        assert isinstance(
            domains["input"], Rectangle
        ), f"init domain must be a Rectangle, got {type(domains['init'])}"
        assert (
            len(domains["input"].lower_bounds)
            == len(domains["input"].upper_bounds)
            == system.n_controls
        ), f"input domain must have {system.n_controls} dimensions, got lower/upperbound mismatch"
