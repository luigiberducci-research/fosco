# MIT License
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Dict, Optional, Tuple, Callable, Any

import gymnasium
import gymnasium as gym
import numpy as np
import torch
from gymnasium.core import ObsType

from fosco.common.domains import Rectangle
from systems import ControlAffineDynamics

RewardFnType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
TermFnType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
TensorType = torch.Tensor | np.ndarray


class SystemEnv(gymnasium.Env):
    """
    Wraps a dynamics model into a gym-like environment.

    This class can wrap a dynamics model to be used as an environment.
    The only requirement to use this class is for the model to use this wrapper is to have a method called
    ``predict()``
    with signature `next_observs, rewards = model.predict(obs,actions, sample=, rng=)`

    Args:
        # todo

    """

    def __init__(
        self,
        system: ControlAffineDynamics,
        termination_fn: TermFnType,
        reward_fn: RewardFnType,
        # generator: Optional[torch.Generator] = None,
        return_np: bool = True,
    ):
        # todo: generator for seeding the environment
        # todo: device to run on gpu
        # todo: propagation method for uncertain dynamical systems (basic: unif random)
        self.system = system
        self.termination_fn = termination_fn
        self.reward_fn = reward_fn

        self.observation_space = self.make_observation_space(system=self.system)
        self.action_space = self.make_action_space(system=self.system)

        self._current_obs: torch.Tensor = None
        self._return_as_np = return_np

    @staticmethod
    def make_observation_space(system: ControlAffineDynamics) -> gym.spaces.Space:
        assert isinstance(
            system.state_domain, Rectangle
        ), "only rectangle domains are supported for observation space"
        state_domain: Rectangle = system.state_domain
        return gym.spaces.Box(
            low=np.array(state_domain.lower_bounds),
            high=np.array(state_domain.upper_bounds),
            shape=(state_domain.dimension,),
        )

    @staticmethod
    def make_action_space(system: ControlAffineDynamics) -> gym.spaces.Space:
        assert isinstance(
            system.input_domain, Rectangle
        ), "only rectangle domains are supported for observation space"
        input_domain: Rectangle = system.input_domain
        return gym.spaces.Box(
            low=np.array(input_domain.lower_bounds),
            high=np.array(input_domain.upper_bounds),
            shape=(input_domain.dimension,),
        )

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[TensorType, dict[str, Any]]:
        if seed:
            super().reset(seed=seed)
            torch.manual_seed(seed)

        default_options = {
            "batch_size": 1,
            "return_as_np": True,
        }
        default_options.update(options or {})

        init_domain = self.system.init_domain
        batch_size = default_options["batch_size"]
        self._return_as_np = default_options["return_as_np"]

        # generate batch of tensor states
        self._current_obs = init_domain.generate_data(batch_size=batch_size)
        info = {}

        # eventually convert to numpy array
        if self._return_as_np:
            obs_batch = self._current_obs.cpu().numpy().astype(np.float32)
        else:
            obs_batch = self._current_obs

        # if no batch mode, unpack the first and only state
        if batch_size == 1:
            return obs_batch[0], info

        # otherwise, return batch of states
        return obs_batch, info

    def step(
        self,
        actions: TensorType,
    ) -> Tuple[TensorType, TensorType, TensorType, TensorType, Dict]:
        # todo: add deterministic or stochastic mode
        """
        Steps the model environment with the given batch of actions.

        Args:
            actions (torch.Tensor or np.ndarray): the actions for each "episode" to rollout.
                Shape must be ``B x A``, where ``B`` is the batch size (i.e., number of episodes),
                and ``A`` is the action dimension. Note that ``B`` must correspond to the
                batch size used when calling :meth:`reset`. If a np.ndarray is given, it's
                converted to a torch.Tensor and sent to the model device.

        Returns:
            (tuple): contains the predicted next observation, reward, termination, truncation flags and metadata.
            The done flag is computed using the termination_fn passed in the constructor.
        """
        # prepare action to batch
        assert (
            actions.ndim == 1 or actions.ndim == 2
        ), "actions must be 1d or batch of 1d actions"
        if actions.ndim == 1:
            return_batch = False
            actions = actions[None]
        else:
            return_batch = True
        assert len(actions.shape) == 2  # batch, action_dim
        assert (
            actions.shape[0] == self._current_obs.shape[0]
        ), "actions must have the same batch size of the current state"
        assert (
            actions.shape[1] == self.system.input_domain.dimension
        ), "actions must match the dimension of the input domain"

        # todo: do we want to differentiate through the dynamics or not?
        with torch.no_grad():
            # if actions is tensor, code assumes it's already on self.device
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions)  # .to(self.device)

            # step
            next_observs = self.system.f(v=self._current_obs, u=actions)
            rewards = self.reward_fn(actions, next_observs)
            terminations = self.termination_fn(actions, next_observs)
            truncations = torch.zeros_like(terminations, dtype=torch.bool)
            infos = {}

            # update state
            self._current_obs = next_observs

            # eventually convert to np
            if self._return_as_np:
                next_observs = next_observs.cpu().numpy()
                rewards = rewards.cpu().numpy()
                terminations = terminations.cpu().numpy()
                truncations = truncations.cpu().numpy()

        if not return_batch:
            next_observs = next_observs[0]
            rewards = rewards[0]
            terminations = terminations[0]
            truncations = truncations[0]

        return next_observs, rewards, terminations, truncations, infos

    def render(self, mode="human"):
        pass

    def evaluate_action_sequences(
        self,
        action_sequences: torch.Tensor,
        initial_state: np.ndarray,
        num_particles: int,
    ) -> torch.Tensor:
        """Evaluates a batch of action sequences on the model.

        Args:
            action_sequences (torch.Tensor): a batch of action sequences to evaluate.  Shape must
                be ``B x H x A``, where ``B``, ``H``, and ``A`` represent batch size, horizon,
                and action dimension, respectively.
            initial_state (np.ndarray): the initial state for the trajectories.
            num_particles (int): number of times each action sequence is replicated. The final
                value of the sequence will be the average over its particles values.

        Returns:
            (torch.Tensor): the accumulated reward for each action sequence, averaged over its
            particles.
        """
        with torch.no_grad():
            assert len(action_sequences.shape) == 3
            population_size, horizon, action_dim = action_sequences.shape
            # either 1-D state or 3-D pixel observation
            assert initial_state.ndim in (1, 3)
            tiling_shape = (num_particles * population_size,) + tuple(
                [1] * initial_state.ndim
            )
            initial_obs_batch = np.tile(initial_state, tiling_shape).astype(np.float32)
            model_state = self.reset(initial_obs_batch, return_as_np=False)
            batch_size = initial_obs_batch.shape[0]
            total_rewards = torch.zeros(batch_size, 1).to(self.device)
            terminated = torch.zeros(batch_size, 1, dtype=bool).to(self.device)
            for time_step in range(horizon):
                action_for_step = action_sequences[:, time_step, :]
                action_batch = torch.repeat_interleave(
                    action_for_step, num_particles, dim=0
                )
                _, rewards, dones, model_state = self.step(
                    action_batch, model_state, sample=True
                )
                rewards[terminated] = 0
                terminated |= dones
                total_rewards += rewards

            total_rewards = total_rewards.reshape(-1, num_particles)
            return total_rewards.mean(dim=1)
