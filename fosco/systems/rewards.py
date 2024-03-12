from typing import Callable

import numpy as np
import torch

from fosco.common.domains import Sphere, Rectangle
from fosco.systems import ControlAffineDynamics

RewardFnType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

REWARD_REGISTRY = {}


def register(cls):
    """
    Decorator to register a reward class in the reward registry.
    """
    REWARD_REGISTRY[cls.__name__] = cls
    return cls


@register
class GoToUnsafeReward(RewardFnType):
    def __init__(self, system: ControlAffineDynamics) -> None:
        unsafe_domain = system.unsafe_domain

        if isinstance(unsafe_domain, Sphere):
            self.goal_x = np.array(unsafe_domain.center)
        elif isinstance(unsafe_domain, Rectangle):
            delta = np.array(unsafe_domain.upper_bounds) - np.array(unsafe_domain.lower_bounds)
            self.goal_x = np.array(unsafe_domain.lower_bounds) + delta / 2.0
        else:
            raise NotImplementedError(
                f"GoToUnsafeReward not implemented for unsafe set of type {type(unsafe_domain)}"
            )

    def __call__(self, actions: torch.Tensor, next_x: torch.Tensor) -> torch.Tensor:
        distances = torch.linalg.norm(next_x - self.goal_x, axis=1)
        rewards = - distances
        assert rewards.shape[0] == actions.shape[0]
        return rewards
