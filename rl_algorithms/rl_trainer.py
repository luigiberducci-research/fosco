from abc import ABC, abstractmethod
from typing import Optional

import torch


class RLTrainer(ABC):
    @abstractmethod
    def train(
            self,
            obs: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            dones: torch.Tensor,
            next_obs: torch.Tensor,
            next_done: Optional[torch.Tensor] = None,
            values: Optional[torch.Tensor] = None,
            logprobs: Optional[torch.Tensor] = None,
            global_step: Optional[int] = None,
    ) -> dict[str, float]:
        pass

    @abstractmethod
    def get_actor(self) -> torch.nn.Module:
        pass
