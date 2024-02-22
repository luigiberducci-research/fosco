from abc import ABC, abstractmethod

import torch


class RLTrainer(ABC):
    @abstractmethod
    def train(
            self,
            obs: torch.Tensor,
            logprobs: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            dones: torch.Tensor,
            next_obs: torch.Tensor,
            next_done: torch.Tensor,
            values: torch.Tensor
    ) -> dict[str, float]:
        pass

    @abstractmethod
    def get_actor(self) -> torch.nn.Module:
        pass
