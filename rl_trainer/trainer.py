from abc import ABC, abstractmethod
from typing import Optional

import gymnasium
import torch
from torch.utils.tensorboard import SummaryWriter


class RLTrainer(ABC):
    @abstractmethod
    def train(
        self, envs: gymnasium.Env, writer: Optional[SummaryWriter], verbose: int = 1,
    ) -> dict[str, float]:
        pass

    @abstractmethod
    def get_actor(self) -> torch.nn.Module:
        pass
