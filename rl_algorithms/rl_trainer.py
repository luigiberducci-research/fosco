from abc import ABC, abstractmethod


class RLTrainer(ABC):
    @abstractmethod
    def train(self) -> None:
        pass
