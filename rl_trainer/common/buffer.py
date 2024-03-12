from abc import ABC, abstractmethod

import torch

from fosco.systems.system_env import TensorType


class Buffer(ABC):
    @abstractmethod
    def push(self, **kwargs) -> None:
        pass

    @abstractmethod
    def sample(self, batch_size: int) -> dict[str, TensorType]:
        pass


class CyclicBuffer(Buffer):
    def __init__(
        self,
        capacity: int,
        feature_shapes: dict[str, tuple[int]],
        device: torch.device
    ) -> None:
        self._capacity = capacity
        self._device = device

        self._buffers = {}
        for feature, shape in feature_shapes.items():
            self._buffers[feature] = torch.zeros(shape, dtype=torch.float32).to(device)

        self._step = 0


    def push(self, **kwargs) -> None:
        updated = {k: False for k in self._buffers}
        for feature, batch in kwargs.items():
            if feature not in self._buffers:
                continue
            self._buffers[feature][self._step] = batch
            updated[feature] = True

        # check consistency: all buffers same size
        assert all(updated.values()), f"expected all buffers to get updated, got {updated}"

        self._step = (self._step + 1) % self._capacity

    def sample(self, batch_size: int = None) -> dict[str, TensorType]:
        return self._buffers
