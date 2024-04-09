from abc import abstractmethod, ABC
from typing import Iterable

import torch
from torch import nn

from fosco.verifier.verifier import SYMBOL


class TorchSymFn(nn.Module, ABC):
    @property
    @abstractmethod
    def input_size(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward_smt(
        self, x: Iterable[SYMBOL]
    ) -> tuple[SYMBOL, Iterable[SYMBOL], list[SYMBOL]]:
        raise NotImplementedError


class TorchSymDiffFn(TorchSymFn, ABC):
    @abstractmethod
    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def gradient_smt(
        self, x: Iterable[SYMBOL]
    ) -> tuple[Iterable[SYMBOL], Iterable[SYMBOL], list[SYMBOL]]:
        raise NotImplementedError


class TorchSymModel(TorchSymFn, ABC):
    @abstractmethod
    def save(self, outdir: str):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def load(logdir: str):
        raise NotImplementedError


class TorchSymDiffModel(TorchSymDiffFn, ABC):
    @abstractmethod
    def save(self, outdir: str):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def load(logdir: str):
        raise NotImplementedError
