from abc import abstractmethod
from torch import nn


class TorchSymbolicModule(nn.Module):
    @abstractmethod
    def forward(self, x):
        raise NotImplementedError

    @abstractmethod
    def forward_smt(self, x):
        raise NotImplementedError

    @abstractmethod
    def gradient(self, x):
        raise NotImplementedError

    @abstractmethod
    def gradient_smt(self, x):
        raise NotImplementedError

    @abstractmethod
    def save(self, outdir: str):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def load(logdir: str):
        raise NotImplementedError
