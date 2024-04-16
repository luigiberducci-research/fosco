import logging
import pathlib
from abc import abstractmethod, ABC
from typing import Callable

from torch import nn
from fosco.logger import LOGGING_LEVELS


class LearnerNN(nn.Module, ABC):
    """
    Abstract base class for learners.
    """

    def __init__(self, verbose: int = 0, **kwargs):
        super().__init__()

        self._learn_method = None

        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(LOGGING_LEVELS[verbose])
        self._logger.debug("Learner initialized")

    @property
    def learn_method(self) -> Callable:
        return self._learn_method

    @learn_method.setter
    def learn_method(self, learn_fn: Callable) -> None:
        self._learn_method = learn_fn

    @abstractmethod
    def _assert_state(self) -> None:
        raise NotImplementedError("")

    @abstractmethod
    def pretrain(self, **kwargs) -> dict:
        raise NotImplementedError

    @abstractmethod
    def update(self, **kwargs) -> dict:
        raise NotImplementedError

    @abstractmethod
    def save(self, outdir: str, model_name: str = "model") -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def load(config_path: str | pathlib.Path):
        raise NotImplementedError
