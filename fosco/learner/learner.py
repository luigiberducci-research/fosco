import logging
import pathlib
from abc import abstractmethod, ABC

from torch import nn
from fosco.logger import LOGGING_LEVELS


class LearnerNN(nn.Module, ABC):
    """
    Abstract base class for learners.
    """

    def __init__(self, verbose: int = 0):
        super().__init__()

        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(LOGGING_LEVELS[verbose])
        self._logger.debug("Learner initialized")

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
    def save(self, outdir: str, model_name: str = "model") -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, model_path: str | pathlib.Path) -> None:
        raise NotImplementedError
