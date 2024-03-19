import logging
from abc import ABC, abstractmethod

import numpy as np
import torch
from matplotlib import pyplot as plt

ImageType = np.ndarray | plt.Figure
VideoType = ImageType | list[ImageType]


class Logger(ABC):
    def __init__(self, config: dict = None, verbose: int = 0):
        self.config = config or {}

        self._assert_state()

        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(verbose)
        self._logger.debug("Logger initialized")

    @abstractmethod
    def _assert_state(self) -> None:
        raise NotImplementedError("")

    @abstractmethod
    def log_scalar(self, tag: str, value: float, step: int, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def log_image(self, tag: str, image: ImageType, step: int, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def log_video(self, tag: str, video: VideoType, step: int, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def log_model(self, tag: str, model: torch.nn.Module, step: int, **kwargs):
        raise NotImplementedError
