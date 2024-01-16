import logging
from abc import ABC, abstractmethod

import numpy as np
from matplotlib import pyplot as plt

ImageType = np.ndarray | plt.Figure
VideoType = ImageType | list[ImageType]


class Logger(ABC):
    def __init__(self, config: dict = None, verbose: int = 0):
        self.config = config or {}

    """
    def debug(self, *args, **kwargs):
        self._logger.debug(*args, **kwargs)

    def info(self, *args, **kwargs):
        self._logger.info(*args, **kwargs)

    def warning(self, *args, **kwargs):
        self._logger.warning(*args, **kwargs)

    def error(self, *args, **kwargs):
        self._logger.error(*args, **kwargs)
    """

    @abstractmethod
    def log_scalar(self, tag: str, value: float, step: int):
        raise NotImplementedError

    @abstractmethod
    def log_image(self, tag: str, image: ImageType, step: int):
        raise NotImplementedError

    @abstractmethod
    def log_video(self, tag: str, video: VideoType, step: int):
        raise NotImplementedError
