import logging
from abc import ABC, abstractmethod

import numpy as np
from matplotlib import pyplot as plt

ImageType = np.ndarray | plt.Figure
VideoType = ImageType | list[ImageType]


class Logger(ABC):
    def __init__(self, config: dict = None, verbose: int = 0):
        self.config = config or {}

        self.verbose = min(max(verbose, 0), 2)
        levels = [logging.WARNING, logging.INFO, logging.DEBUG]
        logging.basicConfig(level=levels[self.verbose])

    def debug(self, *args, **kwargs):
        logging.debug(*args, **kwargs)

    def info(self, *args, **kwargs):
        logging.info(*args, **kwargs)

    def warning(self, *args, **kwargs):
        logging.warning(*args, **kwargs)

    def error(self, *args, **kwargs):
        logging.error(*args, **kwargs)

    @abstractmethod
    def log_scalar(self, tag: str, value: float, step: int):
        raise NotImplementedError

    @abstractmethod
    def log_image(self, tag: str, image: ImageType, step: int):
        raise NotImplementedError

    @abstractmethod
    def log_video(self, tag: str, video: VideoType, step: int):
        raise NotImplementedError
