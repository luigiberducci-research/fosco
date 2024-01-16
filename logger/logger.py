from abc import ABC, abstractmethod

import numpy as np
from matplotlib import pyplot as plt

ImageType = np.ndarray | plt.Figure
VideoType = ImageType | list[ImageType]


class Logger(ABC):
    def __init__(self, config: dict = None):
        self.config = config or {}

    @abstractmethod
    def log_scalar(self, tag: str, value: float, step: int):
        raise NotImplementedError

    @abstractmethod
    def log_image(self, tag: str, image: ImageType, step: int):
        raise NotImplementedError

    @abstractmethod
    def log_video(self, tag: str, video: VideoType, step: int):
        raise NotImplementedError
