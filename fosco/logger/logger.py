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
