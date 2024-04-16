import sys

import torch

from fosco.logger import Logger
from fosco.logger.logger import ImageType


class TextLogger(Logger):
    def __init__(self, config: dict = None, **kwargs):
        super().__init__(config)

        self._warn_every_k = 1000
        self._counters = {
            "scalar": 0,
            "image": 0,
            "video": 0,
            "model": 0,
        }

    def _assert_state(self) -> None:
        pass

    def log_scalar(self, tag: str, value: float, step: int, **kwargs):
        self._counters["scalar"] += 1
        if self._counters["scalar"] % self._warn_every_k == 0:
            self._logger.warning(msg=f"step: {step}, tag: {tag}, value: {value}")

    def log_image(self, tag: str, image: ImageType, step: int, **kwargs):
        self._counters["image"] += 1
        if self._counters["image"] % self._warn_every_k == 0:
            self._logger.warning(msg=f"step: {step}, tag: {tag}, log image not supported")

    def log_video(self, tag: str, image: ImageType, step: int, **kwargs):
        self._counters["video"] += 1
        if self._counters["video"] % self._warn_every_k == 0:
            self._logger.warning(msg=f"step: {step}, tag: {tag}, log video not supported")

    def log_model(self, tag: str, model: torch.nn.Module, step: int, **kwargs):
        self._counters["model"] += 1
        if self._counters["model"] % self._warn_every_k == 0:
            self._logger.warning(msg=f"step: {step}, tag: {tag}, log model not supported")
