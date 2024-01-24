from fosco.logger import Logger
from fosco.logger.logger import ImageType
import logging


class TextLogger(Logger):
    def __init__(self, config: dict = None, **kwargs):
        super().__init__(config)

        self._logger = logging.getLogger(__name__)

    def log_scalar(self, tag: str, value: float, step: int, **kwargs):
        self._logger.info(msg=f"step: {step}, tag: {tag}, value: {value}")

    def log_image(self, tag: str, image: ImageType, step: int, **kwargs):
        self._logger.warning(msg=f"step: {step}, tag: {tag}, log image not supported")

    def log_video(self, tag: str, image: ImageType, step: int, **kwargs):
        self._logger.warning(msg=f"step: {step}, tag: {tag}, log video not supported")
