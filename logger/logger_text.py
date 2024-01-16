from logger import Logger
from logger.logger import ImageType


class TextLogger(Logger):
    def __init__(self, config: dict = None):
        super().__init__(config)

    def log_scalar(self, tag: str, value: float, step: int):
        self.info(msg=f"step: {step}, tag: {tag}, value: {value}")

    def log_image(self, tag: str, image: ImageType, step: int):
        self.warning(msg=f"step: {step}, tag: {tag}, log image not supported")

    def log_video(self, tag: str, image: ImageType, step: int):
        self.warning(msg=f"step: {step}, tag: {tag}, log video not supported")