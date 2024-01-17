import aim

from logger.logger import Logger, VideoType, ImageType


class AimLogger(Logger):
    def __init__(self, config: dict = None, experiment: str = None):
        super().__init__(config)
        self._run = aim.Run(experiment=experiment)

        self._run["config"] = self.config

    def log_scalar(self, tag: str, value: float, step: int, context: dict = None):
        self._run.track(value, name=tag, step=step, context=context)

    def log_image(self, tag: str, image: ImageType, step: int, context: dict = None):
        aim_image = aim.Figure(image)
        self._run.track(aim_image, name=tag, step=step, context=context)

    def log_video(self, tag: str, video: VideoType, step: int, context: dict = None):
        raise NotImplementedError("Aim does not support videos yet")

    def __close__(self):
        self._run.close()
