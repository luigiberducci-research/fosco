import pathlib
from types import NoneType
from typing import Any, Iterable

import aim
import torch.nn

from fosco.logger.logger import Logger, VideoType, ImageType


class AimLogger(Logger):

    def __init__(self, config: dict = None, experiment: str = None, verbose: int = 0):
        super().__init__(config=config, verbose=verbose)
        self._run = aim.Run(experiment=experiment)

        self._run["config"] = self.config

    def _assert_state(self) -> None:
        for k, v in self.config.items():
            self._assert_supported_types(k, v)

    def _assert_supported_types(self, k: str, v: Any) -> None:
        supported_types = [int, float, str, bool, NoneType]
        if type(v) in supported_types:
            return

        try:
            for vi in v:
                self._assert_supported_types(k, vi)
        except TypeError:
            raise TypeError(f"Unsupported type {type(v)} for key {k}")

    def log_scalar(self, tag: str, value: float, step: int, context: dict = None):
        self._run.track(value, name=tag, step=step, context=context)

    def log_image(self, tag: str, image: ImageType, step: int, context: dict = None):
        aim_image = aim.Figure(image)
        self._run.track(aim_image, name=tag, step=step, context=context)

    def log_video(self, tag: str, video: VideoType, step: int, context: dict = None):
        raise NotImplementedError("Aim does not support videos yet")

    def log_model(self, tag: str, model: torch.nn.Module, step: int, **kwargs):
        model_dir = pathlib.Path(self.config["MODEL_DIR"]) / self.config["EXP_NAME"]
        model_dir.mkdir(exist_ok=True, parents=True)
        model_path = model.save(outdir=model_dir, model_name=f"{tag}_{step}")
        aim_string = aim.Text(str(model_path))
        self._run.track(aim_string, name=tag, step=step, context={"model": True})
        return model_path

    def __close__(self):
        self._run.close()
