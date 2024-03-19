import pathlib
from collections import namedtuple
from dataclasses import dataclass, asdict
from typing import Type, Any, Iterable

CegisResult = namedtuple(
    "CegisResult",
    ["found", "net", "infos"]
)


@dataclass
class CegisConfig:
    # system
    TIME_DOMAIN: str = "continuous"
    # fosco
    CERTIFICATE: str = "CBF"
    VERIFIER: str = "Z3"
    VERIFIER_TIMEOUT: int = 30
    RESAMPLING_N: int = 20
    RESAMPLING_STDDEV: float = 5e-3
    CEGIS_MAX_ITERS: int = 10
    ROUNDING: int = 3
    USE_INIT_MODELS: bool = False
    # training
    N_DATA: int = 500
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    # net architecture
    N_HIDDEN_NEURONS: Iterable[int] = (10,)
    ACTIVATION: Iterable[str] = ("relu",)
    # loss
    OPTIMIZER: str = "adam"
    LOSS_MARGINS: dict[str, float] | float = 0.0
    LOSS_WEIGHTS: dict[str, float] | float = 1.0
    LOSS_RELU: str = "relu"
    N_EPOCHS: int = 100
    # seeding
    SEED: int = None
    # logging
    MODEL_DIR: str = "logs/models"
    LOGGER: str = None
    EXP_NAME: str = None

    def __getitem__(self, item):
        return getattr(self, item)

    def dict(self):
        self.MODEL_DIR = str(pathlib.Path(self.MODEL_DIR).absolute())
        return {k: v for k, v in asdict(self).items()}
