from collections import namedtuple
from dataclasses import dataclass, asdict
from typing import Type, Any

from fosco.common.consts import (
    TimeDomain,
    CertificateType,
    VerifierType,
    ActivationType,
    LossReLUType,
)
from fosco.logger import LoggerType
from systems import ControlAffineControllableDynamicalModel

CegisResult = namedtuple("CegisResult", ["found", "net", "infos"])


@dataclass
class CegisConfig:
    # system
    SYSTEM: Type[ControlAffineControllableDynamicalModel] = None
    DOMAINS: dict[str, Any] = None
    TIME_DOMAIN: TimeDomain = TimeDomain.CONTINUOUS
    # fosco
    CERTIFICATE: CertificateType = CertificateType.CBF
    VERIFIER: VerifierType = VerifierType.Z3
    CEGIS_MAX_ITERS: int = 10
    ROUNDING: int = 3
    # training
    DATA_GEN: dict[str, callable] = None
    N_DATA: int = 500
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    # net architecture
    N_HIDDEN_NEURONS: tuple[int, ...] = (10,)
    ACTIVATION: tuple[ActivationType, ...] = (ActivationType.SQUARE,)
    # loss
    LOSS_MARGINS: dict[str, float] | float = 0.0
    LOSS_WEIGHTS: dict[str, float] | float = 1.0
    LOSS_RELU: LossReLUType = LossReLUType.RELU
    LOSS_NETGRAD_WEIGHT: float = 0.0
    N_EPOCHS: int = 100
    # seeding
    SEED: int = None
    # logging
    LOGGER: LoggerType = None

    def __getitem__(self, item):
        return getattr(self, item)

    def dict(self):
        return {k: str(v) for k, v in asdict(self).items()}
