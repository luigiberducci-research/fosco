from abc import abstractmethod, ABC

import numpy as np

from fosco.config import CegisConfig
from fosco.common.domains import Set
from fosco.systems import ControlAffineDynamics


class Certificate(ABC):
    """
    Abstract class for certificates.
    """

    @abstractmethod
    def __init__(
        self,
        system: ControlAffineDynamics,
        vars: dict[str, list],
        domains: dict[str, Set],
        config: CegisConfig,
        verbose: int = 0,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_constraints(self, **kwargs):
        # update this to return a dictionary of tuples (spec, domain)
        raise NotImplementedError


class TrainableCertificate(Certificate):
    """
    Abstract class for certificates that can be trained.
    """

    @abstractmethod
    def learn(self, **kwargs) -> dict[str, float | np.ndarray]:
        raise NotImplementedError
