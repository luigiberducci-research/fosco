import logging
from abc import abstractmethod, ABC

import numpy as np

from fosco.config import CegisConfig
from fosco.common.domains import Set
from fosco.logger import LOGGING_LEVELS
from fosco.systems import ControlAffineDynamics
from fosco.verifier.types import SYMBOL


class Certificate(ABC):
    """
    Abstract class for certificates.
    """

    def __init__(
        self,
        system: ControlAffineDynamics,
        variables: dict[str, list[SYMBOL]],
        domains: dict[str, Set],
        verbose: int = 0,
    ) -> None:
        self.system = system
        self.variables = variables
        self.domains = domains

        self._assert_state()

        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(LOGGING_LEVELS[verbose])
        self._logger.debug("Certificate initialized")

    @abstractmethod
    def _assert_state(self) -> None:
        raise NotImplementedError("")

    @abstractmethod
    def get_constraints(self, **kwargs):
        # update this to return a dictionary of tuples (spec, domain)
        raise NotImplementedError


class TrainableCertificate(Certificate):
    """
    Abstract class for certificates that can be trained.
    """

    @staticmethod
    @abstractmethod
    def learn(**kwargs) -> dict[str, float | np.ndarray | dict]:
        raise NotImplementedError
