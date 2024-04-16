import logging
from abc import ABC, abstractmethod

from fosco.logger import LOGGING_LEVELS


class Translator(ABC):
    """
    Abstract class for symbolic translators.
    """

    def __init__(self, verbose: int = 0):
        self._assert_state()

        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(LOGGING_LEVELS[verbose])
        self._logger.debug("Translator initialized")

    @abstractmethod
    def _assert_state(self) -> None:
        raise NotImplementedError("")

    @abstractmethod
    def translate(self, **kwargs) -> dict:
        """
        Translate a network forward pass and gradients into a symbolic expression.

        Returns:
            - dictionary of symbolic expressions
        """
        raise NotImplementedError
