from abc import abstractmethod, ABC

from fosco.common.domains import Set


class Certificate(ABC):
    """
    Abstract class for certificates.
    """

    @abstractmethod
    def __init__(self, vars: dict[str, list], domains: dict[str, Set]) -> None:
        raise NotImplementedError

    @abstractmethod
    def learn(self, **kwargs):
        # todo update this to return dictionary
        raise NotImplementedError

    @abstractmethod
    def get_constraints(self, **kwargs):
        # update this to return a dictionary of tuples (spec, domain)
        raise NotImplementedError
