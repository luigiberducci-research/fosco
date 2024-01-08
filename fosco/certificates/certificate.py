from abc import abstractmethod, ABC

class Certificate(ABC):
    """
    Abstract class for certificates.
    """

    @abstractmethod
    def learn(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_constraints(self, **kwargs):
        raise NotImplementedError