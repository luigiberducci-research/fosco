from .consolidator import Consolidator


def make_consolidator(**kwargs) -> Consolidator:
    """
    Factory method for consolidator.
    """
    return Consolidator(**kwargs)
