from .additive_bounded import AdditiveBounded
from .uncertainty_wrapper import UNCERTAINTY_REGISTRY

def add_uncertainty(uncertainty_type: str | None, system_fn: callable) -> callable:
    if uncertainty_type is None:
        return system_fn
    if uncertainty_type in UNCERTAINTY_REGISTRY:
        return lambda: UNCERTAINTY_REGISTRY[uncertainty_type](system=system_fn())
    else:
        raise NotImplementedError(f"Uncertainty {uncertainty_type} not implemented")