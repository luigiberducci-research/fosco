from .additive_bounded import AdditiveBounded
from .convex_hull import ConvexHull
from .parametric_uncertainty import ParametricUncertainty
from .uncertainty_wrapper import UNCERTAINTY_REGISTRY

def add_uncertainty(uncertainty_type: str | None, system_fn, **kwargs) -> callable:
    if uncertainty_type is None:
        return system_fn
    if uncertainty_type in UNCERTAINTY_REGISTRY:
        return lambda: UNCERTAINTY_REGISTRY[uncertainty_type](system=system_fn(), **kwargs)
    else:
        raise NotImplementedError(f"Uncertainty {uncertainty_type} not implemented")
