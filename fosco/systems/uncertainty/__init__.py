from .additive_bounded import AdditiveBounded
from .convex_hull import ConvexHull
from .uncertainty_wrapper import UNCERTAINTY_REGISTRY
from .. import ControlAffineDynamics


def add_uncertainty(uncertainty_type: str | None, system: ControlAffineDynamics, **kwargs) -> callable:
    if uncertainty_type is None:
        return system
    if uncertainty_type in UNCERTAINTY_REGISTRY:
        return UNCERTAINTY_REGISTRY[uncertainty_type](system=system, **kwargs)
    else:
        raise NotImplementedError(f"Uncertainty {uncertainty_type} not implemented")
