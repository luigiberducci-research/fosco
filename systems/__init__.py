from typing import Type

from fosco.common import domains
from fosco.common.domains import Set
from .system import (
    ControlAffineDynamics,
    UncertainControlAffineDynamics,
)
from .single_integrator import SingleIntegrator
from .double_integrator import DoubleIntegrator
from .unicycle_model import Unicycle


def make_system(system_id: str) -> Type[ControlAffineDynamics]:
    """
    Factory function for systems.
    """
    if system_id in system.SYSTEM_REGISTRY:
        return system.SYSTEM_REGISTRY[system_id]
    else:
        raise NotImplementedError(f"System {system_id} not implemented")

