from typing import Type

from .system import (
    ControlAffineDynamics,
    UncertainControlAffineDynamics,
)
from .single_integrator import SingleIntegrator
from .double_integrator import DoubleIntegrator
from .system_env import SystemEnv
from .unicycle_model import Unicycle
from .unicycle_acc_model import UnicycleAcc

from .rewards import REWARD_REGISTRY

# register gym env
import gymnasium

for system_id in ["SingleIntegrator", "DoubleIntegrator"]: #system.SYSTEM_REGISTRY:
    for reward_id in REWARD_REGISTRY:
        my_system = system.SYSTEM_REGISTRY[system_id]()
        reward_fn = REWARD_REGISTRY[reward_id](system=my_system)
        entrypoint = lambda: SystemEnv(
            system=my_system,
            reward_fn=reward_fn,
            max_steps=100
        )

        gymnasium.register(id=f"{system_id}-{reward_id}-v0",
                           entry_point=entrypoint)

def make_system(system_id: str) -> Type[ControlAffineDynamics]:
    """
    Factory function for systems.
    """
    if system_id in system.SYSTEM_REGISTRY:
        return system.SYSTEM_REGISTRY[system_id]
    else:
        raise NotImplementedError(f"System {system_id} not implemented")
