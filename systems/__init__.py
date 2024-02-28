from typing import Type

from systems.system import (
    ControlAffineDynamics,
    UncertainControlAffineDynamics,
    SYSTEM_REGISTRY
)
from systems.single_integrator import SingleIntegrator
from systems.double_integrator import DoubleIntegrator
from systems.unicycle_model import Unicycle
from systems.unicycle_acc_model import UnicycleAcc

from systems.rewards import REWARD_REGISTRY
import gymnasium


def make_system(system_id: str) -> Type[ControlAffineDynamics]:
    """
    Factory function for systems.
    """
    if system_id in system.SYSTEM_REGISTRY:
        return system.SYSTEM_REGISTRY[system_id]
    else:
        raise NotImplementedError(f"System {system_id} not implemented")


# register gym env

for system_id in ["SingleIntegrator", "DoubleIntegrator"]:
    for reward_id in REWARD_REGISTRY:
        env_id = f"{system_id}-{reward_id}-v0"
        sys = make_system(system_id=system_id)()
        rew_fn = REWARD_REGISTRY[reward_id](system=sys)
        gymnasium.register(
            id=env_id,
            entry_point='systems.system_env:SystemEnv',
            kwargs={"system": sys, 'reward_fn': rew_fn, "max_steps": 100}
        )
