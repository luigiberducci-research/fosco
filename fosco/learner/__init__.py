from typing import Type

import torch

from fosco.common.consts import TimeDomain
from fosco.learner.learner import LearnerNN
from fosco.systems import ControlAffineDynamics, UncertainControlAffineDynamics


def make_learner(
    system: ControlAffineDynamics, time_domain: TimeDomain | str
) -> Type[LearnerNN]:
    if isinstance(time_domain, str):
        time_domain = TimeDomain[time_domain.upper()]

    if (
        isinstance(system, UncertainControlAffineDynamics)
        and time_domain == TimeDomain.CONTINUOUS
    ):
        from fosco.learner.learner_rcbf_ct import LearnerRobustCT

        return LearnerRobustCT
    elif time_domain == TimeDomain.CONTINUOUS:
        from fosco.learner.learner_cbf_ct import LearnerCT

        return LearnerCT
    else:
        raise NotImplementedError(
            f"Unsupported learner for system {type(system)} and time domain {time_domain}"
        )


def make_optimizer(optimizer: str | None, **kwargs) -> torch.optim.Optimizer:
    if optimizer is None or optimizer == "adam":
        return torch.optim.Adam(**kwargs)
    elif optimizer == "sgd":
        return torch.optim.SGD(**kwargs)
    else:
        raise NotImplementedError(f"Optimizer {optimizer} not implemented")
