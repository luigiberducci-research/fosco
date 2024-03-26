from typing import Type

import torch

from fosco.common.consts import TimeDomain
from fosco.learner.learner import LearnerNN
from fosco.systems import ControlAffineDynamics, UncertainControlAffineDynamics


def make_learner(
    system: ControlAffineDynamics
) -> Type[LearnerNN]:

    if isinstance(system, UncertainControlAffineDynamics):
        from fosco.learner.learner_rcbf_ct import LearnerRobustCT

        return LearnerRobustCT
    else:
        from fosco.learner.learner_cbf_ct import LearnerCT
        return LearnerCT


def make_optimizer(optimizer: str | None, **kwargs) -> torch.optim.Optimizer:
    if optimizer is None or optimizer == "adam":
        return torch.optim.Adam(**kwargs)
    elif optimizer == "sgd":
        return torch.optim.SGD(**kwargs)
    else:
        raise NotImplementedError(f"Optimizer {optimizer} not implemented")
