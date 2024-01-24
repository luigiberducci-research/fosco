from abc import abstractmethod
from typing import Callable, Type

import torch
from torch import nn

from fosco.common.activations import activation
from fosco.common.consts import ActivationType, TimeDomain
from models.network import TorchMLP
from systems import ControlAffineDynamics
from systems.system import UncertainControlAffineDynamics


class LearnerNN(nn.Module):
    @abstractmethod
    def pretrain(self, **kwargs) -> dict:
        raise NotImplementedError

    @abstractmethod
    def update(self, **kwargs) -> dict:
        raise NotImplementedError


class LearnerCT(LearnerNN):
    """
    Leaner class for continuous time dynamical models.
    Train a network according to the learn_method provided by the certificate.
    """

    def pretrain(self, **kwargs) -> dict:
        pass

    def __init__(
            self,
            state_size,
            learn_method,
            hidden_sizes: tuple[int, ...],
            activation: tuple[ActivationType, ...],
            lr: float,
            weight_decay: float,
            initial_models: dict[str, nn.Module] | None = None,
    ):
        super(LearnerCT, self).__init__()

        # certificate function
        if "net" in initial_models and initial_models["net"] is not None:
            self.net = initial_models["net"]
        else:
            self.net = TorchMLP(
                input_size=state_size,
                output_size=1,
                hidden_sizes=hidden_sizes,
                activation=activation,
            )

        if len(list(self.parameters())) > 0:
            self.optimizer = torch.optim.AdamW(
                params=self.parameters(), lr=lr, weight_decay=weight_decay,
            )
        else:
            self.optimizer = None

        self.learn_method = learn_method

    def update(self, datasets, xdot_func, **kwargs) -> dict:
        output = self.learn_method(self, self.optimizer, datasets, xdot_func)
        return output


class LearnerRobustCT(LearnerCT):
    """
    Learner class for continuous time dynamical models with uncertainty.
    Train two networks, one for the certificate function and one for the compensator.
    """

    def __init__(
            self,
            state_size,
            learn_method,
            hidden_sizes: tuple[int, ...],
            activation: tuple[ActivationType, ...],
            lr: float,
            weight_decay: float,
            initial_models: dict[str, nn.Module] | None = None,
    ):
        super(LearnerRobustCT, self).__init__(
            state_size=state_size,
            learn_method=learn_method,
            hidden_sizes=hidden_sizes,
            activation=activation,
            lr=lr,
            weight_decay=weight_decay,
            initial_models=initial_models
        )

        # compensator for additive state disturbances
        if "xsigma" in initial_models and initial_models["xsigma"] is not None:
            self.xsigma = initial_models["xsigma"]
        else:
            self.xsigma = TorchMLP(
                input_size=state_size,
                hidden_sizes=hidden_sizes,
                activation=activation,
                output_size=1,
                output_activation="relu",
            )

        # overriden optimizer with all module parameters
        if len(list(self.parameters())) > 0:
            self.optimizer = torch.optim.AdamW(
                params=self.parameters(), lr=lr, weight_decay=weight_decay,
            )
        else:
            self.optimizer = None


def make_learner(
    system: ControlAffineDynamics, time_domain: TimeDomain
) -> Type[LearnerNN]:
    if (
        isinstance(system, UncertainControlAffineDynamics)
        and time_domain == TimeDomain.CONTINUOUS
    ):
        return LearnerRobustCT
    elif time_domain == TimeDomain.CONTINUOUS:
        return LearnerCT
    else:
        raise NotImplementedError(
            f"Unsupported learner for system {type(system)} and time domain {time_domain}"
        )
