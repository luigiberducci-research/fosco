import logging
import pathlib
from abc import abstractmethod
from typing import Type

import torch
from torch import nn

from fosco.common.activations import activation
from fosco.common.consts import ActivationType, TimeDomain
from fosco.common.timing import timed
from fosco.logger import LOGGING_LEVELS
from fosco.models import TorchMLP
from fosco.systems import ControlAffineDynamics
from fosco.systems import UncertainControlAffineDynamics


class LearnerNN(nn.Module):

    def __init__(self, verbose: int = 0):
        super().__init__()

        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(LOGGING_LEVELS[verbose])
        self._logger.debug("Learner initialized")

    @abstractmethod
    def pretrain(self, **kwargs) -> dict:
        raise NotImplementedError

    @abstractmethod
    def update(self, **kwargs) -> dict:
        raise NotImplementedError

    @abstractmethod
    def save(self, model_path: str | pathlib.Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, model_path: str | pathlib.Path) -> None:
        raise NotImplementedError


class LearnerCT(LearnerNN):
    """
    Leaner class for continuous time dynamical models.
    Train a network according to the learn_method provided by the certificate.
    """

    def __init__(
            self,
            state_size,
            learn_method,
            hidden_sizes: tuple[int, ...],
            activation: tuple[ActivationType, ...],
            optimizer: str | None,
            lr: float,
            weight_decay: float,
            initial_models: dict[str, nn.Module] | None = None,
            verbose: int = 0
    ):
        super(LearnerCT, self).__init__(verbose=verbose)

        # certificate function
        if initial_models and "net" in initial_models and initial_models["net"] is not None:
            self.net = initial_models["net"]
        else:
            self.net = TorchMLP(
                input_size=state_size,
                output_size=1,
                hidden_sizes=hidden_sizes,
                activation=activation,
            )

        self.optimizers = {}
        if len(list(self.parameters())) > 0:
            self.optimizers["barrier"] = make_optimizer(
                optimizer, params=self.parameters(), lr=lr, weight_decay=weight_decay
            )

        self.learn_method = learn_method

    def pretrain(self, **kwargs) -> dict:
        pass

    @timed
    def update(self, datasets, xdot_func, **kwargs) -> dict:
        output = self.learn_method(self, self.optimizers, datasets, xdot_func)
        return output

    def save(self, model_path: str | pathlib.Path) -> None:
        assert isinstance(model_path, str) or isinstance(model_path,
                                                         pathlib.Path), f"wrong path type {type(model_path)}"

        model_path = pathlib.Path(model_path) if isinstance(model_path, str) else model_path
        assert (model_path.is_dir() and model_path.exists()) or (model_path.suffix == ".pt"), f"expected dir or filepath with suffix .pt, got {model_path} with suffix {model_path.suffix}"

        if not model_path.suffix == ".pt":
            model_path = model_path / "learner.pt"
        else:
            model_path.parent.mkdir(parents=True, exist_ok=True)

        learner_state = self.state_dict()
        torch.save(learner_state, model_path)

        self._logger.info(f"Saved learner to {model_path}")

    def load(self, model_path: pathlib.Path) -> None:
        assert (isinstance(model_path, str) or
                isinstance(model_path, pathlib.Path)), f"wrong path type {type(model_path)}"

        model_path = pathlib.Path(model_path) if isinstance(model_path, str) else model_path
        if not model_path.exists():
            raise FileNotFoundError(f"learner checkpoint not found at {model_path}")
        learner_state = torch.load(model_path)
        self.load_state_dict(learner_state)

        self._logger.info(f"Loaded learner from {model_path}")


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
            optimizer: str | None,
            lr: float,
            weight_decay: float,
            initial_models: dict[str, nn.Module] | None = None,
            verbose: int = 0
    ):
        super(LearnerRobustCT, self).__init__(
            state_size=state_size,
            learn_method=learn_method,
            hidden_sizes=hidden_sizes,
            activation=activation,
            optimizer=optimizer,
            lr=lr,
            weight_decay=weight_decay,
            initial_models=initial_models,
            verbose=verbose
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
        if len(list(self.xsigma.parameters())) > 0:
            self.optimizers["barrier"] = make_optimizer(
                optimizer, params=self.parameters(), lr=lr, weight_decay=weight_decay
            )


def make_learner(
        system: ControlAffineDynamics, time_domain: TimeDomain | str
) -> Type[LearnerNN]:
    if isinstance(time_domain, str):
        time_domain = TimeDomain[time_domain]

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


def make_optimizer(optimizer: str | None, **kwargs) -> torch.optim.Optimizer:
    if optimizer is None or optimizer == "adam":
        return torch.optim.Adam(**kwargs)
    elif optimizer == "sgd":
        return torch.optim.SGD(**kwargs)
    else:
        raise NotImplementedError(f"Optimizer {optimizer} not implemented")
