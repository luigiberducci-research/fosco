import pathlib
from abc import abstractmethod
from typing import Callable, Type, Mapping, Any

import torch
from torch import nn

from fosco.common.activations import activation
from fosco.common.consts import ActivationType, TimeDomain
from fosco.common.timing import timed
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

    def pretrain(self, **kwargs) -> dict:
        pass

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

        self.optimizers = {}
        if len(list(self.parameters())) > 0:
            self.optimizers["barrier"] = make_optimizer(
                optimizer, params=self.parameters(), lr=lr, weight_decay=weight_decay
            )

        self.learn_method = learn_method

    @timed
    def update(self, datasets, xdot_func, **kwargs) -> dict:
        output = self.learn_method(self, self.optimizers, datasets, xdot_func)
        return output

    def save(self, model_path: str | pathlib.Path) -> None:
        assert isinstance(model_path, str) or isinstance(model_path, pathlib.Path), f"wrong path type {type(model_path)}"

        model_path = pathlib.Path(model_path) if isinstance(model_path, str) else model_path
        if model_path.is_dir():
            model_path.mkdir(parents=True, exist_ok=True)
            model_path = model_path / "learner.pt"
        else:
            model_dir = model_path.parent
            model_dir.mkdir(parents=True, exist_ok=True)

        learner_state = self.state_dict()
        torch.save(learner_state, model_path)

    def load(self, model_path: pathlib.Path) -> None:
        assert (isinstance(model_path, str) or
                isinstance(model_path, pathlib.Path)), f"wrong path type {type(model_path)}"

        model_path = pathlib.Path(model_path) if isinstance(model_path, str) else model_path
        if not model_path.exists():
            raise FileNotFoundError(f"learner checkpoint not found at {model_path}")
        learner_state = torch.load(model_path)
        self.load_state_dict(learner_state)


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


def make_optimizer(optimizer: str | None, **kwargs) -> torch.optim.Optimizer:
    if optimizer is None or optimizer == "adam":
        return torch.optim.Adam(**kwargs)
    elif optimizer == "sgd":
        return torch.optim.SGD(**kwargs)
    else:
        raise NotImplementedError(f"Optimizer {optimizer} not implemented")
