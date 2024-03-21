import pathlib
from typing import Callable

import torch
from torch import nn

from fosco.common.consts import ActivationType
from fosco.common.timing import timed
from fosco.learner import LearnerNN, make_optimizer
from fosco.models import TorchMLP


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
        verbose: int = 0,
    ):
        super(LearnerCT, self).__init__(verbose=verbose)

        # certificate function
        if (
            initial_models
            and "net" in initial_models
            and initial_models["net"] is not None
        ):
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

    def _assert_state(self) -> None:
        assert isinstance(
            self.net, nn.Module
        ), f"Expected nn.Module, got {type(self.net)}"
        assert isinstance(
            self.optimizers, dict
        ), f"Expected dict, got {type(self.optimizers)}"
        assert all(
            [isinstance(v, torch.optim.Optimizer) for v in self.optimizers.values()]
        ), f"Expected dict of optimizers, got {self.optimizers}"
        assert isinstance(
            self.learn_method, Callable
        ), f"Expected callable, got {self.learn_method}"

    def pretrain(self, **kwargs) -> dict:
        raise NotImplementedError

    @timed
    def update(self, datasets, xdot_func, **kwargs) -> dict:
        output = self.learn_method(self, self.optimizers, datasets, xdot_func)
        return output

    def save(self, outdir: str, model_name: str = "model") -> None:
        net_path = self.net.save(outdir=outdir, model_name=f"{model_name}_barrier")
        self._logger.info(f"Saved learner barrier to {net_path}")

    def load(self, model_path: pathlib.Path) -> None:
        raise NotImplementedError("To be fixed to match the new save method")
        assert isinstance(model_path, str) or isinstance(
            model_path, pathlib.Path
        ), f"wrong path type {type(model_path)}"

        model_path = (
            pathlib.Path(model_path) if isinstance(model_path, str) else model_path
        )
        if not model_path.exists():
            raise FileNotFoundError(f"learner checkpoint not found at {model_path}")
        learner_state = torch.load(model_path)
        self.load_state_dict(learner_state)

        self._logger.info(f"Loaded learner from {model_path}")
