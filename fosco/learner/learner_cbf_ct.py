import pathlib
from typing import Callable, Optional

import torch
import yaml
from torch import nn

from fosco.certificates.cbf import TrainableCBF
from fosco.common.consts import ActivationType, DomainName, LossReLUType
from fosco.common.timing import timed
from fosco.learner import LearnerNN, make_optimizer
from fosco.models import TorchMLP, SequentialTorchMLP


class LearnerCBF(LearnerNN):
    """
    Leaner class for continuous time dynamical models.
    Train a network according to the learn_method provided by the certificate.
    """

    def __init__(
        self,
        state_size: int,
        hidden_sizes: tuple[int, ...],
        activation: tuple[ActivationType, ...],
        epochs: int,
        lr: float,
        weight_decay: float,
        loss_margins: dict[str, float] | float,
        loss_weights: dict[str, float] | float,
        loss_relu: str,
        optimizer: Optional[str] = None,
        initial_models: Optional[dict[str, nn.Module]] = None,
        verbose: int = 0,
    ):
        super(LearnerCBF, self).__init__(verbose=verbose)

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

        self.optimizer_type = optimizer
        self.optimizer_lr = lr
        self.optimizer_wd = weight_decay

        self.optimizers = {}
        if len(list(self.parameters())) > 0:
            self.optimizers["barrier"] = make_optimizer(
                optimizer,
                params=self.parameters(),
                lr=self.optimizer_lr,
                weight_decay=self.optimizer_wd,
            )

        # loss parameters
        self.loss_keys = [
            DomainName.XI.value,
            DomainName.XU.value,
            DomainName.XD.value,
            "conservative_b",
        ]
        self.loss_relu = LossReLUType[loss_relu.upper()]
        self.epochs = epochs

        # process loss margins
        if isinstance(loss_margins, float):
            self.loss_margins = {k: loss_margins for k in self.loss_keys}
        else:
            self.loss_margins = loss_margins

        # process loss weights
        if isinstance(loss_weights, float):
            self.loss_weights = {k: loss_weights for k in self.loss_keys}
        else:
            self.loss_weights = loss_weights

        self.learn_method = TrainableCBF.learn

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

        assert isinstance(
            self.loss_relu, LossReLUType
        ), f"Expected LossReLUType, got {type(self.loss_relu)}"
        assert (
            isinstance(self.epochs, int) and self.epochs >= 0
        ), f"Expected non-neg int for epochs, got {self.epochs}"

        assert all(
            [k in self.loss_margins for k in self.loss_keys]
        ), f"Missing loss margin for any {self.loss_keys}, got {self.loss_margins}"

        assert all(
            [k in self.loss_weights for k in self.loss_keys]
        ), f"Missing loss weight for any {self.loss_keys}, got {self.loss_weights}"

    def pretrain(self, **kwargs) -> dict:
        raise NotImplementedError

    @timed
    def update(self, datasets, xdot_func, **kwargs) -> dict:
        output = self.learn_method(
            learner=self,
            optimizers=self.optimizers,
            datasets=datasets,
        )
        return output

    def save(self, outdir: str, model_name: str = "model") -> None:
        if not isinstance(self.net, TorchMLP) and not isinstance(
            self.net, SequentialTorchMLP
        ):
            raise ValueError(
                f"Saving model supported only for TorchMLP, got {type(self.net)}"
            )

        net_path = self.net.save(outdir=outdir, model_name=f"{model_name}_barrier")

        # save params.yaml with learner configuration
        params = {
            "module": self.__module__,
            "class": self.__class__.__name__,
            "kwargs": {
                "state_size": self.net.input_size,
                "hidden_sizes": [l.out_features for l in self.net.layers[:-1]],
                "activation": [a.value for a in self.net.acts[:-1]],
                "optimizer": self.optimizer_type,
                "epochs": self.epochs,
                "loss_margins": self.loss_margins,
                "loss_weights": self.loss_weights,
                "loss_relu": self.loss_relu.value,
                "lr": self.optimizer_lr,
                "weight_decay": self.optimizer_wd,
            },
        }

        param_path = pathlib.Path(outdir) / f"{model_name}.yaml"
        with open(param_path, "w") as f:
            yaml.dump(params, f)

        return str(param_path)

    @staticmethod
    def load(config_path: str | pathlib.Path):
        config_path = pathlib.Path(config_path)
        assert config_path.exists(), f"config file {config_path} does not exist"
        assert (
            config_path.suffix == ".yaml"
        ), f"expected .yaml file, got {config_path.suffix}"

        # load params.yaml
        with open(config_path, "r") as f:
            params = yaml.load(f, Loader=yaml.FullLoader)

        assert all(
            [k in params for k in ["module", "class", "kwargs"]]
        ), f"Missing keys in {params.keys()}"
        assert (
            params["module"] == "fosco.learner.learner_cbf_ct"
        ), f"Expected fosco.learner.learner_cbf_ct, got {params['module']}"
        assert (
            params["class"] == "LearnerCBF"
        ), f"Expected LearnerCBF, got {params['class']}"

        kwargs = params["kwargs"]
        learner = LearnerCBF(**kwargs)

        # load net
        learner.net = learner.net.load(
            config_path.parent / f"{config_path.stem}_barrier.yaml"
        )

        return learner
