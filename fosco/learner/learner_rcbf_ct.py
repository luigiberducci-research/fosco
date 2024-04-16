import pathlib
from typing import Optional

import yaml
from torch import nn

from fosco.certificates.rcbf import TrainableRCBF
from fosco.common.consts import ActivationType
from fosco.learner import make_optimizer
from fosco.learner.learner_cbf_ct import LearnerCBF
from fosco.models.network import SequentialTorchMLP, RobustGate, TorchMLP


class LearnerRobustCBF(LearnerCBF):
    """
    Learner class for continuous time dynamical models with uncertainty.
    Train two networks, one for the certificate function and one for the compensator.
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
        super(LearnerRobustCBF, self).__init__(
            state_size=state_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
            optimizer=optimizer,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            loss_margins=loss_margins,
            loss_weights=loss_weights,
            loss_relu=loss_relu,
            initial_models=initial_models,
            verbose=verbose,
        )

        # compensator for additive state disturbances
        if initial_models and "xsigma" in initial_models and initial_models["xsigma"] is not None:
            self.xsigma = initial_models["xsigma"]
        else:
            # we design the compensator to depend on the barrier function
            # xsigma(x) = xsigma(h(x))
            head_model = RobustGate(
                activation_type="hsigmoid"
            )
            self.xsigma = SequentialTorchMLP(
                mlps=[self.net, head_model],
                register_module=[False, True],
            )

        # override optimizer with all module parameters
        if len(list(self.xsigma.parameters())) > 0:
            self.optimizers["xsigma"] = make_optimizer(
                optimizer,
                params=self.xsigma.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )

        # add extra loss margin for uncertainty loss
        self.loss_keys = self.loss_keys + ["robust", "conservative_sigma"]
        for loss_k in ["robust", "conservative_sigma"]:
            if isinstance(loss_margins, float):
                self.loss_margins[loss_k] = loss_margins
            else:
                assert loss_k in loss_margins, f"Missing loss margin {loss_k}, got {loss_margins}"
                self.loss_margins[loss_k] = loss_margins[loss_k]

            # add extra loss weight for uncertainty loss
            if isinstance(loss_weights, float):
                self.loss_weights[loss_k] = loss_weights
            else:
                assert loss_k in loss_weights, f"Missing loss weight {loss_k}, got {loss_weights}"
                self.loss_weights[loss_k] = loss_weights[loss_k]

        self.learn_method = TrainableRCBF.learn

    def _assert_state(self) -> None:
        super()._assert_state()
        assert isinstance(
            self.xsigma, nn.Module
        ), f"Expected nn.Module, got {type(self.xsigma)}"

    def save(self, outdir: str, model_name: str = "model") -> None:
        param_path = super().save(outdir=outdir, model_name=model_name)

        if not isinstance(self.xsigma, TorchMLP) and not isinstance(self.xsigma, SequentialTorchMLP):
            raise ValueError(f"Saving model supported only for TorchMLP, got sigma {type(self.xsigma)}")
        xsigma_path = self.xsigma.save(outdir=outdir, model_name=f"{model_name}_compensator")

        return param_path

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
                params["module"] == "fosco.learner.learner_rcbf_ct"
        ), f"Expected fosco.learner.learner_rcbf_ct, got {params['module']}"
        assert (
                params["class"] == "LearnerRobustCBF"
        ), f"Expected LearnerRobustCBF, got {params['class']}"

        kwargs = params["kwargs"]
        learner = LearnerRobustCBF(**kwargs)

        # load models
        learner.net = learner.net.load(config_path.parent / f"{config_path.stem}_barrier.yaml")
        learner.xsigma = learner.xsigma.load(config_path.parent / f"{config_path.stem}_compensator.yaml")

        return learner
