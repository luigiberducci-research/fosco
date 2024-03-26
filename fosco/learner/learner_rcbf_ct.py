import torch.nn
from torch import nn

from fosco.common.consts import ActivationType
from fosco.learner import make_optimizer
from fosco.learner.learner_cbf_ct import LearnerCT
from fosco.models import TorchMLP
from fosco.models.network import SequentialTorchMLP


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
        verbose: int = 0,
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
            verbose=verbose,
        )

        # compensator for additive state disturbances
        if "xsigma" in initial_models and initial_models["xsigma"] is not None:
            self.xsigma = initial_models["xsigma"]
        else:
            # we design the compensator to depend on the barrier function
            # xsigma(x) = xsigma(h(x))
            head_mlp = TorchMLP(
                input_size=1,
                hidden_sizes=(2,),
                output_size=1,
                activation=("tanh",),
                output_activation="relu"
            )
            self.xsigma = SequentialTorchMLP(
                mlps=[self.net, head_mlp],
                register_module=[False, True],
            )

        # overriden optimizer with all module parameters
        if len(list(self.xsigma.parameters())) > 0:
            # self.optimizers["barrier"] = make_optimizer(
            #    optimizer, params=self.parameters(), lr=lr, weight_decay=weight_decay
            # )
            self.optimizers["xsigma"] = make_optimizer(
                optimizer,
                params=self.xsigma.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )

    def _assert_state(self) -> None:
        super()._assert_state()
        assert isinstance(
            self.xsigma, nn.Module
        ), f"Expected nn.Module, got {type(self.xsigma)}"

    def save(self, outdir: str, model_name: str = "model") -> None:
        super().save(outdir, model_name)
        sigma_path = self.xsigma.save(outdir=outdir, model_name=f"{model_name}_sigma")
        self._logger.info(f"Saved learner sigma to {sigma_path}")

    def load(self, model_path: str) -> None:
        raise NotImplementedError("To be fixed to match the new save method")
