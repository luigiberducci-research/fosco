from typing import Iterable

import numpy as np
import torch
from torch import nn

from fosco.common.activations import activation
from fosco.common.activations_symbolic import activation_sym, activation_der_sym
from fosco.common.consts import ActivationType
from fosco.verifier import SYMBOL
from models.torchsym import TorchSymDiffModel


class TorchMLP(TorchSymDiffModel):

    def __init__(
        self,
        input_size: int,
        hidden_sizes: tuple[int, ...],
        activation: tuple[str | ActivationType, ...],
        output_size: int = 1,
        output_activation: str | ActivationType = "linear",
    ):
        super(TorchMLP, self).__init__()
        assert len(hidden_sizes) == len(
            activation
        ), "hidden sizes and activation must have the same length"

        self.input_size = input_size
        self.output_size = output_size
        self.layers = []

        # activations
        self.acts = []
        for act in activation:
            act = ActivationType[act.upper()] if isinstance(act, str) else act
            self.acts.append(act)

        # hidden layers
        n_prev, k = self.input_size, 1
        for n_hid in hidden_sizes:
            layer = nn.Linear(n_prev, n_hid)
            self.register_parameter(f"W{k}", layer.weight)
            self.register_parameter(f"b{k}", layer.bias)
            self.layers.append(layer)
            n_prev = n_hid
            k = k + 1

        # last layer
        layer = nn.Linear(n_prev, self.output_size)
        self.register_parameter(f"W{k}", layer.weight)
        self.register_parameter(f"b{k}", layer.bias)
        self.layers.append(layer)

        act = (
            ActivationType[output_activation.upper()]
            if isinstance(output_activation, str)
            else output_activation
        )
        self.acts.append(act)

        assert len(self.layers) == len(
            self.acts
        ), "layers and activations must have the same length"
        assert (
            self.output_size == self.layers[-1].out_features
        ), "output size does not match last layer size"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for idx, layer in enumerate(self.layers):
            z = layer(y)
            y = activation(self.acts[idx], z)

        return y

    def forward_smt(self, x: Iterable[SYMBOL]) -> tuple[SYMBOL, Iterable[SYMBOL]]:
        input_vars = np.array(x).reshape(-1, 1)

        z, _ = network_until_last_layer(net=self, input_vars=input_vars)

        last_layer = self.layers[-1].weight.data.numpy()
        z = last_layer @ z
        if self.layers[-1].bias is not None:
            z += self.layers[-1].bias.data.numpy()[:, None]

        assert z.shape == (1, 1), f"Wrong shape of z, expected (1, 1), got {z.shape}"

        # last activation
        z = activation_sym(self.acts[-1], z)

        return z[0, 0], []

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        x_clone = torch.clone(x).requires_grad_()
        y = self(x_clone)
        dydx = torch.autograd.grad(
            outputs=y,
            inputs=x_clone,
            grad_outputs=torch.ones_like(y),
            create_graph=True,
            retain_graph=True,
        )[0]
        return dydx

    def gradient_smt(self, x: Iterable[SYMBOL]) -> tuple[Iterable[SYMBOL], Iterable[SYMBOL]]:
        input_vars = np.array(x).reshape(-1, 1)

        z, jacobian = network_until_last_layer(net=self, input_vars=input_vars)

        last_layer = self.layers[-1].weight.data.numpy()


        zhat = last_layer @ z
        if self.layers[-1].bias is not None:
            zhat += self.layers[-1].bias.data.numpy()[:, None]

        # last activation
        z = activation_sym(self.acts[-1], zhat)

        jacobian = last_layer @ jacobian
        jacobian = np.diagflat(activation_der_sym(self.acts[-1], zhat)) @ jacobian

        gradV = jacobian

        assert z.shape == (1, 1)
        assert gradV.shape == (
            1,
            self.input_size,
        ), f"Wrong shape of gradV, expected (1, {self.input_size}), got {gradV.shape}"

        return gradV, []

    def save(self, outdir: str):
        import pathlib
        import yaml

        outdir = pathlib.Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        # save model.pt
        torch.save(self.state_dict(), outdir / "model.pt")

        # save params.yaml with net configuration
        params = {
            "input_size": self.input_size,
            "hidden_sizes": [layer.out_features for layer in self.layers[:-1]],
            "activation": [act.name for act in self.acts[:-1]],
            "output_size": self.layers[-1].out_features,
            "output_activation": self.acts[-1].name,
        }
        with open(outdir / "params.yaml", "w") as f:
            yaml.dump(params, f)

    @staticmethod
    def load(logdir: str):
        import pathlib
        import yaml

        logdir = pathlib.Path(logdir)
        assert logdir.exists(), f"directory {logdir} does not exist"

        # load params.yaml
        with open(logdir / "params.yaml", "r") as f:
            params = yaml.load(f, Loader=yaml.FullLoader)

        # load model.pt
        model = TorchMLP(**params)
        model.load_state_dict(torch.load(logdir / "model.pt"))
        return model


def network_until_last_layer(
        net: TorchMLP, input_vars: Iterable[SYMBOL], round: int = -1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Utility for symbolic forward pass excluding the last layer.

    :param net: network model
    :param input_vars: list of symbolic variables
    :return: tuple (net output, its jacobian)
    """
    z = input_vars
    jacobian = np.eye(net.input_size, net.input_size)

    for idx, layer in enumerate(net.layers[:-1]):
        w = layer.weight.data.numpy()
        if layer.bias is not None:
            b = layer.bias.data.numpy()[:, None]
        else:
            b = np.zeros((layer.out_features, 1))


        zhat = w @ z + b
        z = activation_sym(net.acts[idx], zhat)

        jacobian = w @ jacobian
        jacobian = np.diagflat(activation_der_sym(net.acts[idx], zhat)) @ jacobian

    return z, jacobian