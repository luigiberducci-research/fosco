import pathlib
from typing import Iterable, Optional

import numpy as np
import torch
from torch import nn

from fosco.common.activations import activation
from fosco.common.activations_symbolic import activation_sym, activation_der_sym
from fosco.common.consts import ActivationType
from fosco.verifier.verifier import SYMBOL
from fosco.models.torchsym import TorchSymDiffModel


def make_mlp(
    input_size: int,
    hidden_sizes: tuple[int, ...],
    hidden_activation: tuple[str | ActivationType, ...],
    output_size: int,
    output_activation: str | ActivationType,
):
    """
    Make a multi-layer perceptron model.
    Returns a list of layers and activations.
    """
    assert len(hidden_sizes) == len(
        hidden_activation
    ), "hidden sizes and activation must have the same length"

    layers = []
    acts = []
    n_prev, k = input_size, 1
    for n_hid, act in zip(hidden_sizes, hidden_activation):
        layer = nn.Linear(n_prev, n_hid)
        act_fn = ActivationType[act.upper()] if isinstance(act, str) else act
        acts.append(act_fn)
        layers.append(layer)
        n_prev = n_hid
        k = k + 1

    layer = nn.Linear(n_prev, output_size)
    layers.append(layer)

    act = (
        ActivationType[output_activation.upper()]
        if isinstance(output_activation, str)
        else output_activation
    )
    acts.append(act)

    assert len(layers) == len(acts), "layers and activations must have the same length"
    assert (
        output_size == layers[-1].out_features
    ), "output size does not match last layer size"

    return layers, acts


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

        self.input_size: int = input_size
        self.output_size: int = output_size

        self.layers, self.acts = make_mlp(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            hidden_activation=activation,
            output_size=output_size,
            output_activation=output_activation,
        )

        # register layers
        for idx, layer in enumerate(self.layers):
            self.register_parameter(f"W{idx}", layer.weight)
            self.register_parameter(f"b{idx}", layer.bias)

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

    def forward_smt(
        self, x: Iterable[SYMBOL]
    ) -> tuple[SYMBOL, Iterable[SYMBOL], list[SYMBOL]]:
        input_vars = np.array(x).reshape(-1, 1)

        z, _ = network_until_last_layer(net=self, input_vars=input_vars)

        last_layer = self.layers[-1].weight.data.numpy()
        z = last_layer @ z
        if self.layers[-1].bias is not None:
            z += self.layers[-1].bias.data.numpy()[:, None]

        assert z.shape == (
            self.output_size,
            1,
        ), f"Wrong shape of z, expected ({self.output_size}, 1), got {z.shape}"

        # last activation
        z = activation_sym(self.acts[-1], z)

        # if z is 1d, squeeze it and return symbolic expression
        if isinstance(z, np.ndarray):
            z = z.squeeze()
            if z.shape == ():
                z = z.item()

        var_list = list(set([iv for iv in input_vars.flatten()]))
        return z, [], var_list

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

    def gradient_smt(
        self, x: Iterable[SYMBOL]
    ) -> tuple[Iterable[SYMBOL], Iterable[SYMBOL], list[SYMBOL]]:
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

        var_list = list(set([iv for iv in input_vars.flatten()]))
        return gradV, [], var_list

    def save(self, outdir: str, model_name: str = "model") -> str:
        import pathlib
        import yaml

        outdir = pathlib.Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        # save model.pt
        model_path = outdir / f"{model_name}.pt"
        torch.save(self.state_dict(), model_path)

        # save params.yaml with net configuration
        params = {
            "module": self.__module__,
            "class": self.__class__.__name__,
            "kwargs": {
                "input_size": self.input_size,
                "hidden_sizes": [layer.out_features for layer in self.layers[:-1]],
                "activation": [act.name for act in self.acts[:-1]],
                "output_size": self.layers[-1].out_features,
                "output_activation": self.acts[-1].name,
            },
        }

        param_path = model_path.parent / f"{model_name}.yaml"
        with open(param_path, "w") as f:
            yaml.dump(params, f)

        return str(param_path)

    @staticmethod
    def load(config_path: str | pathlib.Path):
        import pathlib
        import yaml

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
            params["module"] == "fosco.models.network"
        ), f"Expected fosco.models.network, got {params['module']}"
        assert (
            params["class"] == "TorchMLP"
        ), f"Expected TorchMLP, got {params['class']}"

        # load model.pt
        model_path = config_path.parent / f"{config_path.stem}.pt"
        kwargs = params["kwargs"]
        model = TorchMLP(**kwargs)
        model.load_state_dict(torch.load(model_path))

        return model


class SequentialTorchMLP(TorchSymDiffModel):
    def __init__(
        self,
        mlps: list[TorchMLP | pathlib.Path],
        register_module: list[bool] = None,
        model_dir: Optional[str] = None,
    ):
        super(SequentialTorchMLP, self).__init__()

        # load models if paths are given
        for idx, mlp_or_path in enumerate(mlps):
            if isinstance(mlp_or_path, pathlib.Path) or isinstance(mlp_or_path, str):
                assert (
                    model_dir is not None
                ), "model_dir must be given if mlp path is given"
                config_path = pathlib.Path(model_dir) / mlp_or_path
                mlps[idx] = TorchMLP.load(config_path=config_path)

        assert all(
            [isinstance(mlp, TorchMLP) for mlp in mlps]
        ), f"All models must be of type TorchMLP, got {mlps}"

        # check if output size of previous model matches input size of next model
        for idx, mlp in enumerate(mlps[:-1]):
            curr_output_size = mlp.output_size
            next_input_size = mlps[idx + 1].input_size
            assert (
                curr_output_size == next_input_size
            ), f"Output size of model {idx} does not match input size of model {idx + 1}"

        self.mlps = mlps
        self.register_module_bool = register_module or [True] * len(mlps)

        self.input_size: int = mlps[0].input_size
        self.output_size: int = mlps[-1].output_size

        # register models
        for idx, mlp in enumerate(self.mlps):
            if not self.register_module_bool[idx]:
                continue
            self.add_module(f"mlp_{idx}", mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for mlp in self.mlps:
            y = mlp(y)
        return y

    def forward_smt(
        self, x: Iterable[SYMBOL]
    ) -> tuple[SYMBOL, Iterable[SYMBOL], list[SYMBOL]]:
        input_vars = np.array(x).reshape(-1, 1)

        z = input_vars
        z_constraints = []
        # note: we assume the input vars are the one given to the first mlp
        # other mlps do not introduce auxiliary variables
        for mlp in self.mlps:
            z, new_constr, _ = mlp.forward_smt(z)
            z_constraints.extend(new_constr)

        # if z is 1d, squeeze it and return symbolic expression
        if isinstance(z, np.ndarray):
            z = z.squeeze()
            if z.shape == ():
                z = z.item()

        var_list = list(set([iv for iv in input_vars.flatten()]))
        return z, z_constraints, var_list

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

    def gradient_smt(
        self, x: Iterable[SYMBOL]
    ) -> tuple[Iterable[SYMBOL], Iterable[SYMBOL], list[SYMBOL]]:
        input_vars = np.array(x).reshape(-1, 1)

        z = input_vars
        jacobian = np.eye(self.input_size, self.input_size)
        z_constraints = []
        for mlp in self.mlps:
            z, new_jacobian = network_until_last_layer(mlp, z)
            z_constraints.extend(new_jacobian)
            jacobian = new_jacobian @ jacobian

        var_list = list(set([iv for iv in input_vars.flatten()]))
        return jacobian, z_constraints, var_list

    def save(self, outdir: str, model_name: str = "model") -> str:
        import pathlib
        import yaml

        outdir = pathlib.Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        # save mlp models
        mlp_paths = []
        for idx, mlp in enumerate(self.mlps):
            submodel_name = f"{model_name}_{idx}"
            mlp.save(outdir=str(outdir), model_name=submodel_name)
            mlp_paths.append(f"{submodel_name}.yaml")

        # save params.yaml with net configuration
        params = {
            "module": self.__module__,
            "class": self.__class__.__name__,
            "kwargs": {
                "mlps": mlp_paths,
                "register_module": self.register_module_bool,
                "model_dir": str(outdir.absolute()),
            },
        }

        param_path = outdir / f"{model_name}.yaml"
        with open(param_path, "w") as f:
            yaml.dump(params, f)

        return str(param_path)

    @staticmethod
    def load(config_path: str | pathlib.Path):
        import pathlib
        import yaml

        config_path = pathlib.Path(config_path)
        assert config_path.exists(), f"directory {config_path} does not exist"
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
            params["module"] == "fosco.models.network"
        ), f"Expected fosco.models.network, got {params['module']}"
        assert (
            params["class"] == "SequentialTorchMLP"
        ), f"Expected SequentialTorchMLP, got {params['class']}"

        kwargs = params["kwargs"]
        model = SequentialTorchMLP(**kwargs)

        return model


def network_until_last_layer(
    net: TorchMLP, input_vars: Iterable[SYMBOL]
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
