import torch
from torch import nn

from fosco.common.activations import activation
from fosco.common.consts import ActivationType


class TorchMLP(nn.Module):
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

        act = ActivationType[output_activation.upper()] if isinstance(output_activation, str) else output_activation
        self.acts.append(act)

        assert len(self.layers) == len(self.acts), "layers and activations must have the same length"
        assert self.output_size == self.layers[-1].out_features, "output size does not match last layer size"

    def forward(self, x):
        y = x
        for idx, layer in enumerate(self.layers):
            z = layer(y)
            y = activation(self.acts[idx], z)

        return y

    def compute_net_gradnet(self, S: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the value of the neural network and its gradient.

        Computes gradient using autograd.

            S (torch.Tensor): input tensor

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (nn, grad_nn)
        """
        S_clone = torch.clone(S).requires_grad_()
        nn = self(S_clone)

        grad_nn = torch.autograd.grad(
            outputs=nn,
            inputs=S_clone,
            grad_outputs=torch.ones_like(nn),
            create_graph=True,
            retain_graph=True,
        )[0]
        return nn, grad_nn

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


