import importlib
import pathlib

import yaml

from .torchsym import TorchSymFn, TorchSymDiffFn, TorchSymModel, TorchSymDiffModel
from .network import TorchMLP, SequentialTorchMLP, make_mlp


def load_model(config_path: str | pathlib.Path) -> TorchSymDiffModel | TorchSymModel:
    """Load a model from a config file."""
    assert config_path.endswith(".yaml"), f"expected .yaml file, got {config_path}"
    assert pathlib.Path(config_path).exists(), f"model path {config_path} does not exist"

    with open(config_path, "r") as file:
        params = yaml.safe_load(file)

    assert "class" in params, f"expected 'class' in {params}, got {params.keys()}"

    module = importlib.import_module(params["module"])
    cls = getattr(module, params["class"])
    return cls(**params["kwargs"])
