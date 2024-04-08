import importlib
import pathlib

import numpy as np
import torch
import yaml

from fosco.models import TorchSymDiffModel, TorchSymModel


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def load_model(config_path: str | pathlib.Path) -> TorchSymDiffModel | TorchSymModel:
    """Load a model from a config file."""
    assert str(config_path).endswith(".yaml"), f"expected .yaml file, got {config_path}"
    assert pathlib.Path(config_path).exists(), f"model path {config_path} does not exist"

    with open(config_path, "r") as file:
        params = yaml.safe_load(file)

    assert "class" in params, f"expected 'class' in {params}, got {params.keys()}"

    module = importlib.import_module(params["module"])
    cls = getattr(module, params["class"])
    return cls.load(config_path=config_path)