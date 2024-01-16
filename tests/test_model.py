import os
import unittest

import torch

from fosco.common.consts import ActivationType


class TestModel(unittest.TestCase):
    def test_save_mlp_model(self):
        from fosco.models.network import TorchMLP
        tmp_dir = "tmp"

        # if exists, remove tmp_dir
        import shutil
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

        act = ActivationType.RELU
        model = TorchMLP(input_size=2, hidden_sizes=(4, 4), activation=(act, act), output_size=1)
        model.save(outdir=tmp_dir)

        # check if model saved
        self.assertTrue(os.path.exists(tmp_dir))
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, "model.pt")))
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, "params.yaml")))

        # try to load model
        model2 = TorchMLP.load(logdir=tmp_dir)
        self.assertEqual(model.input_size, model2.input_size)
        self.assertEqual(model.output_size, model2.output_size)
        self.assertEqual(model.acts, model2.acts)
        self.assertEqual(len(model.layers), len(model2.layers))
        for layer, layer2 in zip(model.layers, model2.layers):
            self.assertTrue(torch.allclose(layer.weight, layer2.weight))
            self.assertTrue(torch.allclose(layer.bias, layer2.bias))

        # remove tmp_dir
        shutil.rmtree(tmp_dir)

    def test_mlp_relu_out(self):
        from fosco.models.network import TorchMLP

        model = TorchMLP(input_size=2,
                         hidden_sizes=(4, 4), activation=("relu", "relu"),
                         output_size=1, output_activation="relu")

        x_batch = torch.randn(10, 2)
        y_batch = model(x_batch)

        self.assertEqual(y_batch.shape, (10, 1), f"expected shape (10, 1), got {y_batch.shape}")
        self.assertTrue(torch.all(y_batch >= 0.0), f"relu output must be non-negative, got {y_batch}")





