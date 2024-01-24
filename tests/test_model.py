import os
import unittest

import torch
import z3

from fosco.common.consts import ActivationType
from systems import make_domains


class TestModel(unittest.TestCase):

    def test_torchsym_model(self):
        from models.network import TorchMLP

        model = TorchMLP(
            input_size=2, hidden_sizes=(4, 4), activation=("relu", "relu"), output_size=1
        )

        # numerical
        x_batch = torch.randn(10, 2)
        y_batch = model(x_batch)
        dydx_batch = model.gradient(x_batch)

        self.assertTrue(y_batch.shape == (10, 1))
        self.assertTrue(dydx_batch.shape == (10, 2))

        # symbolic
        x_sym = z3.Reals("x0 x1")
        y_sym = model.forward_smt(x_sym)
        dydx_sym = model.gradient_smt(x_sym)

        self.assertTrue(isinstance(y_sym, z3.ArithRef))
        self.assertTrue(all([isinstance(dydx, z3.ArithRef) for dydx in dydx_sym[0]]), f"dydx_sym: {dydx_sym}")





    def test_save_mlp_model(self):
        from models.network import TorchMLP

        tmp_dir = "tmp"

        # if exists, remove tmp_dir
        import shutil

        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

        act = ActivationType.RELU
        model = TorchMLP(
            input_size=2, hidden_sizes=(4, 4), activation=(act, act), output_size=1
        )
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
        from models.network import TorchMLP

        model = TorchMLP(
            input_size=2,
            hidden_sizes=(4, 4),
            activation=("relu", "relu"),
            output_size=1,
            output_activation="relu",
        )

        x_batch = torch.randn(10, 2)
        y_batch = model(x_batch)

        self.assertEqual(
            y_batch.shape, (10, 1), f"expected shape (10, 1), got {y_batch.shape}"
        )
        self.assertTrue(
            torch.all(y_batch >= 0.0),
            f"relu output must be non-negative, got {y_batch}",
        )

    def test_use_init_model(self):
        """
        This test check the init model is correctly loaded and used.
        """
        from systems import make_system
        from barriers import make_barrier
        from fosco.config import CegisConfig
        from fosco.cegis import Cegis

        system_type = "single_integrator"
        system_fn = make_system(system_type)
        barrier_dict = make_barrier(system=system_fn())
        init_barrier = barrier_dict["barrier"]

        sets = {
            k: s for k, s in make_domains(system_id=system_type).items() if k in ["lie", "input", "init", "unsafe"]
        }
        data_gen = {
            "init": lambda n: sets["init"].generate_data(n),
            "unsafe": lambda n: sets["unsafe"].generate_data(n),
            "lie": lambda n: torch.concatenate(
                [sets["lie"].generate_data(n), sets["input"].generate_data(n)], dim=1
            )
        }

        cfg = CegisConfig(
            SYSTEM=system_fn,
            DOMAINS=sets,
            DATA_GEN=data_gen,
            USE_INIT_MODELS=True,
            CEGIS_MAX_ITERS=10,
        )

        cegis = Cegis(config=cfg, verbose=2)

        self.assertTrue(isinstance(cegis.learner.net, type(init_barrier)), "type mismatch")

        # numerical check on in-out
        x = torch.randn(10, 2, 1)
        y = cegis.learner.net(x)
        y_init = init_barrier(x)
        self.assertTrue(torch.allclose(y, y_init), f"expected {y_init}, got {y}")


