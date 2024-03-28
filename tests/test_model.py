import os
import unittest

import matplotlib.pyplot as plt
import numpy as np
import torch
import z3

from fosco.common.consts import ActivationType, DomainName
from fosco.models import TorchMLP
from fosco.models.network import SequentialTorchMLP
from tests.test_translator import check_smt_equivalence
import shutil


class TestModel(unittest.TestCase):
    def test_torchsym_model(self):
        from fosco.models import TorchMLP

        model = TorchMLP(
            input_size=2,
            hidden_sizes=(4, 4),
            activation=("relu", "relu"),
            output_size=1,
        )

        # numerical
        x_batch = torch.randn(10, 2)
        y_batch = model(x_batch)
        dydx_batch = model.gradient(x_batch)

        self.assertTrue(y_batch.shape == (10, 1))
        self.assertTrue(dydx_batch.shape == (10, 2))

        # symbolic
        x_sym = z3.Reals("x0 x1")
        y_sym, y_constr, y_vars = model.forward_smt(x_sym)
        dydx_sym, dydx_constr, dydx_vars = model.gradient_smt(x_sym)

        self.assertTrue(isinstance(y_sym, z3.ArithRef))
        self.assertTrue(all([isinstance(c, z3.BoolRef) for c in y_constr]))
        self.assertTrue(
            all([isinstance(dydx, z3.ArithRef) for dydx in dydx_sym[0]]),
            f"dydx_sym: {dydx_sym}",
        )
        self.assertTrue(all([isinstance(c, z3.BoolRef) for c in dydx_constr]))

    def test_save_mlp_model(self):
        from fosco.models import TorchMLP

        tmp_dir = "tmp"

        # if exists, remove tmp_dir

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
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, "model.yaml")))

        # try to load model
        model2 = TorchMLP.load(config_path=os.path.join(tmp_dir, "model.yaml"))
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
        from fosco.models import TorchMLP

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
        from fosco.systems import make_system
        from barriers import make_barrier
        from fosco.config import CegisConfig
        from fosco.cegis import Cegis

        system_type = "SingleIntegrator"
        system_fn = make_system(system_type)
        init_barrier = make_barrier(system=system_fn(), model_to_load="default")

        sets = system_fn().domains
        assert all(
            [dn.value in sets for dn in [DomainName.XI, DomainName.XU, DomainName.XD]]
        )

        data_gen = {
            DomainName.XI.value: lambda n: sets[DomainName.XI.value].generate_data(n),
            DomainName.XU.value: lambda n: sets[DomainName.XU.value].generate_data(n),
            DomainName.XD.value: lambda n: torch.concatenate(
                [
                    sets[DomainName.XD.value].generate_data(n),
                    sets[DomainName.UD.value].generate_data(n),
                ],
                dim=1,
            ),
        }

        cfg = CegisConfig(BARRIER_TO_LOAD="default", CEGIS_MAX_ITERS=10,)

        cegis = Cegis(system=system_fn(), domains=sets, config=cfg, data_gen=data_gen)

        self.assertTrue(
            isinstance(cegis.learner.net, type(init_barrier)), "type mismatch"
        )

        # numerical check on in-out
        x = torch.randn(10, 2)
        y = cegis.learner.net(x)
        y_init = init_barrier(x)
        self.assertTrue(torch.allclose(y, y_init), f"expected {y_init}, got {y}")

    def test_make_mlp(self):
        from fosco.models import make_mlp

        layers, acts = make_mlp(
            input_size=2,
            hidden_sizes=(4, 4),
            hidden_activation=("relu", "relu"),
            output_size=1,
            output_activation="linear",
        )

        self.assertTrue(len(layers) == 3)
        self.assertTrue(len(acts) == 3)
        self.assertTrue(all([isinstance(l, torch.nn.Linear) for l in layers]))
        self.assertTrue(all([isinstance(a, ActivationType) for a in acts]))
        self.assertTrue(all([a == ActivationType.RELU for a in acts[:-1]]))
        self.assertTrue(acts[-1] == ActivationType.LINEAR)

    def test_sequential_mlp(self):
        from fosco.models.network import SequentialTorchMLP

        mlp1 = TorchMLP(
            input_size=2,
            hidden_sizes=(4,),
            activation=("relu",),
            output_size=4,
            output_activation="linear",
        )
        mlp2 = TorchMLP(
            input_size=4,
            hidden_sizes=(4,),
            activation=("relu",),
            output_size=1,
            output_activation="linear",
        )

        model = SequentialTorchMLP(mlps=[mlp1, mlp2],)

        x_batch = torch.randn(10, 2)
        y_batch = model(x_batch)
        dydx_batch = model.gradient(x_batch)
        ytarget_batch = mlp2(mlp1(x_batch))

        self.assertTrue(
            torch.allclose(y_batch, ytarget_batch),
            f"expected {ytarget_batch}, got {y_batch}",
        )
        self.assertTrue(
            dydx_batch.shape == (10, 2),
            f"dydx expected to have shape (10, 2), got {dydx_batch.shape}",
        )

        # symbolic
        x_sym = z3.Reals("x0 x1")
        y_sym, y_constr, y_vars = model.forward_smt(x_sym)

        y1_sym, y1_constr, y1_vars = mlp1.forward_smt(x=x_sym)
        y2_sym, y2_constr, y2_vars = mlp2.forward_smt(x=y1_sym)

        self.assertTrue(
            isinstance(y2_sym, z3.ArithRef), f"expected z3.ArithRef, got {type(y2_sym)}"
        )
        self.assertTrue(
            isinstance(y_sym, z3.ArithRef), f"expected z3.ArithRef, got {type(y_sym)}"
        )

        self.assertTrue(
            check_smt_equivalence(y_sym, y2_sym),
            "symbolic expressions are not equivalent",
        )

    def test_save_sequential_model(self):
        from fosco.models import TorchMLP

        tmp_dir = "tmp"

        # if exists, remove tmp_dir
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

        mlp1 = TorchMLP(
            input_size=2,
            hidden_sizes=(4,),
            activation=("relu",),
            output_size=4,
            output_activation="linear",
        )
        mlp2 = TorchMLP(
            input_size=4,
            hidden_sizes=(4,),
            activation=("relu",),
            output_size=1,
            output_activation="linear",
        )
        model = SequentialTorchMLP(mlps=[mlp1, mlp2])
        model.save(outdir=tmp_dir, model_name="model")

        # check if model saved
        self.assertTrue(os.path.exists(tmp_dir))
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, "model.yaml")))
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, "model_0.pt")))
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, "model_0.yaml")))
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, "model_1.pt")))
        self.assertTrue(os.path.exists(os.path.join(tmp_dir, "model_1.yaml")))

        # try to load model
        config_path = os.path.join(tmp_dir, "model.yaml")
        model2 = SequentialTorchMLP.load(config_path=config_path)
        self.assertEqual(model.input_size, model2.input_size)
        self.assertEqual(model.output_size, model2.output_size)
        for mlp1, mlp2 in zip(model.mlps, model2.mlps):
            self.assertEqual(mlp1.input_size, mlp2.input_size)
            self.assertEqual(mlp1.output_size, mlp2.output_size)
            self.assertEqual(mlp1.acts, mlp2.acts)
            self.assertEqual(len(mlp1.layers), len(mlp2.layers))
            for layer, layer2 in zip(mlp1.layers, mlp2.layers):
                self.assertTrue(torch.allclose(layer.weight, layer2.weight))
                self.assertTrue(torch.allclose(layer.bias, layer2.bias))

        # remove tmp_dir
        shutil.rmtree(tmp_dir)

    def test_robust_gate_forward(self):
        from fosco.models.network import RobustGate

        batch = torch.linspace(-10.0, 10.0, 100).reshape(-1, 1)
        model = RobustGate(activation_type="hsigmoid")

        # numerical ground truth
        y = model(batch).detach().numpy()
        dydx = model.gradient(batch)

        self.assertTrue(y.shape == (100, 1))
        self.assertTrue(dydx.shape == (100, 1))

        # symbolic
        x_sym = z3.Reals("x")
        y_sym, y_constr, y_vars = model.forward_smt(x_sym)
        dydx_sym, dydx_constr, dydx_vars = model.gradient_smt(x_sym)

        self.assertTrue(isinstance(y_sym, z3.ArithRef))
        self.assertTrue(isinstance(dydx_sym, z3.ArithRef))
        self.assertTrue(len(y_constr) == 0)
        self.assertTrue(len(dydx_constr) == 0)
        self.assertEqual(y_vars, x_sym)
        self.assertEqual(y_vars, dydx_vars)

        # for each point in batch, check symbolic equivalence
        symbpoints = []
        for x in batch:
            y_val = model(torch.tensor(x)).item()
            dydx_val = model.gradient(torch.tensor(x)).item()
            y_sym_val = z3.simplify(z3.substitute(y_sym, (x_sym[0], z3.RealVal(x.item()))))
            dydx_sym_val = z3.simplify(z3.substitute(y_sym, (x_sym[0], z3.RealVal(x.item()))))

            symbpoints.append(float(y_sym_val.as_fraction().numerator/y_sym_val.as_fraction().denominator))
            #self.assertTrue(
            #    np.isclose(float(str(y_sym_val)), y_val, atol=1e-3),
            #    f"expected {y_val}, got {y_sym_val}",
            #)
            #self.assertTrue(
            #    np.isclose(float(str(dydx_sym_val)), dydx_val, atol=1e-3),
            #    f"expected {dydx_val}, got {dydx_sym_val}",
            #)

        #plt.plot(batch, y, label="y")
        #plt.plot(batch, symbpoints, label="y_sim")
        #plt.show()




