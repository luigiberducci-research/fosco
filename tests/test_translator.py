import unittest

import numpy as np
import torch

from fosco.models.network import TorchMLP
from fosco.translator import MLPZ3Translator, make_translator
from fosco.verifier import VerifierZ3


class TestTranslator(unittest.TestCase):
    def test_translator_linear_layer(self):
        import z3

        n_vars = 2

        x = VerifierZ3.new_vars(n_vars, base="x")
        x = np.array(x).reshape(-1, 1)

        nn = TorchMLP(input_size=n_vars, hidden_sizes=(), activation=(), output_size=1)

        xdot = np.array(x).reshape(-1, 1)

        translator = MLPZ3Translator(rounding=-1)

        expr_nn, expr_nndot = translator.get_symbolic_formula(x, nn, xdot)
        assert isinstance(expr_nn, z3.ArithRef)
        assert isinstance(expr_nndot, z3.ArithRef)

        w1 = nn.W1.detach().numpy().flatten()
        b1 = nn.b1.detach().numpy().flatten()

        expected_expr_nn = w1 @ x + b1
        grad_nn = w1
        expected_expr_nndot = grad_nn @ xdot

        expected_expr_nn = z3.simplify(expected_expr_nn[0])
        expected_expr_nndot = z3.simplify(expected_expr_nndot[0])

        assert str(expr_nn) == str(
            expected_expr_nn
        ), f"Wrong symbolic formula for V, got {expr_nn}"
        assert str(expr_nndot) == str(
            expected_expr_nndot
        ), f"Wrong symbolic formula for Vdot, got {expr_nndot}"

    def test_translator_two_layers(self):
        import z3

        n_vars = 2

        x = VerifierZ3.new_vars(n_vars, base="x")
        x = np.array(x).reshape(-1, 1)

        nn = TorchMLP(
            input_size=n_vars, hidden_sizes=(5,), activation=("relu",), output_size=1
        )

        xdot = np.array(x).reshape(-1, 1)

        translator = MLPZ3Translator(rounding=-1)

        expr_nn, expr_nndot = translator.get_symbolic_formula(x, nn, xdot)
        assert isinstance(expr_nn, z3.ArithRef)
        assert isinstance(expr_nndot, z3.ArithRef)

        w1 = nn.W1.detach().numpy()
        b1 = nn.b1.detach().numpy()[:, None]
        w2 = nn.W2.detach().numpy()
        b2 = nn.b2.detach().numpy()[:, None]

        _If = z3.If
        # compute symbolic hidden layer
        h1 = w1 @ x + b1
        z1 = np.zeros_like(h1)
        for i in range(z1.shape[0]):
            z1[i, 0] = _If(h1[i, 0] > 0, h1[i, 0], 0)
        # compute symbolic output layer
        z2 = w2 @ z1 + b2

        expected_expr_nn = z2[0, 0]
        expected_expr_nn = z3.simplify(expected_expr_nn)
        assert str(expr_nn) == str(
            expected_expr_nn
        ), f"Wrong symbolic formula for V, got {expr_nn}"

        # compute symbolic gradient dy/dx = dy/dz dz/dx
        dy_dz = w2
        dh_dx = w1
        # create dzdh symbolic matrix of shape (2, 2)
        dz_dh = np.array(
            [
                [z3.RealVal(0) for _ in range(dh_dx.shape[0])]
                for _ in range(dh_dx.shape[0])
            ]
        )
        for i in range(dz_dh.shape[0]):
            for j in range(dz_dh.shape[1]):
                if i == j:
                    dz_dh[i, j] = _If(h1[i, 0] > 0, z3.RealVal(1), z3.RealVal(0))
                else:
                    dz_dh[i, j] = z3.RealVal(0)
        grad_nn = dy_dz @ (dz_dh @ dh_dx)

        expected_expr_nndot = (grad_nn @ xdot)[0, 0]
        expected_expr_nndot = z3.simplify(expected_expr_nndot)
        assert str(expr_nndot) == str(
            expected_expr_nndot
        ), f"Wrong symbolic formula for Vdot, got {expr_nndot}"

    def test_translator_three_layers(self):
        import z3

        n_vars = 2

        x = VerifierZ3.new_vars(n_vars, base="x")
        x = np.array(x).reshape(-1, 1)

        nn = TorchMLP(
            input_size=n_vars,
            hidden_sizes=(5, 6),
            activation=("relu", "relu"),
            output_size=1,
        )

        xdot = np.array(x).reshape(-1, 1)

        translator = MLPZ3Translator(rounding=-1)

        expr_nn, expr_nndot = translator.get_symbolic_formula(x, nn, xdot)
        assert isinstance(expr_nn, z3.ArithRef)
        assert isinstance(expr_nndot, z3.ArithRef)

        w1 = nn.W1.detach().numpy()
        b1 = nn.b1.detach().numpy()[:, None]
        w2 = nn.W2.detach().numpy()
        b2 = nn.b2.detach().numpy()[:, None]
        w3 = nn.W3.detach().numpy()
        b3 = nn.b3.detach().numpy()[:, None]

        _If = z3.If
        # compute symbolic hidden layer
        h1 = w1 @ x + b1
        o1 = np.zeros_like(h1)
        for i in range(o1.shape[0]):
            o1[i, 0] = _If(h1[i, 0] > 0, h1[i, 0], 0)
        # compute symbolic hidden layer
        h2 = w2 @ o1 + b2
        o2 = np.zeros_like(h2)
        for i in range(o2.shape[0]):
            o2[i, 0] = _If(h2[i, 0] > 0, h2[i, 0], 0)
        # compute symbolic output layer
        o3 = w3 @ o2 + b3

        expected_expr_nn = o3[0, 0]
        expected_expr_nn = z3.simplify(expected_expr_nn)
        assert str(expr_nn) == str(
            expected_expr_nn
        ), f"Wrong symbolic formula for V, got {expr_nn}"

        # compute symbolic gradient dy/dx = dy/dh_i prod_{i} dh_i/dh_{i-1} dh_1/dx
        dy_do2 = w3
        dh2_do1 = w2
        do2_dh2 = np.array(
            [
                [z3.RealVal(0) for _ in range(dh2_do1.shape[0])]
                for _ in range(dh2_do1.shape[0])
            ]
        )
        for i in range(do2_dh2.shape[0]):
            for j in range(do2_dh2.shape[1]):
                if i == j:
                    do2_dh2[i, j] = _If(h2[i, 0] > 0, z3.RealVal(1), z3.RealVal(0))
                else:
                    do2_dh2[i, j] = z3.RealVal(0)

        dh1_dx = w1
        do1_dh1 = np.array(
            [
                [z3.RealVal(0) for _ in range(dh1_dx.shape[0])]
                for _ in range(dh1_dx.shape[0])
            ]
        )
        for i in range(do1_dh1.shape[0]):
            for j in range(do1_dh1.shape[1]):
                if i == j:
                    do1_dh1[i, j] = _If(h1[i, 0] > 0, z3.RealVal(1), z3.RealVal(0))
                else:
                    do1_dh1[i, j] = z3.RealVal(0)

        for name, matrix in zip(
            ["dy_do3", "do2_dh2", "dh2_do1", "do2_dh1", "dh1_dx"],
            [dy_do2, do2_dh2, dh2_do1, do1_dh1, dh1_dx],
        ):
            print(f"{name}:{matrix.shape}")

        grad_nn = dy_do2 @ (do2_dh2 @ (dh2_do1 @ (do1_dh1 @ dh1_dx)))

        expected_expr_nndot = (grad_nn @ xdot)[0, 0]
        expected_expr_nndot = z3.simplify(expected_expr_nndot)

        assert str(expr_nndot) == str(
            expected_expr_nndot
        ), f"Wrong symbolic formula for Vdot, got {expr_nndot}"

    def test_separation_symbolic_functions(self):
        n_vars = 2

        x = VerifierZ3.new_vars(n_vars, base="x")
        x = np.array(x).reshape(-1, 1)

        nn = TorchMLP(input_size=n_vars, hidden_sizes=(), activation=(), output_size=1)

        xdot = np.array(x).reshape(-1, 1)

        translator = MLPZ3Translator(rounding=-1)

        expr_nn, expr_nndot = translator.get_symbolic_formula(x, nn, xdot)

        expr_nn2 = translator.get_symbolic_net(x, nn)
        expr_nn_grad = translator.get_symbolic_net_grad(x, nn)
        expr_nndot2 = (expr_nn_grad @ xdot)[0, 0]

        assert str(expr_nn) == str(
            expr_nn2
        ), f"Wrong symbolic formula for V, got {expr_nn}"
        assert str(expr_nndot) == str(
            expr_nndot2
        ), f"Wrong symbolic formula for Vdot, got {expr_nndot}"

    def test_factory(self):
        from fosco.common.consts import VerifierType
        from fosco.common.consts import TimeDomain

        translator = make_translator(
            verifier_type=VerifierType.Z3, time_domain=TimeDomain.CONTINUOUS
        )
        self.assertTrue(isinstance(translator, MLPZ3Translator))

        self.assertRaises(
            NotImplementedError,
            make_translator,
            verifier_type=VerifierType.Z3,
            time_domain=TimeDomain.DISCRETE,
        )
