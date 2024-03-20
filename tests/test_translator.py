import unittest

import numpy as np

from fosco.models import TorchMLP
from fosco.translator import MLPTranslator, make_translator, RobustMLPTranslator


def check_smt_equivalence(expr1, expr2):
    import z3

    s = z3.Solver()
    s.add(z3.Not(expr1 == expr2))

    return s.check() == z3.unsat


class TestTranslator(unittest.TestCase):
    def test_translator_linear_layer(self):
        from fosco.verifier.z3_verifier import VerifierZ3
        from fosco.verifier.types import Z3SYMBOL

        n_vars = 2

        x = VerifierZ3.new_vars(n_vars, base="x")
        x = np.array(x).reshape(-1, 1)
        xdot = np.array(x).reshape(-1, 1)  # dummy xdot = x

        nn = TorchMLP(input_size=n_vars, hidden_sizes=(), activation=(), output_size=1)

        translator = MLPTranslator()
        result_dict, elapsed_time = translator.translate(
            x_v_map={"v": x}, V_net=nn, xdot=xdot
        )
        expr_nn = result_dict["V_symbolic"]
        expr_nndot = result_dict["Vdot_symbolic"]
        self.assertTrue(isinstance(expr_nn, Z3SYMBOL))
        self.assertTrue(isinstance(expr_nndot, Z3SYMBOL))

        w0 = nn.W0.detach().numpy().flatten()
        b0 = nn.b0.detach().numpy().flatten()

        expected_expr_nn = w0 @ x + b0
        grad_nn = w0
        expected_expr_nndot = grad_nn @ xdot

        expected_expr_nn = expected_expr_nn[0]
        expected_expr_nndot = expected_expr_nndot[0]

        ok_nn = check_smt_equivalence(expr_nn, expected_expr_nn)
        self.assertTrue(
            ok_nn,
            f"Wrong symbolic formula for V. Got: \n{expr_nn}, expected: \n{expected_expr_nn}",
        )

        ok_grad = check_smt_equivalence(expr_nndot, expected_expr_nndot)
        self.assertTrue(
            ok_grad,
            f"Wrong symbolic formula for Vdot. Got: \n{expr_nndot}, expected: \n{expected_expr_nndot}",
        )

    def test_translator_two_layers(self):
        from fosco.verifier.z3_verifier import VerifierZ3
        from fosco.verifier.types import Z3SYMBOL

        n_vars = 2

        z3_fns = VerifierZ3.solver_fncts()
        x = VerifierZ3.new_vars(n_vars, base="x")
        x = np.array(x).reshape(-1, 1)

        nn = TorchMLP(
            input_size=n_vars, hidden_sizes=(5,), activation=("relu",), output_size=1
        )

        xdot = np.array(x).reshape(-1, 1)

        translator = MLPTranslator()
        result_dict, elapsed_time = translator.translate(
            x_v_map={"v": x}, V_net=nn, xdot=xdot
        )
        expr_nn = result_dict["V_symbolic"]
        expr_nndot = result_dict["Vdot_symbolic"]
        self.assertTrue(isinstance(expr_nn, Z3SYMBOL))
        self.assertTrue(isinstance(expr_nndot, Z3SYMBOL))

        w0 = nn.W0.detach().numpy()
        b0 = nn.b0.detach().numpy()[:, None]
        w1 = nn.W1.detach().numpy()
        b1 = nn.b1.detach().numpy()[:, None]

        _If = z3_fns["If"]
        # compute symbolic hidden layer
        h1 = w0 @ x + b0
        z1 = np.zeros_like(h1)
        for i in range(z1.shape[0]):
            z1[i, 0] = _If(h1[i, 0] > 0, h1[i, 0], 0)
        # compute symbolic output layer
        z2 = w1 @ z1 + b1

        expected_expr_nn = z2[0, 0]

        ok_nn = check_smt_equivalence(expr_nn, expected_expr_nn)
        self.assertTrue(
            ok_nn,
            f"Wrong symbolic formula for V. Got: \n{expr_nn}, expected: \n{expected_expr_nn}",
        )

        # compute symbolic gradient dy/dx = dy/dz dz/dx
        dy_dz = w1
        dh_dx = w0
        # create dzdh symbolic matrix of shape (2, 2)
        dz_dh = np.array(
            [
                [z3_fns["RealVal"](0) for _ in range(dh_dx.shape[0])]
                for _ in range(dh_dx.shape[0])
            ]
        )
        for i in range(dz_dh.shape[0]):
            for j in range(dz_dh.shape[1]):
                if i == j:
                    dz_dh[i, j] = _If(
                        h1[i, 0] > 0, z3_fns["RealVal"](1), z3_fns["RealVal"](0)
                    )
                else:
                    dz_dh[i, j] = z3_fns["RealVal"](0)
        grad_nn = dy_dz @ (dz_dh @ dh_dx)

        expected_expr_nndot = (grad_nn @ xdot)[0, 0]

        ok_grad = check_smt_equivalence(expr_nndot, expected_expr_nndot)
        self.assertTrue(
            ok_grad,
            f"Wrong symbolic formula for Vdot. Got: \n{expr_nndot}, expected: \n{expected_expr_nndot}",
        )

    def test_translator_three_layers(self):
        from fosco.verifier.z3_verifier import VerifierZ3
        from fosco.verifier.types import Z3SYMBOL

        n_vars = 2

        z3_fns = VerifierZ3.solver_fncts()
        x = VerifierZ3.new_vars(n_vars, base="x")
        x = np.array(x).reshape(-1, 1)

        nn = TorchMLP(
            input_size=n_vars,
            hidden_sizes=(5, 6),
            activation=("relu", "relu"),
            output_size=1,
        )

        xdot = np.array(x).reshape(-1, 1)

        translator = MLPTranslator()
        result_dict, elapsed_time = translator.translate(
            x_v_map={"v": x}, V_net=nn, xdot=xdot
        )
        expr_nn = result_dict["V_symbolic"]
        expr_nndot = result_dict["Vdot_symbolic"]
        self.assertTrue(isinstance(expr_nn, Z3SYMBOL))
        self.assertTrue(isinstance(expr_nndot, Z3SYMBOL))

        w0 = nn.W0.detach().numpy()
        b0 = nn.b0.detach().numpy()[:, None]
        w1 = nn.W1.detach().numpy()
        b1 = nn.b1.detach().numpy()[:, None]
        w2 = nn.W2.detach().numpy()
        b2 = nn.b2.detach().numpy()[:, None]

        _If = z3_fns["If"]
        # compute symbolic hidden layer
        h1 = w0 @ x + b0
        o1 = np.zeros_like(h1)
        for i in range(o1.shape[0]):
            o1[i, 0] = _If(h1[i, 0] > 0, h1[i, 0], 0)
        # compute symbolic hidden layer
        h2 = w1 @ o1 + b1
        o2 = np.zeros_like(h2)
        for i in range(o2.shape[0]):
            o2[i, 0] = _If(h2[i, 0] > 0, h2[i, 0], 0)
        # compute symbolic output layer
        o3 = w2 @ o2 + b2

        expected_expr_nn = o3[0, 0]

        ok_nn = check_smt_equivalence(expr_nn, expected_expr_nn)
        self.assertTrue(
            ok_nn,
            f"Wrong symbolic formula for V. Got: \n{expr_nn}, expected: \n{expected_expr_nn}",
        )

        # compute symbolic gradient dy/dx = dy/dh_i prod_{i} dh_i/dh_{i-1} dh_1/dx
        dy_do2 = w2
        dh2_do1 = w1
        do2_dh2 = np.array(
            [
                [z3_fns["RealVal"](0) for _ in range(dh2_do1.shape[0])]
                for _ in range(dh2_do1.shape[0])
            ]
        )
        for i in range(do2_dh2.shape[0]):
            for j in range(do2_dh2.shape[1]):
                if i == j:
                    do2_dh2[i, j] = _If(
                        h2[i, 0] > 0, z3_fns["RealVal"](1), z3_fns["RealVal"](0)
                    )
                else:
                    do2_dh2[i, j] = z3_fns["RealVal"](0)

        dh1_dx = w0
        do1_dh1 = np.array(
            [
                [z3_fns["RealVal"](0) for _ in range(dh1_dx.shape[0])]
                for _ in range(dh1_dx.shape[0])
            ]
        )
        for i in range(do1_dh1.shape[0]):
            for j in range(do1_dh1.shape[1]):
                if i == j:
                    do1_dh1[i, j] = _If(
                        h1[i, 0] > 0, z3_fns["RealVal"](1), z3_fns["RealVal"](0)
                    )
                else:
                    do1_dh1[i, j] = z3_fns["RealVal"](0)

        for name, matrix in zip(
            ["dy_do3", "do2_dh2", "dh2_do1", "do2_dh1", "dh1_dx"],
            [dy_do2, do2_dh2, dh2_do1, do1_dh1, dh1_dx],
        ):
            print(f"{name}:{matrix.shape}")

        grad_nn = dy_do2 @ (do2_dh2 @ (dh2_do1 @ (do1_dh1 @ dh1_dx)))

        expected_expr_nndot = (grad_nn @ xdot)[0, 0]

        ok_grad = check_smt_equivalence(expr_nndot, expected_expr_nndot)
        self.assertTrue(
            ok_grad,
            f"Wrong symbolic formula for Vdot. Got: \n{expr_nndot}, expected: \n{expected_expr_nndot}",
        )

    def test_relu_out(self):
        """
        Test simbolic formula for a network with relu activation in the output layer.
        """
        from fosco.verifier.z3_verifier import VerifierZ3
        from fosco.verifier.types import Z3SYMBOL

        z3_fns = VerifierZ3.solver_fncts()
        _If = z3_fns["If"]

        n_vars = 2

        x = VerifierZ3.new_vars(n_vars, base="x")
        x = np.array(x).reshape(-1, 1)

        nn = TorchMLP(
            input_size=n_vars,
            hidden_sizes=(),
            activation=(),
            output_size=1,
            output_activation="relu",
        )

        xdot = np.array(x).reshape(-1, 1)

        translator = MLPTranslator()
        result_dict, elapsed_time = translator.translate(
            x_v_map={"v": x}, V_net=nn, xdot=xdot
        )
        expr_nn = result_dict["V_symbolic"]
        expr_nndot = result_dict["Vdot_symbolic"]
        self.assertTrue(isinstance(expr_nn, Z3SYMBOL))
        self.assertTrue(isinstance(expr_nndot, Z3SYMBOL))

        w0 = nn.W0.detach().numpy()
        b0 = nn.b0.detach().numpy()[:, None]

        # compute symbolic output layer
        z1 = w0 @ x + b0
        z1 = _If(z1[0, 0] > 0, z1[0, 0], 0)  # add output layer with relu activation

        expected_expr_nn = z1

        ok_nn = check_smt_equivalence(expr_nn, expected_expr_nn)
        self.assertTrue(
            ok_nn,
            f"Wrong symbolic formula for V. Got: \n{expr_nn}, expected: \n{expected_expr_nn}",
        )

        # compute symbolic gradient dy/dx = dy/dz dz/dx
        RealVal_ = z3_fns["RealVal"]
        dy_dz = np.array([[_If(z1 > 0, RealVal_(1), RealVal_(0))]])
        dz_dx = w0

        grad_nn = dy_dz @ dz_dx
        expected_expr_nndot = (grad_nn @ xdot)[0, 0]

        ok_grad = check_smt_equivalence(expr_nndot, expected_expr_nndot)
        self.assertTrue(
            ok_grad,
            f"Wrong symbolic formula for Vdot. Got: \n{expr_nndot}, expected: \n{expected_expr_nndot}",
        )

    def test_translator_two_layers_relu_out(self):
        from fosco.verifier.z3_verifier import VerifierZ3
        from fosco.verifier.types import Z3SYMBOL

        z3_fns = VerifierZ3.solver_fncts()
        _If = z3_fns["If"]

        n_vars = 2

        x = VerifierZ3.new_vars(n_vars, base="x")
        x = np.array(x).reshape(-1, 1)

        nn = TorchMLP(
            input_size=n_vars,
            hidden_sizes=(5,),
            activation=("relu",),
            output_size=1,
            output_activation="relu",
        )

        xdot = np.array(x).reshape(-1, 1)

        translator = MLPTranslator()
        result_dict, elapsed_time = translator.translate(
            x_v_map={"v": x}, V_net=nn, xdot=xdot
        )
        expr_nn = result_dict["V_symbolic"]
        expr_nndot = result_dict["Vdot_symbolic"]
        self.assertTrue(isinstance(expr_nn, Z3SYMBOL))
        self.assertTrue(isinstance(expr_nndot, Z3SYMBOL))

        w1 = nn.W0.detach().numpy()
        b1 = nn.b0.detach().numpy()[:, None]
        w2 = nn.W1.detach().numpy()
        b2 = nn.b1.detach().numpy()[:, None]

        # compute symbolic hidden layer
        h1 = w1 @ x + b1
        z1 = np.zeros_like(h1)
        for i in range(z1.shape[0]):
            z1[i, 0] = _If(h1[i, 0] > 0, h1[i, 0], 0)
        # compute symbolic output layer
        z2 = w2 @ z1 + b2
        z2 = _If(z2[0, 0] > 0, z2[0, 0], 0)  # add output layer with relu activation

        expected_expr_nn = z2

        ok_nn = check_smt_equivalence(expr_nn, expected_expr_nn)
        self.assertTrue(
            ok_nn,
            f"Wrong symbolic formula for V. Got: \n{expr_nn}, expected: \n{expected_expr_nn}",
        )

        # compute symbolic gradient dy/dx = dy/dz dz/dx
        RealVal_ = z3_fns["RealVal"]
        dact_dy = np.array([[_If(z2 > 0, RealVal_(1), RealVal_(0))]])
        dy_dz = w2
        dh_dx = w1
        # create dzdh symbolic matrix of shape (2, 2)
        dz_dh = np.array(
            [
                [RealVal_(0) for _ in range(dh_dx.shape[0])]
                for _ in range(dh_dx.shape[0])
            ]
        )
        for i in range(dz_dh.shape[0]):
            for j in range(dz_dh.shape[1]):
                if i == j:
                    dz_dh[i, j] = _If(h1[i, 0] > 0, RealVal_(1), RealVal_(0))
                else:
                    dz_dh[i, j] = RealVal_(0)
        grad_nn = dact_dy @ (dy_dz @ (dz_dh @ dh_dx))

        expected_expr_nndot = (grad_nn @ xdot)[0, 0]
        ok_grad = check_smt_equivalence(expr_nndot, expected_expr_nndot)
        self.assertTrue(
            ok_grad,
            f"Wrong symbolic formula for Vdot. Got: \n{expr_nndot}, expected: \n{expected_expr_nndot}",
        )

    def test_factory(self):
        from fosco.common.consts import VerifierType
        from fosco.common.consts import TimeDomain
        from fosco.common.consts import CertificateType

        translator = make_translator(
            certificate_type=CertificateType.CBF,
            verifier_type=VerifierType.Z3,
            time_domain=TimeDomain.CONTINUOUS,
        )
        self.assertTrue(isinstance(translator, MLPTranslator))

        translator = make_translator(
            certificate_type=CertificateType.RCBF,
            verifier_type=VerifierType.Z3,
            time_domain=TimeDomain.CONTINUOUS,
        )
        self.assertTrue(isinstance(translator, RobustMLPTranslator))

        self.assertRaises(
            NotImplementedError,
            make_translator,
            certificate_type=CertificateType.CBF,
            verifier_type=VerifierType.Z3,
            time_domain=TimeDomain.DISCRETE,
        )

    def test_sympy_activation(self):
        """
        Test activation functions in sympy
        """
        from fosco.common.consts import ActivationType
        from fosco.common.activations_symbolic import activation_sym

        import z3
        import sympy
        import dreal

        spx = np.array(sympy.symbols("x y")).reshape(-1, 1)
        z3x = np.array(z3.Real("x y")).reshape(-1, 1)
        drx = np.array([dreal.Variable(v) for v in ["x", "y"]]).reshape(-1, 1)
        for act in ActivationType:
            act_sp = activation_sym(act, spx)
            act_z3 = activation_sym(act, z3x)
            act_dr = activation_sym(act, drx)

            print(act)
            print("sympy", act_sp)
            print("z3", act_z3)
            print("dreal", act_dr)
            print("")
