import unittest

import matplotlib.pyplot as plt
import numpy as np
import torch
import z3

from fosco.common.activations import activation, activation_der
from fosco.common.activations_symbolic import activation_sym, activation_der_sym
from fosco.common.consts import ActivationType
from fosco.verifier.types import Z3SYMBOL


class TestActivations(unittest.TestCase):
    def test_all_activations_z3(self):
        x_batch = torch.linspace(-10, 10, 500).reshape(-1, 1)
        x_sym = np.array([z3.Reals("x")])

        act_success = []
        act_fail = []
        for activation_type in ActivationType:
            try:
                y_batch = activation(activation_type, p=x_batch)
                y_sym = activation_sym(select=activation_type, p=x_sym)[0, 0]
            except NotImplementedError:
                act_fail.append(activation_type)
                continue

            symbolic_points = []
            for x in x_batch:
                y_sym_val = z3.simplify(
                    z3.substitute(y_sym, (x_sym[0, 0], z3.RealVal(x.item())))
                )
                symbolic_points.append(
                    float(
                        y_sym_val.as_fraction().numerator
                        / y_sym_val.as_fraction().denominator
                    )
                )
            symbolic_points = torch.tensor(symbolic_points)

            # plt.plot(x_batch, y_batch, label="torch")
            # plt.plot(x_batch, symbolic_points, label="z3")
            # plt.legend()
            # plt.show()

            self.assertTrue(
                torch.allclose(y_batch[:, 0], symbolic_points, atol=1e-2),
                f"Activation {activation_type} failed",
            )

            act_success.append(activation_type)

        print(
            f"Z3 symbolic activations passed {len(act_success)}/{len(ActivationType)}"
        )
        print(f"Failed activations: {act_fail}")

    def test_all_derivative_z3(self):
        x_batch = torch.linspace(-10, 10, 500).reshape(-1, 1)
        x_sym = np.array([z3.Reals("x")])

        act_success = []
        act_fail = []
        for activation_type in ActivationType:
            try:
                y_batch = activation_der(activation_type, p=x_batch)
                y_sym = activation_der_sym(select=activation_type, p=x_sym)[0, 0]
            except NotImplementedError:
                act_fail.append(activation_type)
                continue

            symbolic_points = []
            for x in x_batch:
                if isinstance(y_sym, Z3SYMBOL):
                    y_sym_val = z3.simplify(
                        z3.substitute(y_sym, (x_sym[0, 0], z3.RealVal(x.item())))
                    )
                    y_sym_val = float(
                        y_sym_val.as_fraction().numerator
                        / y_sym_val.as_fraction().denominator
                    )
                else:
                    y_sym_val = y_sym
                symbolic_points.append(y_sym_val)
            symbolic_points = torch.tensor(symbolic_points).float()

            # plt.plot(x_batch, y_batch, label="torch")
            # plt.plot(x_batch, symbolic_points, label="z3")
            # plt.legend()
            # plt.show()

            self.assertTrue(
                torch.allclose(y_batch[:, 0], symbolic_points, atol=1e-2),
                f"Activation {activation_type} failed",
            )

            act_success.append(activation_type)

        print(
            f"Z3 symbolic activations passed {len(act_success)}/{len(ActivationType)}"
        )
        print(f"Failed activations: {act_fail}")
