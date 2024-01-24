import unittest

import numpy as np
import torch
import z3

from systems.uncertainty_wrappers import AdditiveBoundedUncertainty
from tests.test_translator import check_smt_equivalence


class TestControlAffineDynamicalSystem(unittest.TestCase):
    def test_single_integrator(self):
        from systems.single_integrator import SingleIntegrator

        x = np.zeros((10, 2))
        u = np.ones((10, 2))
        T = 10.0
        dt = 0.1

        f = SingleIntegrator()

        t = dt
        while t < T:
            x = x + dt * f(x, u)
            t += dt

        self.assertTrue(np.allclose(x, 10.0 * np.ones_like(x)), f"got {x}")

    def test_single_integrator_z3(self):
        from systems.single_integrator import SingleIntegrator

        state_vars = ["x", "y"]
        input_vars = ["vx", "vy"]
        x = [z3.Real(var) for var in state_vars]
        u = [z3.Real(var) for var in input_vars]

        f = SingleIntegrator()

        xdot = f.f(x, u)

        self.assertTrue(
            str(xdot[0]) == input_vars[0], "expected xdot = vx, got {xdot[0]}"
        )
        self.assertTrue(
            str(xdot[1]) == input_vars[1], "expected ydot = vy, got {xdot[1]}"
        )


class TestUncertainControlAffineDynamicalSystem(unittest.TestCase):
    def test_noisy_single_integrator(self):
        from systems.single_integrator import SingleIntegrator

        x = np.zeros((10, 2))
        u = np.ones((10, 2))
        z = np.ones((10, 2)) * 0.1

        T = 10.0
        dt = 0.1

        f = AdditiveBoundedUncertainty(base_system=SingleIntegrator())

        t = dt
        while t < T:
            x = x + dt * f(x, u, z)
            t += dt

        # 11.0 because we have a small additive noise of 0.1
        self.assertTrue(np.allclose(x, 11.0 * np.ones_like(x)), f"got {x}")

    def test_noisy_double_integrator(self):
        from systems.double_integrator import DoubleIntegrator

        x = np.zeros((10, 4))
        u = np.ones((10, 2)) * 0.1
        z = np.ones((10, 4)) * 0.1

        T = 10.0
        dt = 0.1

        f = AdditiveBoundedUncertainty(base_system=DoubleIntegrator())

        t = dt
        while t < T:
            x = x + dt * f(x, u, z)
            t += dt

        self.assertTrue(
            np.allclose(x[:, :2], 10.9 * np.ones_like(x[:, :2])),
            f"got positions {x[:, :2]}",
        )
        self.assertTrue(
            np.allclose(x[:, 2:], 2.0 * np.ones_like(x[:, 2:])),
            f"got velocities {x[:, 2:]}",
        )

    def test_noisy_single_integrator_z3(self):
        from systems.single_integrator import SingleIntegrator

        state_vars = ["x", "y"]
        input_vars = ["vx", "vy"]
        uncertain_vars = ["zx", "zy"]

        x = [z3.Real(var) for var in state_vars]
        u = [z3.Real(var) for var in input_vars]
        z = [z3.Real(var) for var in uncertain_vars]

        f = AdditiveBoundedUncertainty(base_system=SingleIntegrator())

        xdot = f.f(x, u, z)

        self.assertTrue(
            str(xdot[0]) == f"{input_vars[0]} + {uncertain_vars[0]}",
            f"expected xdot = vx + zx, got {xdot[0]}",
        )
        self.assertTrue(
            str(xdot[1]) == f"{input_vars[1]} + {uncertain_vars[1]}",
            f"expected ydot = vy + zy, got {xdot[1]}",
        )

    def test_only_nominal_flag_numerical(self):
        """
        Check that simulating the uncertain system with only_nominal=True
        is actually equivalent to simulating the nominal system.
        """
        from systems.single_integrator import SingleIntegrator

        x = np.zeros((10, 2))
        u = np.ones((10, 2))
        z = np.ones((10, 2)) * 0.1

        T = 10.0
        dt = 0.1

        f = SingleIntegrator()
        fz = AdditiveBoundedUncertainty(base_system=SingleIntegrator())

        t = dt
        while t < T:
            dx1 = f(x, u)
            dx2 = fz(x, u, z, only_nominal=True)

            self.assertTrue(np.allclose(dx1, dx2), f"mismatch dx, got {dx1} and {dx2}")

            x = x + dt * dx1
            t += dt

        # 11.0 because we have a small additive noise of 0.1
        self.assertTrue(np.allclose(x, 10.0 * np.ones_like(x)), f"got {x}")

    def test_only_nominal_flag_symbolic(self):
        """
        Check that the symbolic expression for the uncertain dynamics with only_nominal=True
        actually matches the symbolic expression for the nominal dynamics.
        """
        from systems.single_integrator import SingleIntegrator

        state_vars = ["x", "y"]
        input_vars = ["vx", "vy"]
        uncertain_vars = ["zx", "zy"]

        x = [z3.Real(var) for var in state_vars]
        u = [z3.Real(var) for var in input_vars]
        z = [z3.Real(var) for var in uncertain_vars]

        f = SingleIntegrator()
        fz = AdditiveBoundedUncertainty(base_system=SingleIntegrator())

        xdot = f.f(x, u)
        xdotz = fz.f(x, u, z, only_nominal=True)

        self.assertTrue(
            str(xdot[0]) == str(xdotz[0]), f"expected xdot = vx, got {xdotz[0]}",
        )
        self.assertTrue(
            str(xdot[1]) == str(xdotz[1]), f"expected ydot = vy, got {xdotz[1]}",
        )

    def test_properties_and_methods(self):
        from systems.single_integrator import SingleIntegrator

        f = SingleIntegrator()
        fz = AdditiveBoundedUncertainty(base_system=SingleIntegrator())

        self.assertEqual(f.n_vars, fz.n_vars)
        self.assertEqual(f.n_controls, fz.n_controls)
        self.assertEqual(f.id, fz.id)

        x = torch.rand((10, f.n_vars, 1))
        self.assertTrue(torch.isclose(f.fx_torch(x), fz.fx_torch(x)).all())
        self.assertTrue(torch.isclose(f.gx_torch(x), fz.gx_torch(x)).all())

        sx = z3.Reals("x y")
        self.assertTrue([check_smt_equivalence(f1, f2) for f1, f2 in zip(f.fx_smt(sx), fz.fx_smt(sx))])
        self.assertTrue([check_smt_equivalence(f1, f2) for f1, f2 in zip(f.gx_smt(sx), fz.gx_smt(sx))])


