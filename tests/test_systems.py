import unittest

import numpy as np
import torch
import z3

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
        from fosco.verifier.z3_verifier import VerifierZ3

        x = VerifierZ3.new_vars(2, base="x")
        u = VerifierZ3.new_vars(2, base="u")

        f = SingleIntegrator()

        xdot = f.f(x, u)

        self.assertTrue(
            check_smt_equivalence(xdot[0], u[0]), f"expected xdot = vx, got {xdot[0]}"
        )
        self.assertTrue(
            check_smt_equivalence(xdot[1], u[1]), f"expected ydot = vy, got {xdot[1]}"
        )


class TestUncertainControlAffineDynamicalSystem(unittest.TestCase):
    def test_noisy_single_integrator(self):
        from systems.single_integrator import SingleIntegrator
        from systems.uncertainty.additive_bounded import AdditiveBounded

        x = np.zeros((10, 2))
        u = np.ones((10, 2))
        z = np.ones((10, 2)) * 0.1

        T = 10.0
        dt = 0.1

        f = AdditiveBounded(system=SingleIntegrator())

        t = dt
        while t < T:
            x = x + dt * f(x, u, z)
            t += dt

        # 11.0 because we have a small additive noise of 0.1
        self.assertTrue(np.allclose(x, 11.0 * np.ones_like(x)), f"got {x}")

    def test_noisy_double_integrator(self):
        from systems.double_integrator import DoubleIntegrator
        from systems.uncertainty.additive_bounded import AdditiveBounded

        x = np.zeros((10, 4))
        u = np.ones((10, 2)) * 0.1
        z = np.ones((10, 4)) * 0.1

        T = 10.0
        dt = 0.1

        f = AdditiveBounded(system=DoubleIntegrator())

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
        from systems.uncertainty.additive_bounded import AdditiveBounded
        from fosco.verifier.z3_verifier import VerifierZ3

        x = VerifierZ3.new_vars(2, base="x")
        u = VerifierZ3.new_vars(2, base="u")
        z = VerifierZ3.new_vars(2, base="z")

        f = AdditiveBounded(system=SingleIntegrator())

        xdot = f.f(x, u, z)

        self.assertTrue(
            check_smt_equivalence(xdot[0], u[0] + z[0]),
            f"expected xdot = vx + zx, got {xdot[0]}",
        )
        self.assertTrue(
            check_smt_equivalence(xdot[1], u[1] + z[1]),
            f"expected ydot = vy + zy, got {xdot[1]}",
        )

    def test_only_nominal_flag_numerical(self):
        """
        Check that simulating the uncertain system with only_nominal=True
        is actually equivalent to simulating the nominal system.
        """
        from systems.single_integrator import SingleIntegrator
        from systems.uncertainty.additive_bounded import AdditiveBounded

        x = np.zeros((10, 2))
        u = np.ones((10, 2))
        z = np.ones((10, 2)) * 0.1

        T = 10.0
        dt = 0.1

        f = SingleIntegrator()
        fz = AdditiveBounded(system=SingleIntegrator())

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
        from systems.uncertainty.additive_bounded import AdditiveBounded
        from fosco.verifier.z3_verifier import VerifierZ3

        x = VerifierZ3.new_vars(2, base="x")
        u = VerifierZ3.new_vars(2, base="u")
        z = VerifierZ3.new_vars(2, base="z")

        f = SingleIntegrator()
        fz = AdditiveBounded(system=SingleIntegrator())

        xdot = f.f(x, u)
        xdotz = fz.f(x, u, z, only_nominal=True)

        self.assertTrue(
            check_smt_equivalence(xdot[0], xdotz[0]),
            f"expected xdot = vx, got {xdotz[0]}",
        )
        self.assertTrue(
            check_smt_equivalence(xdot[1], xdotz[1]),
            f"expected ydot = vy, got {xdotz[1]}",
        )

    def test_properties_and_methods(self):
        from systems.single_integrator import SingleIntegrator
        from systems.uncertainty.additive_bounded import AdditiveBounded
        from fosco.verifier.z3_verifier import VerifierZ3

        f = SingleIntegrator()
        fz = AdditiveBounded(system=SingleIntegrator())

        self.assertEqual(f.n_vars, fz.n_vars)
        self.assertEqual(f.n_controls, fz.n_controls)

        x = torch.rand((10, f.n_vars, 1))
        self.assertTrue(torch.isclose(f.fx_torch(x), fz.fx_torch(x)).all())
        self.assertTrue(torch.isclose(f.gx_torch(x), fz.gx_torch(x)).all())

        sx = VerifierZ3.new_vars(f.n_vars, base="x")
        # check equivalence for each term in the array of symbolic expressions, fx.shape = (n_vars,)
        self.assertTrue(
            all(
                [
                    f1 == f2 or check_smt_equivalence(f1, f2)
                    for f1, f2 in zip(f.fx_smt(sx), fz.fx_smt(sx))
                ]
            )
        )
        # check equivalence for each term in the matrix of symbolic expressions, gx.shape = (n_vars, n_controls)
        self.assertTrue(
            all(
                [
                    f1 == f2 or check_smt_equivalence(f1, f2)
                    for fv1, fv2 in zip(f.gx_smt(sx), fz.gx_smt(sx))
                    for f1, f2 in zip(fv1, fv2)
                ]
            )
        )

        # check incremental id assignment
        id1 = f.id
        id2 = fz.id

        self.assertTrue(
            isinstance(id1, str), f"expected id1 to be a string, got {type(id1)}"
        )
        self.assertTrue(
            isinstance(id2, str), f"expected id2 to be a string, got {type(id2)}"
        )
        self.assertTrue(id1 != id2, f"expected id1 != id2, got {id1} == {id2}")
        self.assertTrue(id1 in id2, f"expected id1 in id2, got {id1} not in {id2}")

    def test_unicycle(self):
        from systems import make_system

        debug_plot = False

        f = make_system(system_id="Unicycle")()
        self.assertEqual(f.n_vars, 3)
        self.assertEqual(f.n_controls, 2)

        n = 10
        x = np.zeros((n, 3))
        if n > 1:
            u = np.array([[1.0, -1.0 + 2 * i / (n - 1)] for i in range(n)])
        else:
            u = np.array([[1.0, 0.0]])

        T = 2.0
        dt = 0.1
        t = dt

        xs = [x]
        while t < T:
            x = x + dt * f(x, u)
            t += dt
            xs.append(x)

        if debug_plot:
            import matplotlib.pyplot as plt

            xs = np.array(xs)
            plt.plot(xs[:, :, 0], xs[:, :, 1])
            plt.show()

    def test_unicycle_symbolic(self):
        from systems import make_system
        from fosco.verifier.dreal_verifier import VerifierDR

        f = make_system(system_id="Unicycle")()

        fns = VerifierDR.solver_fncts()
        x = VerifierDR.new_vars(f.n_vars, base="x")
        u = VerifierDR.new_vars(f.n_controls, base="u")

        xdot = f.f(x, u)
        self.assertTrue(
            xdot[0] == u[0] * fns["cos"](x[2]),
            f"expected xdot[0] == u[0] * Cos(x[2]), got {xdot[0]}",
        )
        self.assertTrue(
            xdot[1] == u[0] * fns["sin"](x[2]),
            f"expected xdot[1] == u[0] * Sin(x[2]), got {xdot[1]}",
        )
        self.assertTrue(xdot[2] == u[1], f"expected xdot[2] == u[1], got {xdot[2]}")
