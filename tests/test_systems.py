import unittest

import numpy as np
import torch

from fosco.systems import make_system
from tests.test_translator import check_smt_equivalence


class TestControlAffineDynamicalSystem(unittest.TestCase):
    def test_single_integrator(self):
        f = make_system(system_id="SingleIntegrator")()

        x = np.zeros((10, 2))
        u = np.ones((10, 2))
        T = 10.0
        dt = 0.1

        t = dt
        while t < T:
            x = x + dt * f(x, u)
            t += dt

        self.assertTrue(np.allclose(x, 10.0 * np.ones_like(x)), f"got {x}")

    def test_single_integrator_z3(self):
        from fosco.verifier.z3_verifier import VerifierZ3

        f = make_system(system_id="SingleIntegrator")()

        x = VerifierZ3.new_vars(2, base="x")
        u = VerifierZ3.new_vars(2, base="u")

        xdot = f.f(x, u)

        self.assertTrue(
            check_smt_equivalence(xdot[0], u[0]), f"expected xdot = vx, got {xdot[0]}"
        )
        self.assertTrue(
            check_smt_equivalence(xdot[1], u[1]), f"expected ydot = vy, got {xdot[1]}"
        )


class TestUncertainControlAffineDynamicalSystem(unittest.TestCase):
    def test_noisy_single_integrator(self):
        from fosco.systems.uncertainty import AdditiveBounded

        f = make_system(system_id="SingleIntegrator")
        f = AdditiveBounded(system=f())

        x = np.zeros((10, 2))
        u = np.ones((10, 2))
        z = np.ones((10, 2)) * 0.1

        T = 10.0
        dt = 0.1

        t = dt
        while t < T:
            x = x + dt * f(x, u, z)
            t += dt

        # 11.0 because we have a small additive noise of 0.1
        self.assertTrue(np.allclose(x, 11.0 * np.ones_like(x)), f"got {x}")

    def test_noisy_double_integrator(self):
        from fosco.systems.uncertainty import AdditiveBounded

        x = np.zeros((10, 4))
        u = np.ones((10, 2)) * 0.1
        z = np.ones((10, 4)) * 0.1

        T = 10.0
        dt = 0.1

        f = make_system(system_id="DoubleIntegrator")
        f = AdditiveBounded(system=f())

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
        from fosco.systems.uncertainty import AdditiveBounded
        from fosco.verifier.z3_verifier import VerifierZ3

        x = VerifierZ3.new_vars(2, base="x")
        u = VerifierZ3.new_vars(2, base="u")
        z = VerifierZ3.new_vars(2, base="z")

        f = make_system(system_id="SingleIntegrator")
        f = AdditiveBounded(system=f())

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
        from fosco.systems.uncertainty import AdditiveBounded

        x = np.zeros((10, 2))
        u = np.ones((10, 2))
        z = np.ones((10, 2)) * 0.1

        T = 10.0
        dt = 0.1

        f = make_system(system_id="SingleIntegrator")()
        fz = AdditiveBounded(system=f)

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
        from fosco.systems.uncertainty import AdditiveBounded
        from fosco.verifier.z3_verifier import VerifierZ3

        x = VerifierZ3.new_vars(2, base="x")
        u = VerifierZ3.new_vars(2, base="u")
        z = VerifierZ3.new_vars(2, base="z")

        f = make_system(system_id="SingleIntegrator")()
        fz = AdditiveBounded(system=f)

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
        from fosco.systems.uncertainty import AdditiveBounded
        from fosco.verifier.z3_verifier import VerifierZ3
        from fosco.systems import make_system

        system_id = "SingleIntegrator"
        f = make_system(system_id=system_id)()
        self.assertEqual(system_id, f.id)
        fz = AdditiveBounded(system=f)

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

    def test_unicycle_numpy(self):
        debug_plot = False

        system_id = "Unicycle"
        f = make_system(system_id=system_id)()
        self.assertEqual(f.n_vars, 3)
        self.assertEqual(f.n_controls, 2)
        self.assertEqual(system_id, f.id)

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
        xs = np.array(xs)

        if debug_plot:
            import matplotlib.pyplot as plt

            plt.plot(xs[:, :, 0], xs[:, :, 1])
            plt.show()

        first_traj = xs[:, 0, :]
        last_traj = xs[:, -1, :]
        self.assertTrue(
            np.allclose(first_traj[:, 0], last_traj[:, 0]),
            f"expectd same trajectory for x coord, got {first_traj[:, 0]} and {last_traj[:, 0]}",
        )
        self.assertTrue(
            np.allclose(first_traj[:, 1], -last_traj[:, 1]),
            f"expectd mirrored trajectory for y coord, got {first_traj[:, 1]} and {last_traj[:, 1]}",
        )
        self.assertTrue(
            np.allclose(first_traj[:, 2], -last_traj[:, 2]),
            f"expectd mirrored trajectory for theta coord, got {first_traj[:, 1]} and {last_traj[:, 1]}",
        )

    def test_unicycle_torch(self):
        debug_plot = False

        system_id = "Unicycle"
        f = make_system(system_id=system_id)()
        self.assertEqual(f.n_vars, 3)
        self.assertEqual(f.n_controls, 2)
        self.assertEqual(system_id, f.id)

        n = 10
        x = torch.zeros((n, 3))
        u = np.array([[1.0, -1.0 + 2 * i / (n - 1)] for i in range(n)])

        T = 2.0
        dt = 0.1
        t = dt

        xs = [x]
        while t < T:
            x = x + dt * f(x, u)
            t += dt
            xs.append(x)
        xs = torch.stack(xs)

        if debug_plot:
            import matplotlib.pyplot as plt

            nxs = np.array(xs)
            plt.plot(nxs[:, :, 0], nxs[:, :, 1])
            plt.show()

        first_traj = xs[:, 0, :]
        last_traj = xs[:, -1, :]
        self.assertTrue(
            torch.allclose(first_traj[:, 0], last_traj[:, 0]),
            f"expectd same trajectory for x coord, got {first_traj[:, 0]} and {last_traj[:, 0]}",
        )
        self.assertTrue(
            torch.allclose(first_traj[:, 1], -last_traj[:, 1]),
            f"expectd mirrored trajectory for y coord, got {first_traj[:, 1]} and {last_traj[:, 1]}",
        )
        self.assertTrue(
            torch.allclose(first_traj[:, 2], -last_traj[:, 2]),
            f"expectd mirrored trajectory for theta coord, got {first_traj[:, 1]} and {last_traj[:, 1]}",
        )

    def test_unicycle_symbolic(self):
        from fosco.verifier.dreal_verifier import VerifierDR

        system_id = "Unicycle"
        f = make_system(system_id=system_id)()
        self.assertEqual(system_id, f.id)

        fns = VerifierDR.solver_fncts()
        x = VerifierDR.new_vars(f.n_vars, base="x")
        u = VerifierDR.new_vars(f.n_controls, base="u")

        xdot = f.f(x, u)
        self.assertTrue(
            xdot[0] == u[0] * fns["Cos"](x[2]),
            f"expected xdot[0] == u[0] * Cos(x[2]), got {xdot[0]}",
        )
        self.assertTrue(
            xdot[1] == u[0] * fns["Sin"](x[2]),
            f"expected xdot[1] == u[0] * Sin(x[2]), got {xdot[1]}",
        )
        self.assertTrue(xdot[2] == u[1], f"expected xdot[2] == u[1], got {xdot[2]}")
        self.assertTrue(xdot[2] == u[1], f"expected xdot[2] == u[1], got {xdot[2]}")

    def test_unicycle_acc_numpy(self):
        debug_plot = False

        system_id = "UnicycleAcc"
        f = make_system(system_id=system_id)()
        self.assertEqual(f.n_vars, 4)
        self.assertEqual(f.n_controls, 2)
        self.assertEqual(system_id, f.id)

        n = 10
        x = np.zeros((n, f.n_vars))
        u = np.array([[1.0, -1.0 + 2 * i / (n - 1)] for i in range(n)])

        T = 2.0
        dt = 0.1
        t = dt

        xs = [x]
        while t < T:
            x = x + dt * f(x, u)
            t += dt
            xs.append(x)
        xs = np.array(xs)

        if debug_plot:
            import matplotlib.pyplot as plt

            plt.plot(xs[:, :, 0], xs[:, :, 1])
            plt.show()

        first_traj = xs[:, 0, :]
        last_traj = xs[:, -1, :]
        self.assertTrue(
            np.allclose(first_traj[:, 0], last_traj[:, 0]),
            f"expectd same trajectory for x coord, got {first_traj[:, 0]} and {last_traj[:, 0]}",
        )
        self.assertTrue(
            np.allclose(first_traj[:, 1], -last_traj[:, 1]),
            f"expectd mirrored trajectory for y coord, got {first_traj[:, 1]} and {last_traj[:, 1]}",
        )
        self.assertTrue(
            np.allclose(first_traj[:, 2], last_traj[:, 2]),
            f"expectd mirrored trajectory for velocity coord, got {first_traj[:, 1]} and {last_traj[:, 1]}",
        )
        self.assertTrue(
            np.allclose(first_traj[:, 3], -last_traj[:, 3]),
            f"expectd mirrored trajectory for theta coord, got {first_traj[:, 1]} and {last_traj[:, 1]}",
        )

    def test_unicycle_acc_torch(self):
        debug_plot = False

        system_id = "UnicycleAcc"
        f = make_system(system_id=system_id)()
        self.assertEqual(f.n_vars, 4)
        self.assertEqual(f.n_controls, 2)
        self.assertEqual(system_id, f.id)

        n = 10
        x = torch.zeros((n, f.n_vars))
        u = torch.tensor([[1.0, -1.0 + 2 * i / (n - 1)] for i in range(n)])

        T = 2.0
        dt = 0.1
        t = dt

        xs = [x]
        while t < T:
            x = x + dt * f(x, u)
            t += dt
            xs.append(x)
        xs = torch.stack(xs)

        if debug_plot:
            import matplotlib.pyplot as plt

            nxs = np.array(xs)
            plt.plot(nxs[:, :, 0], nxs[:, :, 1])
            plt.show()

        first_traj = xs[:, 0, :]
        last_traj = xs[:, -1, :]
        self.assertTrue(
            torch.allclose(first_traj[:, 0], last_traj[:, 0]),
            f"expectd same trajectory for x coord, got {first_traj[:, 0]} and {last_traj[:, 0]}",
        )
        self.assertTrue(
            torch.allclose(first_traj[:, 1], -last_traj[:, 1]),
            f"expectd mirrored trajectory for y coord, got {first_traj[:, 1]} and {last_traj[:, 1]}",
        )
        self.assertTrue(
            torch.allclose(first_traj[:, 2], last_traj[:, 2]),
            f"expectd mirrored trajectory for velocity coord, got {first_traj[:, 1]} and {last_traj[:, 1]}",
        )
        self.assertTrue(
            torch.allclose(first_traj[:, 3], -last_traj[:, 3]),
            f"expectd mirrored trajectory for theta coord, got {first_traj[:, 1]} and {last_traj[:, 1]}",
        )

    def test_discrete_time(self):
        from fosco.systems import SYSTEM_REGISTRY
        from fosco.systems.discrete_time.system_dt import EulerDTSystem
        from fosco.systems import ControlAffineDynamics

        dt = 0.1
        for system_id in SYSTEM_REGISTRY:
            system = make_system(system_id=system_id)()
            dt_system = EulerDTSystem(system=system, dt=dt)

            self.assertTrue(isinstance(dt_system, ControlAffineDynamics))
            self.assertTrue(system.id in dt_system.id)
            self.assertTrue(f"dt{dt}" in dt_system.id)
            self.assertTrue(all([a == b for a, b in zip(system.vars, dt_system.vars)]))
            self.assertTrue(all([a == b for a, b in zip(system.controls, dt_system.controls)]))

