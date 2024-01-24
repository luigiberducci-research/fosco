import unittest

import numpy as np
import torch
import z3

from barriers import make_barrier
from models.torchsym import TorchSymModel
from systems import make_system, add_uncertainty


class TestBarriers(unittest.TestCase):
    def test_single_integrator_cbf_torch(self):
        system = make_system("single_integrator")()

        barrier_dict = make_barrier(system=system)
        cbf = barrier_dict["barrier"]
        assert isinstance(cbf, TorchSymModel), f"expected TorchSymModel, got {type(cbf)}"

        x = torch.rand((1000, system.n_vars))

        # test numerical forward/gradient
        hx = cbf(x=x)
        self.assertTrue(len(hx.shape) == 1 and hx.shape[0] == x.shape[0],
                        f"expected shape (1000,), got {hx.shape}")

        dhdx = cbf.gradient(x=x)
        self.assertTrue(dhdx.shape == (1000, system.n_vars),
                        f"expected shape (1000, {system.n_vars}), got {dhdx.shape}")

    def test_single_integrator_cbf_smt(self):
        system = make_system("single_integrator")()

        barrier_dict = make_barrier(system=system)
        cbf = barrier_dict["barrier"]
        assert isinstance(cbf, TorchSymModel), f"expected TorchSymModel, got {type(cbf)}"

        # test symbolic translation
        sx = z3.Reals("x y")
        hx = cbf.forward_smt(x=sx)
        assert isinstance(hx, z3.ArithRef), f"expected z3.ArithRef, got {type(hx)}"

        dhdx = cbf.gradient_smt(x=sx)
        self.assertTrue(isinstance(dhdx, np.ndarray),
                        f"expected np array, got {type(dhdx)}")
        self.assertTrue(all(isinstance(dhdxi, z3.ArithRef) for dhdxi in dhdx[0]),
                        f"expected list of z3.ArithRef, got {type(dhdx[0])}")

    def test_single_integrator_sigma_torch(self):
        system_fn = make_system("single_integrator")
        system_fn = add_uncertainty("additive_bounded", system_fn=system_fn)
        system = system_fn()

        barrier_dict = make_barrier(system=system)
        sigma = barrier_dict["compensator"]
        assert isinstance(sigma, TorchSymModel), f"expected TorchSymModel, got {type(sigma)}"

        x = torch.rand((1000, system.n_vars))

        # test numerical forward/gradient
        sig = sigma(x=x)
        assert len(sig.shape) == 1 and sig.shape[0] == x.shape[0], f"expected shape (1000,), got {sig.shape}"

        with self.assertRaises(NotImplementedError, msg="expected NotImplementedError"):
            dhdx = sigma.gradient(x=x)

    def test_single_integrator_sigma_smt(self):
        system_fn = make_system("single_integrator")
        system_fn = add_uncertainty("additive_bounded", system_fn=system_fn)
        system = system_fn()

        barrier_dict = make_barrier(system=system)
        sigma = barrier_dict["compensator"]
        assert isinstance(sigma, TorchSymModel), f"expected TorchSymModel, got {type(sigma)}"

        # test symbolic translation
        sx = z3.Reals("x y")
        sig = sigma.forward_smt(x=sx)
        self.assertTrue(all(isinstance(sigi, z3.ArithRef) for sigi in sig),
                        f"expected list of z3.ArithRef, got {type(sig)}")

        with self.assertRaises(NotImplementedError, msg="expected NotImplementedError"):
            dhdx = sigma.gradient_smt(x=sx)



