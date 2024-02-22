import unittest

import numpy as np
import torch


from barriers import make_barrier
from models.torchsym import TorchSymDiffModel, TorchSymModel
from systems import make_system
from systems.uncertainty import add_uncertainty


class TestBarriers(unittest.TestCase):
    def test_single_integrator_cbf_torch(self):
        system = make_system("SingleIntegrator")()

        barrier_dict = make_barrier(system=system)
        cbf = barrier_dict["barrier"]
        assert isinstance(
            cbf, TorchSymDiffModel
        ), f"expected TorchSymModel, got {type(cbf)}"

        x = torch.rand((1000, system.n_vars))

        # test numerical forward/gradient
        hx = cbf(x=x)
        self.assertTrue(
            len(hx.shape) == 1 and hx.shape[0] == x.shape[0],
            f"expected shape (1000,), got {hx.shape}",
        )

        dhdx = cbf.gradient(x=x)
        self.assertTrue(
            dhdx.shape == (1000, system.n_vars),
            f"expected shape (1000, {system.n_vars}), got {dhdx.shape}",
        )

    def test_single_integrator_cbf_smt(self):
        from fosco.verifier.z3_verifier import VerifierZ3
        from fosco.verifier.z3_verifier import Z3SYMBOL

        system = make_system("SingleIntegrator")()

        barrier_dict = make_barrier(system=system)
        cbf = barrier_dict["barrier"]
        assert isinstance(
            cbf, TorchSymDiffModel
        ), f"expected TorchSymModel, got {type(cbf)}"

        # test symbolic translation
        sx = VerifierZ3.new_vars(system.n_vars, base="x")
        hx, constr = cbf.forward_smt(x=sx)
        assert isinstance(hx, Z3SYMBOL), f"expected z3.ArithRef, got {type(hx)}"

        dhdx, constr = cbf.gradient_smt(x=sx)
        self.assertTrue(
            isinstance(dhdx, np.ndarray), f"expected np array, got {type(dhdx)}"
        )
        self.assertTrue(
            all(isinstance(dhdxi, Z3SYMBOL) for dhdxi in dhdx[0]),
            f"expected list of z3.ArithRef, got {type(dhdx[0])}",
        )

    def test_single_integrator_sigma_torch(self):
        system_fn = make_system("SingleIntegrator")
        system_fn = add_uncertainty("AdditiveBounded", system_fn=system_fn)
        system = system_fn()

        barrier_dict = make_barrier(system=system)
        sigma = barrier_dict["compensator"]
        assert isinstance(
            sigma, TorchSymModel
        ), f"expected TorchSymModel, got {type(sigma)}"

        x = torch.rand((1000, system.n_vars))

        # test numerical forward/gradient
        sig = sigma(x=x)
        self.assertTrue(
            len(sig.shape) == 1 and sig.shape[0] == x.shape[0],
            f"expected shape (1000,), got {sig.shape}",
        )
        self.assertTrue(
            not hasattr(sigma, "gradient"),
            msg="compensator doesn't have gradient method",
        )
        self.assertTrue(
            not hasattr(sigma, "gradient_smt"),
            msg="compensator doesn't have gradient_smt method",
        )

    def test_single_integrator_sigma_smt(self):
        from fosco.verifier.z3_verifier import VerifierZ3
        from fosco.verifier.z3_verifier import Z3SYMBOL

        system_fn = make_system("SingleIntegrator")
        system_fn = add_uncertainty("AdditiveBounded", system_fn=system_fn)
        system = system_fn()

        barrier_dict = make_barrier(system=system)
        sigma = barrier_dict["compensator"]
        assert isinstance(
            sigma, TorchSymModel
        ), f"expected TorchSymModel, got {type(sigma)}"

        # test symbolic translation
        sx = VerifierZ3.new_vars(system.n_vars, base="x")
        sig, constr = sigma.forward_smt(x=sx)
        self.assertTrue(
            isinstance(sig, Z3SYMBOL), f"expected list of z3.ArithRef, got {type(sig)}"
        )
        self.assertTrue(
            not hasattr(sigma, "gradient"),
            msg="compensator doesn't have gradient method",
        )
        self.assertTrue(
            not hasattr(sigma, "gradient_smt"),
            msg="compensator doesn't have gradient_smt method",
        )
