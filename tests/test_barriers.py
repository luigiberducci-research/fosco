import unittest

import numpy as np
import torch

from barriers import make_barrier, make_compensator
from fosco.models import TorchSymDiffModel, TorchSymModel
from fosco.systems import make_system
from fosco.systems.uncertainty import add_uncertainty


class TestBarriers(unittest.TestCase):
    def test_single_integrator_cbf_torch(self):
        system = make_system("SingleIntegrator")()

        cbf = make_barrier(system=system)
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

        cbf = make_barrier(system=system)
        assert isinstance(
            cbf, TorchSymDiffModel
        ), f"expected TorchSymModel, got {type(cbf)}"

        # test symbolic translation
        sx = VerifierZ3.new_vars(system.n_vars, base="x")
        hx, constr, hx_vars = cbf.forward_smt(x=sx)
        assert isinstance(hx, Z3SYMBOL), f"expected z3.ArithRef, got {type(hx)}"

        dhdx, constr, dhdx_vars = cbf.gradient_smt(x=sx)
        self.assertTrue(
            isinstance(dhdx, np.ndarray), f"expected np array, got {type(dhdx)}"
        )
        self.assertTrue(
            all(isinstance(dhdxi, Z3SYMBOL) for dhdxi in dhdx[0]),
            f"expected list of z3.ArithRef, got {type(dhdx[0])}",
        )
        self.assertTrue(
            all(isinstance(v, Z3SYMBOL) for v in dhdx_vars),
            f"expected list of z3.ArithRef, got {dhdx_vars}",
        )
        self.assertTrue(
            len(set(dhdx_vars)) == len(dhdx_vars),
            msg=f"expected unique variables in sig_vars, got duplicates {dhdx_vars}",
        )

    def test_single_integrator_sigma_torch(self):
        system = make_system("SingleIntegrator")()
        system = add_uncertainty("AdditiveBounded", system=system)

        sigma = make_compensator(system=system)
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

        system = make_system("SingleIntegrator")()
        system = add_uncertainty("AdditiveBounded", system=system)

        sigma = make_compensator(system=system)
        assert isinstance(
            sigma, TorchSymModel
        ), f"expected TorchSymModel, got {type(sigma)}"

        # test symbolic translation
        sx = VerifierZ3.new_vars(system.n_vars, base="x")
        sig, constr, sig_vars = sigma.forward_smt(x=sx)
        self.assertTrue(
            isinstance(sig, Z3SYMBOL), f"expected list of z3.ArithRef, got {type(sig)}"
        )
        self.assertTrue(
            all(isinstance(v, Z3SYMBOL) for v in sig_vars),
            f"expected list of z3.ArithRef, got {sig_vars}",
        )
        self.assertTrue(
            len(set(sig_vars)) == len(sig_vars),
            msg=f"expected unique variables in sig_vars, got duplicates {sig_vars}",
        )
        self.assertTrue(
            not hasattr(sigma, "gradient"),
            msg="compensator doesn't have gradient method",
        )
        self.assertTrue(
            not hasattr(sigma, "gradient_smt"),
            msg="compensator doesn't have gradient_smt method",
        )
