import time
import unittest


from fosco.common.consts import VerifierType
from fosco.verifier import make_verifier, Verifier
from fosco.verifier.verifier import SYMBOL


class TestVerifier(unittest.TestCase):
    def test_simple_constraints(self):
        verifier_fn = make_verifier(type=VerifierType.Z3)

        def constraint_gen(verif: Verifier, C: SYMBOL, sigma, dC: SYMBOL, *args):
            yield {"sat": C >= 0.0}

        def constraint_gen2(verif: Verifier, C: SYMBOL, sigma, dC: SYMBOL, *args):
            And_ = verif.solver_fncts()["And"]
            yield {"unsat": And_(C >= 0.0, C < 0)}

        vars = verifier_fn.new_vars(n=1)
        verifier = verifier_fn(
            solver_vars=vars,
            constraints_method=constraint_gen,
            solver_timeout=10,
            n_counterexamples=1,
        )
        verifier2 = verifier_fn(
            solver_vars=vars,
            constraints_method=constraint_gen2,
            solver_timeout=10,
            n_counterexamples=1,
        )

        C = vars[0] + 1.0
        dC = vars[0] + 6.0
        results, elapsed_time = verifier.verify(
            V_symbolic=C,
            V_symbolic_constr=[],
            Vdot_symbolic=dC,
            Vdot_symbolic_constr=[],
            sigma_symbolic=None,
            sigma_symbolic_constr=[],
            Vdotz_symbolic=None,
            Vdotz_symbolic_constr=[],
        )
        results2, elapsed_time = verifier2.verify(
            V_symbolic=C,
            V_symbolic_constr=[],
            Vdot_symbolic=dC,
            Vdot_symbolic_constr=[],
            sigma_symbolic=None,
            sigma_symbolic_constr=[],
            Vdotz_symbolic=None,
            Vdotz_symbolic_constr=[],
        )

        self.assertTrue(
            len(results["cex"]["sat"]) > 0,
            "expected counterexample for any x > -1, got none",
        )
        self.assertTrue(
            len(results2["cex"]["unsat"]) == 0,
            f"expected no counterexample, got {results2['cex']['unsat']}",
        )

    def test_new_vars_exceptions(self):
        verifier_fn = make_verifier(type=VerifierType.Z3)

        with self.assertRaises(AssertionError):
            # cannot provide both n and var_names
            verifier_fn.new_vars(n=3, var_names=["x1", "x2"])

        with self.assertRaises(AssertionError):
            # cannot provide duplicate variables
            verifier_fn.new_vars(var_names=["x1", "x2", "x1"])

        with self.assertRaises(AssertionError):
            # must provide at least one n or var_names
            verifier_fn.new_vars(n=None, var_names=None)

    def test_simple_constraints_dreal(self):
        verifier_fn = make_verifier(type=VerifierType.DREAL)

        def constraint_gen(verif: Verifier, C: SYMBOL, sigma, dC: SYMBOL, *args):
            yield {"sat": C >= 0.0}

        def constraint_gen2(verif: Verifier, C: SYMBOL, sigma, dC: SYMBOL, *args):
            And_ = verif.solver_fncts()["And"]
            yield {"unsat": And_(C >= 0.0, C < 0)}

        vars = verifier_fn.new_vars(n=1)
        verifier = verifier_fn(
            solver_vars=vars,
            constraints_method=constraint_gen,
            solver_timeout=10,
            n_counterexamples=1,
        )
        verifier2 = verifier_fn(
            solver_vars=vars,
            constraints_method=constraint_gen2,
            solver_timeout=10,
            n_counterexamples=1,
        )

        C = vars[0] + 1.0
        dC = vars[0] + 6.0
        results, elapsed_time = verifier.verify(
            V_symbolic=C,
            V_symbolic_constr=[],
            Vdot_symbolic=dC,
            Vdot_symbolic_constr=[],
            sigma_symbolic=None,
            sigma_symbolic_constr=[],
            Vdotz_symbolic=None,
            Vdotz_symbolic_constr=[],
        )
        results2, elapsed_time = verifier2.verify(
            V_symbolic=C,
            V_symbolic_constr=[],
            Vdot_symbolic=dC,
            Vdot_symbolic_constr=[],
            sigma_symbolic=None,
            sigma_symbolic_constr=[],
            Vdotz_symbolic=None,
            Vdotz_symbolic_constr=[],
        )

        self.assertTrue(
            len(results["cex"]["sat"]) > 0,
            "expected counterexample for any x > -1, got none",
        )
        self.assertTrue(
            len(results2["cex"]["unsat"]) == 0,
            f"expected no counterexample, got {results2['cex']['unsat']}",
        )

    def test_timeout_dreal(self):
        verifier_fn = make_verifier(type=VerifierType.DREAL)
        vars = verifier_fn.new_vars(n=1)
        timeout_s = 1

        def constraint_gen(verif: Verifier, C: SYMBOL, sigma, dC: SYMBOL, *args):
            # force this to take more than timeout_s to check if timeout mechanism works
            yield {"sat": C >= 0.0}

        verifier = verifier_fn(
            solver_vars=vars,
            constraints_method=constraint_gen,
            solver_timeout=timeout_s,
            n_counterexamples=1,
        )

        C = vars[0] + 1.0
        dC = vars[0] + 6.0
        results, elapsed_time = verifier.verify(
            V_symbolic=C,
            V_symbolic_constr=[],
            Vdot_symbolic=dC,
            Vdot_symbolic_constr=[],
            sigma_symbolic=None,
            sigma_symbolic_constr=[],
            Vdotz_symbolic=None,
            Vdotz_symbolic_constr=[],
        )

        self.assertTrue(
            elapsed_time <= timeout_s,
            f"expected verifier to finish within the time limit, got elapsed time {elapsed_time} > {timeout_s}"
        )

    def test_timeout_z3(self):
        verifier_fn = make_verifier(type=VerifierType.Z3)
        vars = verifier_fn.new_vars(n=1)
        timeout_s = 1

        def constraint_gen(verif: Verifier, C: SYMBOL, sigma, dC: SYMBOL, *args):
            # force this to take more than timeout_s to check if timeout mechanism works
            yield {"sat": C >= 0.0}

        verifier = verifier_fn(
            solver_vars=vars,
            constraints_method=constraint_gen,
            solver_timeout=timeout_s,
            n_counterexamples=1,
        )

        C = vars[0] + 1.0
        dC = vars[0] + 6.0
        results, elapsed_time = verifier.verify(
            V_symbolic=C,
            V_symbolic_constr=[],
            Vdot_symbolic=dC,
            Vdot_symbolic_constr=[],
            sigma_symbolic=None,
            sigma_symbolic_constr=[],
            Vdotz_symbolic=None,
            Vdotz_symbolic_constr=[],
        )

        self.assertTrue(
            elapsed_time <= timeout_s,
            f"expected verifier to finish within the time limit, got elapsed time {elapsed_time} > {timeout_s}"
        )
