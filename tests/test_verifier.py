import unittest


from fosco.common.consts import VerifierType
from fosco.verifier import make_verifier, Verifier
from fosco.verifier.verifier import SYMBOL


class TestVerifier(unittest.TestCase):
    def test_simple_constraints(self):
        import z3

        verifier_fn = make_verifier(type=VerifierType.Z3)

        def constraint_gen(verif: Verifier, C: SYMBOL, sigma, dC: SYMBOL, *args):
            yield {"sat": C >= 0.0}

        def constraint_gen2(verif: Verifier, C: SYMBOL, sigma, dC: SYMBOL, *args):
            yield {"unsat": z3.And(C >= 0.0, C < 0)}

        vars = verifier_fn.new_vars(n=1)
        verifier = verifier_fn(solver_vars=vars, constraints_method=constraint_gen,
                               solver_timeout=10, n_counterexamples=1)
        verifier2 = verifier_fn(solver_vars=vars, constraints_method=constraint_gen2,
                                solver_timeout=10, n_counterexamples=1)

        C = vars[0] + 1.0
        dC = vars[0] + 6.0
        results, elapsed_time = verifier.verify(
            V_symbolic=C, V_symbolic_constr=[],
            Vdot_symbolic=dC, Vdot_symbolic_constr=[],
            sigma_symbolic=None, sigma_symbolic_constr=[],
            Vdotz_symbolic=None, Vdotz_symbolic_constr=[]
        )
        results2, elapsed_time = verifier2.verify(
            V_symbolic=C, V_symbolic_constr=[],
            Vdot_symbolic=dC, Vdot_symbolic_constr=[],
            sigma_symbolic=None, sigma_symbolic_constr=[],
            Vdotz_symbolic=None, Vdotz_symbolic_constr=[]
        )

        self.assertTrue(
            len(results["cex"]["sat"]) > 0,
            "expected counterexample for any x > -1, got none",
        )
        self.assertTrue(
            len(results2["cex"]["unsat"]) == 0,
            f"expected no counterexample, got {results2['cex']['unsat']}",
        )
