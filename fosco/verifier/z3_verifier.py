import time
import timeit
from typing import Callable

import z3

from fosco.common.utils import contains_object
from fosco.verifier.types import Z3SYMBOL
from fosco.verifier.verifier import Verifier


class VerifierZ3(Verifier):
    def _assert_state(self) -> None:
        super()._assert_state()
        assert all(
            [isinstance(x, Z3SYMBOL) for x in self.xs]
        ), f"Expected z3 variables, got {self.xs}"
        assert isinstance(
            self.constraints_method, Callable
        ), f"Expected callable, got {self.constraints_method}"

    @staticmethod
    def new_vars(
        n: int | None = None, var_names: list[str] | None = None, base: str = "x"
    ) -> list[Z3SYMBOL]:
        assert (
            n is not None or var_names is not None
        ), "Must provide either n or var_names"
        assert n is None or var_names is None, f"Cannot provide both n and var_names"
        assert var_names is None or len(var_names) == len(
            set(var_names)
        ), "var_names must contain unique identifiers"

        if var_names:
            return [z3.Real(var) for var in var_names]
        else:
            return [z3.Real(base + str(i)) for i in range(n)]

    def new_solver(self):
        return z3.Solver()

    @staticmethod
    def check_type(x) -> bool:
        """
        :param x: any
        :returns: True if z3 compatible, else false
        """
        return contains_object(x, z3.ArithRef)

    @staticmethod
    def replace_point(expr, ver_vars, point):
        """
        :param expr: z3 expr
        :param z3_vars: z3 vars, matrix
        :param ctx: matrix of numerical values
        :return: value of V, Vdot in ctx
        """
        replacements = []
        for i in range(len(ver_vars)):
            try:
                replacements += [(ver_vars[i, 0], z3.RealVal(point[i, 0]))]
            except TypeError:
                replacements += [(ver_vars[i], z3.RealVal(point[i, 0]))]

        replaced = z3.substitute(expr, replacements)

        return z3.simplify(replaced)

    def is_sat(self, res) -> bool:
        return res == z3.sat

    def is_unsat(self, res) -> bool:
        return res == z3.unsat

    def _solver_solve(self, solver, fml):
        """
        :param fml:
        :param solver: z3 solver
        :return:
                res: sat if found ctx
                timedout: true if verification timed out
        """
        try:
            solver.set("timeout", max(1, self._solver_timeout * 1000))
        except:
            pass

        fml = z3.simplify(fml)
        if self._rounding > 0:
            fml = round_expr(fml, rounding=self._rounding)

        self._logger.debug(fml.sexpr())

        timer = timeit.default_timer()
        solver.add(fml)
        res = solver.check()
        timer = timeit.default_timer() - timer

        timedout = timer >= self._solver_timeout
        if timedout:
            self._logger.debug(fml.sexpr())
            self._logger.info(f"Timed out while solving, kill after {timer:.2f} sec")

        return res, timedout

    def _solver_model(self, solver, res):
        return solver.model()

    def _model_result(self, solver, model, x, i):
        try:
            return float(model[x].as_fraction())
        except AttributeError:
            try:
                return float(model[x].approx(10).as_fraction())
            except AttributeError:
                # no variable in model, eg. input in CBF unfeasible condition. return dummy 0.0
                return 0.0
        except TypeError:
            try:
                return float(model[x[0, 0]].as_fraction())
            except:  # when z3 finds non-rational numbers, prints them w/ '?' at the end --> approx 10 decimals
                return float(model[x[0, 0]].approx(10).as_fraction())
        except Exception as e:
            print("model type", type(model))
            print("model", model)
            print("x", x)
            print("i", i)
            raise e

    @staticmethod
    def solver_fncts() -> dict[str, Callable]:
        return {
            "RealVar": z3.Real,
            "And": z3.And,
            "Or": z3.Or,
            "If": z3.If,
            "Not": z3.Not,
            "False": False,
            "True": True,
            "Exists": z3.Exists,
            "ForAll": z3.ForAll,
            "Substitute": z3.substitute,
            "Check": lambda x: contains_object(x, z3.ArithRef),
            "RealVal": z3.RealVal,
            "Sqrt": z3.Sqrt,
        }

    def pretty_formula(fml) -> str:
        return str(z3.simplify(fml))


def round_expr(e: Z3SYMBOL, rounding: int) -> Z3SYMBOL:
    """
    Recursive conversion of coefficients to rounded values.

    Args:
        e:  z3 expression
        rounding: number of decimals to round to

    Returns:
        e: z3 expression with rounded coefficients
    """
    assert rounding > 0, "rounding must be > 0"

    # base case: rational coeff
    if z3.is_const(e) and hasattr(e, "as_fraction"):
        num, den = e.as_fraction().numerator, e.as_fraction().denominator
        return z3.RealVal(round(float(num) / float(den), rounding))

    # recursive case: non-const expr
    args = [round_expr(arg, rounding) for arg in e.children()]
    return e.decl()(*args)
