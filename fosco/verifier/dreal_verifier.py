import multiprocessing.context
import timeit
from multiprocessing.pool import ThreadPool
from typing import Callable

import dreal

from fosco.common.utils import contains_object
from fosco.verifier import Verifier
from fosco.verifier.types import DRSYMBOL


class VerifierDR(Verifier):
    INFINITY: float = 1e300
    SECOND_CHANCE_BOUND: float = 1e3

    def _assert_state(self) -> None:
        super()._assert_state()
        assert all([isinstance(x, DRSYMBOL) for x in self.xs]), f"Expected z3 variables, got {self.xs}"
        assert isinstance(self.constraints_method, Callable), f"Expected callable, got {self.constraints_method}"

    @staticmethod
    def new_vars(
        n: int | None = None, var_names: list[str] | None = None, base: str = "x"
    ) -> list[DRSYMBOL]:
        assert (
            n is not None or var_names is not None
        ), "Must provide either n or var_names"
        assert n is None or var_names is None, f"Cannot provide both n and var_names"
        assert var_names is None or len(var_names) == len(
            set(var_names)
        ), "var_names must contain unique identifiers"

        if var_names:
            return [dreal.Variable(var) for var in var_names]
        else:
            return [dreal.Variable(base + str(i)) for i in range(n)]

    @staticmethod
    def solver_fncts() -> dict[str, Callable]:
        return {
            "Sin": dreal.sin,
            "Cos": dreal.cos,
            "Exp": dreal.exp,
            "And": dreal.And,
            "Or": dreal.Or,
            "If": dreal.if_then_else,
            "Check": lambda x: contains_object(x, dreal.Variable),
            "Not": dreal.Not,
            "False": dreal.Formula.FALSE(),
            "True": dreal.Formula.TRUE(),
            "Substitute": lambda expr, replacement: expr.Substitute(
                replacement[0], replacement[1]
            ),
            "RealVal": lambda real: real,
        }

    @staticmethod
    def new_solver():
        return None

    @staticmethod
    def is_sat(res) -> bool:
        return isinstance(res, dreal.Box)

    def is_unsat(self, res) -> bool:
        bounds_not_ok = not self.within_bounds(res)
        return res is None or bounds_not_ok

    def within_bounds(self, res) -> bool:
        left, right = -self.INFINITY, self.INFINITY
        return isinstance(res, dreal.Box) and all(
            left < interval.mid() < right for x, interval in res.items()
        )

    def _solver_solve(self, solver, fml):
        timedout = False

        pool = ThreadPool(processes=1)
        async_result = pool.apply_async(dreal.CheckSatisfiability, args=(fml, 0.0001))

        try:
            t0 = timeit.default_timer()
            res = async_result.get(timeout=self._solver_timeout)
        except multiprocessing.context.TimeoutError:
            res = None
            timedout = True
            self._logger.info(
                f"Timed out while solving, kill after {timeit.default_timer() - t0} sec"
            )
            return res, timedout

        if self.is_sat(res) and not self.within_bounds(res):
            And_ = self.solver_fncts()["And"]
            self._logger.info("Second chance bound used")
            new_bound = self.SECOND_CHANCE_BOUND
            fml = And_(fml, *(And_(x < new_bound, x > -new_bound) for x in self.xs))
            res = dreal.CheckSatisfiability(fml, 0.0001)

        return res, timedout

    def _solver_model(self, solver, res):
        assert self.is_sat(res)
        return res

    def _model_result(self, solver, model, x, idx):
        return float(model[idx].mid())

    @staticmethod
    def replace_point(expr, ver_vars, point):
        try:
            replacements = {
                ver_vars[i, 0]: float(point[i, 0]) for i in range(len(ver_vars))
            }
        except TypeError:
            replacements = {
                ver_vars[i]: float(point[i, 0]) for i in range(len(ver_vars))
            }

        return expr.Substitute(replacements)

    def pretty_formula(self, fml) -> str:
        return str(fml)
