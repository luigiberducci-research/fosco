import logging
from abc import abstractmethod, ABC
from typing import Callable, Generator, Iterable, Any

import torch

from fosco.common.timing import timed
from fosco.logger import LOGGING_LEVELS
from fosco.verifier.types import SYMBOL

INF: float = 1e300


class Verifier(ABC):
    def __init__(
        self,
        constraints_method: Callable[..., Generator],
        solver_vars: list[SYMBOL],
        solver_timeout: int,
        rounding: int = -1,
        verbose: int = 0,
    ):
        super().__init__()
        self.xs = solver_vars
        self.n = len(solver_vars)
        self.constraints_method = constraints_method
        self._solver_timeout = solver_timeout
        self._rounding = rounding

        # internal vars
        self.iter = -1
        self._last_cex = []

        self._assert_state()

        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(LOGGING_LEVELS[verbose])
        self._logger.debug("Verifier initialized")

    def _assert_state(self) -> None:
        assert (
                self._solver_timeout > 0
        ), f"Solver's timeout must be greater than 0, got {self._solver_timeout}"
        assert isinstance(
            self._solver_timeout, int
        ), "solver timeout must be an integer (in seconds)"
        assert isinstance(self._rounding, int) and self._rounding >= -1, "rounding must be an integer >= -1"

    @staticmethod
    @abstractmethod
    def new_vars(
        n: int | None, var_names: list[str] | None, base: str = "x"
    ) -> list[SYMBOL]:
        """
        Returns a list of symbolic variables.
        It either creates `n` variables with prefix `base`, or creates one variable for each `var_names`.
        """
        raise NotImplementedError("")

    @staticmethod
    @abstractmethod
    def solver_fncts(self) -> dict[str, Callable]:
        raise NotImplementedError("")

    @staticmethod
    @abstractmethod
    def new_solver():
        raise NotImplementedError("")

    @staticmethod
    @abstractmethod
    def is_sat(res) -> bool:
        raise NotImplementedError("")

    @staticmethod
    @abstractmethod
    def is_unsat(res) -> bool:
        raise NotImplementedError("")

    @abstractmethod
    def _solver_solve(self, solver, fml) -> tuple[Any, bool]:
        """
        Returns the result and a boolean indicating if the verification timed out.

        Args:
            solver: solver
            fml: formula to verify
        """
        raise NotImplementedError("")

    @staticmethod
    @abstractmethod
    def _solver_model(self, solver, res):
        raise NotImplementedError("")

    @staticmethod
    @abstractmethod
    def _model_result(solver, model, var, idx):
        raise NotImplementedError("")

    @staticmethod
    @abstractmethod
    def replace_point(expr, ver_vars, point):
        raise NotImplementedError("")

    @staticmethod
    @abstractmethod
    def pretty_formula(fml) -> str:
        raise NotImplementedError("")

    @timed
    def verify(
        self,
        V_symbolic: SYMBOL,
        V_symbolic_constr: Iterable[SYMBOL],
        V_symbolic_vars: list[SYMBOL],
        sigma_symbolic: SYMBOL | None,
        sigma_symbolic_constr: Iterable[SYMBOL],
        sigma_symbolic_vars: list[SYMBOL],
        Vdot_symbolic: SYMBOL,
        Vdot_symbolic_constr: Iterable[SYMBOL],
        Vdot_symbolic_vars: list[SYMBOL],
        Vdot_residual_symbolic: SYMBOL | None,
        Vdot_residual_symbolic_constr: Iterable[SYMBOL],
        Vdot_residual_symbolic_vars: list[SYMBOL],
        **kwargs,
    ):
        """
        :param V_symbolic: z3 expr of function V
        :param sigma_symbolic: z3 expr of function sigma
        :param Vdot_symbolic: z3 expr of Lie derivative of V under nominal model
        :param Vdotz_symbolic: z3 expr of Lie derivative of V under uncertain model
        :return:
                found_lyap: True if C is valid
                C: a list of ctx
        """
        found, timed_out = False, False
        # todo: different verifier may require different inputs -> clean call constraints_method
        fmls = self.constraints_method(
            self,
            V_symbolic,
            V_symbolic_constr,
            V_symbolic_vars,
            sigma_symbolic,
            sigma_symbolic_constr,
            sigma_symbolic_vars,
            Vdot_symbolic,
            Vdot_symbolic_constr,
            Vdot_symbolic_vars,
            Vdot_residual_symbolic,
            Vdot_residual_symbolic_constr,
            Vdot_residual_symbolic_vars,
        )
        results = {}
        solvers = {}
        solver_vars = {}
        solver_aux_vars = {}

        for group in fmls:
            for label, condition_vars in group.items():
                if isinstance(condition_vars, tuple):
                    # CBF returns different variables depending on constraint
                    condition, vars, aux_vars = condition_vars
                else:
                    # Other barriers always use only state variables
                    condition = condition_vars
                    vars = self.xs
                    assert False, "This should not happen"

                s = self.new_solver()
                #self._logger.debug(
                #    f"Constraint: {label}, Formula: {self.pretty_formula(fml=condition)}"
                #)
                res, timedout = self._solver_solve(solver=s, fml=condition)
                results[label] = res
                solvers[label] = s
                solver_vars[label] = vars
                solver_aux_vars[label] = aux_vars
                # if sat, found counterexample; if unsat, C is lyap
                if timedout:
                    self._logger.info(label + "timed out")
            if any(self.is_sat(res) for res in results.values()):
                break

        ces = {label: [] for label in results.keys()}

        if all(self.is_unsat(res) for res in results.values()):
            self._logger.info("No counterexamples found!")
            found = True
        else:
            for index, o in enumerate(results.items()):
                label, res = o
                if self.is_sat(res):
                    original_point = self.compute_model(
                        vars=solver_vars[label], solver=solvers[label], res=res
                    )
                    aux_point = self.compute_model(
                        vars=solver_aux_vars[label], solver=solvers[label], res=res
                    )
                    self._logger.info(
                        f"{label}: Counterexample Found: {solver_vars[label]} = {original_point[0]}, "
                        f"{solver_aux_vars[label]} = {aux_point[0]}"
                    )

                    # debug
                    for sym_name, sym in zip(
                        ["V", "Sigma", "Vdot", "Vdot_residual"],
                        [V_symbolic, sigma_symbolic, Vdot_symbolic, Vdot_residual_symbolic],
                    ):
                        if sym is None:
                            continue
                        replaced = self.replace_point(
                            sym, solver_vars[label], original_point.numpy().T
                        )
                        if hasattr(replaced, "as_fraction"):
                            fraction = replaced.as_fraction()
                            value = float(fraction.numerator / fraction.denominator)
                        else:
                            self._logger.debug(f"Cannot extract value from {replaced} of type {type(replaced)}")
                            value = str(replaced)
                        self._logger.info(f"[cex] {sym_name}: {value}")

                    ces[label] = original_point
                else:
                    self._logger.info(f"{label}: {res}")

        return {"found": found, "cex": ces}

    def compute_model(self, vars, solver, res):
        """
        :param vars: list of solver vars appearing in res
        :param solver: solver
        :return: tensor containing single ctx
        """
        model = self._solver_model(solver, res)
        temp = []
        for i, x in enumerate(vars):
            n = self._model_result(solver, model, x, i)
            temp += [n]

        original_point = torch.tensor(temp)
        return original_point[None, :]


