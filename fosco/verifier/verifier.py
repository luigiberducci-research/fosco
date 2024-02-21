import logging
from abc import abstractmethod, ABC
from typing import Callable, Generator, Iterable, Type, Any

import torch
import z3

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
        n_counterexamples: int,
        rounding: int = -1,
        verbose: int = 0,
    ):
        super().__init__()
        self.xs = solver_vars
        self.n = len(solver_vars)
        self.constraints_method = constraints_method

        # internal vars
        self.iter = -1
        self._last_cex = []

        # todo: move this to consolidator
        self.counterexample_n = n_counterexamples
        self._n_cex_to_keep = self.counterexample_n * 1
        self._solver_timeout = solver_timeout
        self._rounding = rounding

        assert self.counterexample_n > 0
        assert self._n_cex_to_keep > 0
        assert self._solver_timeout > 0

        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(LOGGING_LEVELS[verbose])
        self._logger.debug("Translator initialized")

    @staticmethod
    @abstractmethod
    def new_vars(n: int | None, var_names: list[str] | None, base: str = "x") -> list[SYMBOL]:
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

    @timed
    def verify(
        self,
        V_symbolic: SYMBOL,
        V_symbolic_constr: Iterable[SYMBOL],
        sigma_symbolic: SYMBOL | None,
        sigma_symbolic_constr: Iterable[SYMBOL],
        Vdot_symbolic: SYMBOL,
        Vdot_symbolic_constr: Iterable[SYMBOL],
        Vdotz_symbolic: SYMBOL | None,
        Vdotz_symbolic_constr: Iterable[SYMBOL],
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
            sigma_symbolic,
            sigma_symbolic_constr,
            Vdot_symbolic,
            Vdot_symbolic_constr,
            Vdotz_symbolic,
            Vdotz_symbolic_constr,
        )
        results = {}
        solvers = {}
        solver_vars = {}

        for group in fmls:
            for label, condition_vars in group.items():
                if isinstance(condition_vars, tuple):
                    # CBF returns different variables depending on constraint
                    condition, vars = condition_vars
                else:
                    # Other barriers always use only state variables
                    condition = condition_vars
                    vars = self.xs

                s = self.new_solver()
                res, timedout = self._solver_solve(solver=s, fml=condition)
                results[label] = res
                solvers[label] = s
                solver_vars[label] = vars  # todo: select diff vars for input and state
                # if sat, found counterexample; if unsat, C is lyap
                if timedout:
                    self._logger.info(label + "timed out")
            if any(self.is_sat(res) for res in results.values()):
                break

        ces = {label: [] for label in results.keys()}  # [[] for res in results.keys()]

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
                    self._logger.info(
                        f"{label}: Counterexample Found: {solver_vars[label]} = {original_point}"
                    )

                    # debug
                    for sym_name, sym in zip(
                        ["V", "Sigma", "Vdot", "Vdotz"],
                        [V_symbolic, sigma_symbolic, Vdot_symbolic, Vdotz_symbolic],
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
                            value = None
                        self._logger.debug(f"[cex] {sym_name}: {value}")

                    ces[label] = self.randomise_counterex(original_point)
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

    # given one ctx, useful to sample around it to increase data set
    # these points might *not* be real ctx, but probably close to invalidity condition
    # todo: this is not work of the consolidator?
    def randomise_counterex(self, point):
        """
        :param point: tensor
        :return: list of ctx
        """
        C = []
        # dimensionality issue
        shape = (1, max(point.shape[0], point.shape[1]))
        point = point.reshape(shape)
        for i in range(self.counterexample_n):
            random_point = point + 5 * 1e-3 * torch.randn(
                shape
            )  # todo: parameterize this stddev
            # if self.inner < torch.norm(random_point) < self.outer:
            C.append(random_point)
        C.append(point)
        return torch.stack(C, dim=1)[0, :, :]
