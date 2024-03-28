from typing import Iterable

import numpy as np

from fosco.common.timing import timed
from fosco.models import TorchMLP
from fosco.translator import MLPTranslatorDT
from fosco.verifier.types import SYMBOL
from fosco.verifier.utils import get_solver_simplify


class RobustMLPTranslatorDT(MLPTranslatorDT):
    """
    Translator for robust model to symbolic expressions.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._logger.debug("RobustMLPTranslator initialized")
    @timed
    def translate(
        self,
        x_v_map: dict[str, Iterable[SYMBOL]],
        V_net: TorchMLP,
        sigma_net: TorchMLP,
        xdot: Iterable[SYMBOL],
        xdot_residual: Iterable[SYMBOL] = None,
        **kwargs,
    ):
        """
        Translate a network forward pass and gradients into a symbolic expression
        of the function and Lie derivative w.r.t. the system dynamics.

        Args:
            x_v_map: dict of symbolic variables
            V_net: network model
            xdot: symbolic expression of the nominal system dynamics
            xdot_residual: symbolic expression of the residual dynamics (optional)
            **kwargs:

        Returns:
            dict of symbolic expressions
        """
        assert sigma_net is not None, "sigma_net must be not None"
        assert xdot_residual is not None, "xdot_residual not supported"
        assert isinstance(xdot, np.ndarray), "Expected xdot to be np.ndarray"
        assert isinstance(xdot_residual, np.ndarray), "Expected xdot_residual to be np.ndarray"
        assert xdot_residual.shape == xdot.shape, f"Expected same shape for xdot and xdot_residual, got {xdot_residual.shape} and {xdot.shape}"

        x_vars = x_v_map["v"]
        symplify_fn = get_solver_simplify(x_vars)

        # note: ignore elapsed_time here, it is because of the decorator timed
        symbolic_dict, elapsed_time = super().translate(x_v_map, V_net, xdot)

        # symbolic expressions for the nominal system dynamics
        V_symbolic = symbolic_dict["V_symbolic"]
        Vdot_symbolic = symbolic_dict["Vdot_symbolic"]

        # robust cbf: compensation term
        sigma_symbolic, sigma_symbolic_constr, sigma_symbolic_vars = sigma_net.forward_smt(x=x_vars)
        sigma_symbolic = symplify_fn(sigma_symbolic)

        # time-diff of barrier at the next step under uncertain dynamics
        next_xz = xdot + xdot_residual
        Vnextz_symbolic, Vnextz_symbolic_constr, Vnextz_symbolic_vars = V_net.forward_smt(x=next_xz)
        Vdotz_symbolic = Vnextz_symbolic - V_symbolic

        # residual in the time-difference of V
        Vdot_residual_symbolic = Vdotz_symbolic - Vdot_symbolic
        Vdot_residual_symbolic = symplify_fn(Vdot_residual_symbolic)
        Vdot_residual_symbolic_constr = []
        Vdot_residual_symbolic_vars = []

        assert isinstance(
            sigma_symbolic, SYMBOL
        ), f"Expected V_symbolic to be {SYMBOL}, got {type(sigma_symbolic)}"
        assert isinstance(
            Vdot_residual_symbolic, SYMBOL
        ), f"Expected Vdot_residual_symbolic to be {SYMBOL}, got {type(Vdot_residual_symbolic)}"

        symbolic_dict.update(
            {
                "sigma_symbolic": sigma_symbolic,
                "sigma_symbolic_constr": sigma_symbolic_constr,
                "sigma_symbolic_vars": sigma_symbolic_vars,
                "Vdot_residual_symbolic": Vdot_residual_symbolic,
                "Vdot_residual_symbolic_constr": Vdot_residual_symbolic_constr,
                "Vdot_residual_symbolic_vars": Vdot_residual_symbolic_vars,
            }
        )

        return symbolic_dict
