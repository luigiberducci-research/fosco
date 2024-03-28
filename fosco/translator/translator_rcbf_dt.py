from typing import Iterable

import numpy as np

from fosco.common.timing import timed
from fosco.models import TorchMLP
from fosco.translator import MLPTranslatorDT
from fosco.verifier.types import SYMBOL


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

        # note: ignore elapsed_time here, it is because of the decorator timed
        symbolic_dict, elapsed_time = super().translate(x_v_map, V_net, xdot)

        x_vars = x_v_map["v"]

        # robust cbf: compensation term
        sigma_symbolic, sigma_symbolic_constr, sigma_symbolic_vars = sigma_net.forward_smt(x=x_vars)

        # lie derivative under uncertain dynamics
        next_x = xdot + xdot_residual
        Vnext_symbolic, Vnext_symbolic_constr, Vnext_symbolic_vars = V_net.forward_smt(x=next_x)
        V_symbolic, V_symbolic_constr, V_symbolic_vars = symbolic_dict["V_symbolic"], symbolic_dict["V_symbolic_constr"], symbolic_dict["V_symbolic_vars"]

        Vdot_residual_symbolic = Vnext_symbolic - V_symbolic
        Vdot_residual_symbolic_constr = V_symbolic_constr
        Vdot_residual_symbolic_vars = V_symbolic_vars

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
