from typing import Iterable

import numpy as np

from fosco.common.timing import timed
from fosco.models import TorchMLP
from fosco.translator.translator_cbf import MLPTranslator
from fosco.verifier.types import SYMBOL


class RobustMLPTranslator(MLPTranslator):
    """
    Translator for robust model to symbolic expressions.
    """

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
            xdotz: symbolic expression of the uncertain system dynamics (optional)
            xdot_residual: symbolic expression of the residual dynamics (optional)
            **kwargs:

        Returns:
            dict of symbolic expressions
        """
        assert sigma_net is not None, "sigma_net must be not None"
        # assert xdotz is not None, "xdotz must be not None"
        assert xdot_residual is not None, "xdot_residual not supported"

        # note: ignore elapsed_time here, it is because of the decorator timed
        symbolic_dict, elapsed_time = super().translate(x_v_map, V_net, xdot)

        x_vars = x_v_map["v"]
        xdot_residual = np.array(xdot_residual).reshape(-1, 1)

        # robust cbf: compensation term
        sigma_symbolic, sigma_symbolic_constr, sigma_symbolic_vars = sigma_net.forward_smt(x=x_vars)

        # lie derivative under uncertain dynamics
        Vgrad_symbolic, Vgrad_symbolic_constr, Vgrad_symbolic_vars = V_net.gradient_smt(x=x_vars)
        Vdot_residual_symbolic = (Vgrad_symbolic @ xdot_residual)[0, 0]
        Vdot_residual_symbolic_constr = Vgrad_symbolic_constr
        Vdot_residual_symbolic_vars = Vgrad_symbolic_vars

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
