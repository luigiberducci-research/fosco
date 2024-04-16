from typing import Iterable

import numpy as np

from fosco.common.timing import timed
from fosco.models import TorchMLP
from fosco.translator import Translator
from fosco.verifier.types import SYMBOL


class MLPTranslator(Translator):
    """
    Translate a continuous-time cbf network into a symbolic expression for forward pass and Lie derivative.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._logger.debug("MLPTranslator initialized")

    def _assert_state(self) -> None:
        pass

    @timed
    def translate(
        self,
        x_v_map: dict[str, Iterable[SYMBOL]],
        V_net: TorchMLP,
        xdot: Iterable[SYMBOL],
        **kwargs,
    ) -> dict:
        """
        Translate a network forward pass and gradients into a symbolic expression
        of the function and Lie derivative w.r.t. the system dynamics.

        Args:
            x_v_map: dict of symbolic variables
            V_net: network model
            xdot: symbolic expression of the nominal system dynamics
            **kwargs:

        Returns:
            dict of symbolic expressions
        """
        assert "v" in x_v_map, "x_v_map must contain key 'v' for symbolic variables"

        x_vars = x_v_map["v"]
        xdot = np.array(xdot).reshape(-1, 1)

        V_symbolic, V_symbolic_constr, V_symbolic_vars = V_net.forward_smt(x=x_vars)
        Vgrad_symbolic, Vdot_symbolic_constr, Vgrad_symbolic_vars = V_net.gradient_smt(
            x=x_vars
        )
        Vdot_symbolic = (Vgrad_symbolic @ xdot)[0, 0]
        Vdot_symbolic_vars = Vgrad_symbolic_vars

        assert isinstance(
            V_symbolic, SYMBOL
        ), f"Expected V_symbolic to be {SYMBOL}, got {type(V_symbolic)}"
        assert isinstance(
            Vdot_symbolic, SYMBOL
        ), f"Expected Vdot_symbolic to be {SYMBOL}, got {type(Vdot_symbolic)}"

        return {
            "V_symbolic": V_symbolic,
            "V_symbolic_constr": V_symbolic_constr,
            "V_symbolic_vars": V_symbolic_vars,
            "Vdot_symbolic": Vdot_symbolic,
            "Vdot_symbolic_constr": Vdot_symbolic_constr,
            "Vdot_symbolic_vars": Vdot_symbolic_vars,
        }
