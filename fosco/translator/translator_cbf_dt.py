from typing import Iterable

from fosco.common.timing import timed
from fosco.models import TorchMLP
from fosco.translator import Translator
from fosco.verifier.types import SYMBOL
from fosco.verifier.utils import get_solver_simplify


class MLPTranslatorDT(Translator):
    """
    Translate a discrete-time cbf network into a symbolic expression for forward pass and Delta-barrier.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._logger.debug("MLPTranslatorDT initialized")

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
        Translate a network forward pass and change over time into a symbolic expression.

        Args:
            x_v_map: dict of symbolic variables
            V_net: network model
            xdot: symbolic expression of the discrete-time system dynamics
            **kwargs:

        Returns:
            dict of symbolic expressions
        """
        assert "v" in x_v_map, "x_v_map must contain key 'v' for symbolic variables"

        x_vars = x_v_map["v"]
        symplify_fn = get_solver_simplify(x_vars)

        V_symbolic, V_symbolic_constr, V_symbolic_vars = V_net.forward_smt(x=x_vars)
        V_symbolic = symplify_fn(V_symbolic)
        Vnext_symbolic, Vnext_symbolic_constr, Vnext_symbolic_vars = V_net.forward_smt(
            x=xdot
        )
        Vnext_symbolic = symplify_fn(Vnext_symbolic)
        Vdot_symbolic = Vnext_symbolic - V_symbolic
        Vdot_symbolic = symplify_fn(Vdot_symbolic)
        Vdot_symbolic_constr = V_symbolic_constr
        Vdot_symbolic_vars = V_symbolic_vars

        assert len(V_symbolic_constr) == 0, "Expected no constraints for V_symbolic"
        assert (
            len(Vnext_symbolic_constr) == 0
        ), "Expected no constraints for Vnext_symbolic"

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
