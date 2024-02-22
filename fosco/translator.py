import logging
from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np

from fosco.common.consts import VerifierType, TimeDomain, CertificateType
from fosco.common.timing import timed
from models.network import TorchMLP
from fosco.verifier.verifier import SYMBOL
from fosco.logger import LOGGING_LEVELS


class Translator(ABC):
    """
    Abstract class for symbolic translators.
    """

    @abstractmethod
    def translate(self, **kwargs) -> dict:
        """
        Translate a network forward pass and gradients into a symbolic expression.

        Returns:
            - dictionary of symbolic expressions
        """
        raise NotImplementedError


class MLPTranslator(Translator):
    """
    Translate a network forward pass and gradients into a symbolic expression.
    """

    def __init__(self, verbose: int = 0):
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(LOGGING_LEVELS[verbose])
        self._logger.debug("Translator initialized")

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
            xdotz: symbolic expression of the uncertain system dynamics (optional)
            **kwargs:

        Returns:
            dict of symbolic expressions
        """
        assert "v" in x_v_map, "x_v_map must contain key 'v' for symbolic variables"

        x_vars = x_v_map["v"]
        xdot = np.array(xdot).reshape(-1, 1)

        V_symbolic, V_symbolic_constr = V_net.forward_smt(x=x_vars)
        Vgrad_symbolic, Vdot_symbolic_constr = V_net.gradient_smt(x=x_vars)
        Vdot_symbolic = (Vgrad_symbolic @ xdot)[0, 0]

        assert isinstance(
            V_symbolic, SYMBOL
        ), f"Expected V_symbolic to be {SYMBOL}, got {type(V_symbolic)}"
        assert isinstance(
            Vdot_symbolic, SYMBOL
        ), f"Expected Vdot_symbolic to be {SYMBOL}, got {type(Vdot_symbolic)}"

        return {
            "V_symbolic": V_symbolic,
            "Vdot_symbolic": Vdot_symbolic,
            "V_symbolic_constr": V_symbolic_constr,
            "Vdot_symbolic_constr": Vdot_symbolic_constr,
        }


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
        xdotz: Iterable[SYMBOL] = None,
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
            **kwargs:

        Returns:
            dict of symbolic expressions
        """
        assert sigma_net is not None, "sigma_net must be not None"
        assert xdotz is not None, "xdotz must be not None"

        # note: ignore elapsed_time here, it is because of the decorator timed
        symbolic_dict, elapsed_time = super().translate(x_v_map, V_net, xdot)

        x_vars = x_v_map["v"]
        xdotz = np.array(xdotz).reshape(-1, 1)

        # robust cbf: compensation term
        sigma_symbolic, sigma_symbolic_constr = sigma_net.forward_smt(x=x_vars)

        # lie derivative under uncertain dynamics
        Vgrad_symbolic, Vdotz_symbolic_constr = V_net.gradient_smt(x=x_vars)
        Vdotz_symbolic = (Vgrad_symbolic @ xdotz)[0, 0]

        assert isinstance(
            sigma_symbolic, SYMBOL
        ), f"Expected V_symbolic to be {SYMBOL}, got {type(sigma_symbolic)}"
        assert isinstance(
            Vdotz_symbolic, SYMBOL
        ), f"Expected Vdot_symbolic to be {SYMBOL}, got {type(Vdotz_symbolic)}"

        symbolic_dict.update(
            {
                "Vdotz_symbolic": Vdotz_symbolic,
                "sigma_symbolic": sigma_symbolic,
                "sigma_symbolic_constr": sigma_symbolic_constr,
                "Vdotz_symbolic_constr": Vdotz_symbolic_constr,
            }
        )

        return symbolic_dict


def make_translator(
    certificate_type: CertificateType,
    verifier_type: VerifierType,
    time_domain: TimeDomain,
    **kwargs,
) -> Translator:
    """
    Factory function for translators.
    """
    if time_domain == TimeDomain.CONTINUOUS and verifier_type in [
        VerifierType.Z3,
        VerifierType.DREAL,
    ]:
        if certificate_type == CertificateType.RCBF:
            return RobustMLPTranslator(**kwargs)
        elif certificate_type == CertificateType.CBF:
            return MLPTranslator(**kwargs)
        else:
            raise NotImplementedError(
                f"Translator for certificate={certificate_type} and time={time_domain} not implemented"
            )
    else:
        raise NotImplementedError(
            f"Translator for verifier={verifier_type} and time={time_domain} not implemented"
        )
