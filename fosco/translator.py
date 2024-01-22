import logging
from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np
import z3

from fosco.common.activations_symbolic import activation_sym, activation_der_sym
from fosco.common.consts import VerifierType, TimeDomain, CertificateType
from fosco.models.network import TorchMLP
from fosco.verifier import SYMBOL
from logger import LOGGING_LEVELS
from systems import ControlAffineControllableDynamicalModel
from systems.system import UncertainControlAffineControllableDynamicalModel


class Translator(ABC):
    """
    Abstract class for symbolic translators.
    """

    @abstractmethod
    def translate(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_symbolic_net(self, input_vars: Iterable[SYMBOL], net: TorchMLP) -> SYMBOL:
        """
        Translate a network forward pass into a symbolic expression.

        :param input_vars: list of symbolic variables
        :param net: network model
        :return: symbolic expression
        """
        raise NotImplementedError

    @abstractmethod
    def get_symbolic_net_grad(
        self, input_vars: Iterable[SYMBOL], net: TorchMLP
    ) -> Iterable[SYMBOL]:
        """
        Translate the network gradient w.r.t. the input into a symbolic expression.

        :param input_vars: list of symbolic variables
        :param net: network model
        :return:
        """
        raise NotImplementedError


class MLPZ3Translator(Translator):
    """
    Symbolic translator for feed-forward neural networks to z3 expressions.
    """

    def __init__(self, rounding: int = 3, verbose: int = 0):
        self.round = rounding

        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(LOGGING_LEVELS[verbose])
        self._logger.debug("Translator initialized")

    def translate(
        self,
        x_v_map: dict[str, Iterable[SYMBOL]],
        V_net: TorchMLP,
        xdot: Iterable[SYMBOL],
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
        x_vars = x_v_map["v"]
        xdot = np.array(xdot).reshape(-1, 1)

        V_symbolic = self.get_symbolic_net(x_vars, V_net)
        Vdot_symbolic = (self.get_symbolic_net_grad(x_vars, V_net) @ xdot)[0, 0]

        assert isinstance(
            V_symbolic, z3.ArithRef
        ), f"Expected V_symbolic to be z3.ArithRef, got {type(V_symbolic)}"
        assert isinstance(
            Vdot_symbolic, z3.ArithRef
        ), f"Expected Vdot_symbolic to be z3.ArithRef, got {type(Vdot_symbolic)}"

        return {
            "V_symbolic": V_symbolic,
            "Vdot_symbolic": Vdot_symbolic,
        }

    def get_symbolic_formula(self, x, net, xdot):
        """
        Return symbolic expression of V and Vdot.

        :param net: network model
        :param x: symbolic variables
        :param xdot: symbolic expression of the system dynamics
        :return: tuple (V, Vdot)
        """
        V = self.get_symbolic_net(x, net)
        Vdot = (self.get_symbolic_net_grad(x, net) @ xdot)[0, 0]
        return V, Vdot

    def get_symbolic_net(self, input_vars: Iterable[SYMBOL], net: TorchMLP) -> SYMBOL:
        """
        Translate a MLP forward pass into a symbolic expression.

        :param input_vars: list of symbolic variables
        :param net: network model
        :return: symbolic expression
        """
        input_vars = np.array(input_vars).reshape(-1, 1)

        # todo: remove separate management of last layer
        z, _ = self.network_until_last_layer(net, input_vars)

        if self.round < 0:
            last_layer = net.layers[-1].weight.data.numpy()
        else:
            last_layer = np.round(net.layers[-1].weight.data.numpy(), self.round)

        z = last_layer @ z
        if net.layers[-1].bias is not None:
            z += net.layers[-1].bias.data.numpy()[:, None]
        assert z.shape == (1, 1), f"Wrong shape of z, expected (1, 1), got {z.shape}"

        # last activation
        z = activation_sym(net.acts[-1], z)

        V = z[0, 0]
        V = z3.simplify(V)

        return V

    def get_symbolic_net_grad(
        self, input_vars: Iterable[SYMBOL], net: TorchMLP
    ) -> Iterable[SYMBOL]:
        """
        Translate the MLP gradient w.r.t. the input into a symbolic expression.

        :param input_vars: list of symbolic variables
        :param net: network model
        :return:
        """
        input_vars = np.array(input_vars).reshape(-1, 1)

        # todo: remove separate management of last layer
        z, jacobian = self.network_until_last_layer(net, input_vars)

        if self.round < 0:
            last_layer = net.layers[-1].weight.data.numpy()
        else:
            last_layer = np.round(net.layers[-1].weight.data.numpy(), self.round)

        zhat = last_layer @ z
        if net.layers[-1].bias is not None:
            zhat += net.layers[-1].bias.data.numpy()[:, None]

        # last activation
        z = activation_sym(net.acts[-1], zhat)

        jacobian = last_layer @ jacobian
        jacobian = np.diagflat(activation_der_sym(net.acts[-1], zhat)) @ jacobian

        gradV = jacobian

        assert z.shape == (1, 1)
        assert gradV.shape == (
            1,
            net.input_size,
        ), f"Wrong shape of gradV, expected (1, {net.input_size}), got {gradV.shape}"

        # z3 simplification
        for i in range(net.input_size):
            gradV[0, i] = (
                z3.simplify(gradV[0, i])
                if isinstance(gradV[0, i], z3.ArithRef)
                else gradV[0, i]
            )

        return gradV

    def network_until_last_layer(
        self, net: TorchMLP, input_vars: Iterable[SYMBOL]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Symbolic forward pass excluding the last layer.

        :param net: network model
        :param input_vars: list of symbolic variables
        :return: tuple (net output, its jacobian)
        """
        z = input_vars
        jacobian = np.eye(net.input_size, net.input_size)

        for idx, layer in enumerate(net.layers[:-1]):
            if self.round < 0:
                w = layer.weight.data.numpy()
                if layer.bias is not None:
                    b = layer.bias.data.numpy()[:, None]
                else:
                    b = np.zeros((layer.out_features, 1))
            elif self.round >= 0:
                w = np.round(layer.weight.data.numpy(), self.round)
                if layer.bias is not None:
                    b = np.round(layer.bias.data.numpy(), self.round)[:, None]
                else:
                    b = np.zeros((layer.out_features, 1))

            zhat = w @ z + b
            z = activation_sym(net.acts[idx], zhat)

            jacobian = w @ jacobian
            jacobian = np.diagflat(activation_der_sym(net.acts[idx], zhat)) @ jacobian

        return z, jacobian


class RobustMLPZ3Translator(MLPZ3Translator):
    """
    Symbolic translator for robust model to z3 expressions.
    """

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

        symbolic_dict = super().translate(x_v_map, V_net, xdot)

        x_vars = x_v_map["v"]
        xdotz = np.array(xdotz).reshape(-1, 1)

        # robust cbf: compensation term
        sigma_symbolic = self.get_symbolic_net(x_vars, sigma_net)

        # lie derivative under uncertain dynamics
        Vdotz_symbolic = (self.get_symbolic_net_grad(x_vars, V_net) @ xdotz)[0, 0]

        assert isinstance(
            sigma_symbolic, z3.ArithRef
        ), f"Expected V_symbolic to be z3.ArithRef, got {type(V_symbolic)}"
        assert isinstance(
            Vdotz_symbolic, z3.ArithRef
        ), f"Expected Vdot_symbolic to be z3.ArithRef, got {type(Vdot_symbolic)}"

        symbolic_dict.update(
            {"Vdotz_symbolic": Vdotz_symbolic, "sigma_symbolic": sigma_symbolic,}
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
    if verifier_type == VerifierType.Z3 and time_domain == TimeDomain.CONTINUOUS:
        if certificate_type == CertificateType.RCBF:
            return RobustMLPZ3Translator(**kwargs)
        elif certificate_type == CertificateType.CBF:
            return MLPZ3Translator(**kwargs)
        else:
            raise NotImplementedError(
                f"Translator for certificate={certificate_type} and time={time_domain} not implemented"
            )
    else:
        raise NotImplementedError(
            f"Translator for verifier={verifier_type} and time={time_domain} not implemented"
        )
