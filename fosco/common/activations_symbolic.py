# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging

import numpy as np


from fosco.verifier.types import Z3SYMBOL, DRSYMBOL, SPSYMBOL

import z3
import sympy

from fosco.verifier.utils import get_solver_fns

try:
    import dreal as dr
except Exception as e:
    logging.exception("Exception while importing dReal")

from fosco.common import consts


def activation_sym(select, p):
    if select == consts.ActivationType.IDENTITY:
        return p
    elif select == consts.ActivationType.RELU:
        return relu(p)
    elif select == consts.ActivationType.LINEAR:
        return p
    elif select == consts.ActivationType.SQUARE:
        return square_sym(p)

    elif select == consts.ActivationType.REQU:
        return requ_sym(p)
    elif select == consts.ActivationType.TANH:
        return hyper_tan(p)
    elif select == consts.ActivationType.HTANH:
        return hard_hyper_tan(p)
    elif select == consts.ActivationType.SIGMOID:
        return sigm(p)
    elif select == consts.ActivationType.HSIGMOID:
        return hard_sigm(p)
    elif select == consts.ActivationType.SOFTPLUS:
        return softplus(p)
    elif select == consts.ActivationType.COSH:
        return cosh(p)

    elif select == consts.ActivationType.RATIONAL:
        return rational_sym(p)
    else:
        raise NotImplementedError(f"Activation {select} not implemented")


def activation_der_sym(select, p):
    fns = get_solver_fns(p[0])
    if select == consts.ActivationType.IDENTITY:
        return np.ones(p.shape)
    elif select == consts.ActivationType.RELU:
        return step(p)
    elif select == consts.ActivationType.LINEAR:
        return np.ones(p.shape)
    elif select == consts.ActivationType.SQUARE:
        return 2 * p

    elif select == consts.ActivationType.REQU:
        return requ_der_z3(p)
    elif select == consts.ActivationType.TANH:
        return hyper_tan_der(p)
    elif select == consts.ActivationType.HTANH:
        return hard_hyper_tan_der(p)
    elif select == consts.ActivationType.SIGMOID:
        return sigm_der(p)
    elif select == consts.ActivationType.HSIGMOID:
        return hard_sigm_der(p)
    elif select == consts.ActivationType.COSH:
        return sinh(p)

    elif select == consts.ActivationType.RATIONAL:
        return rational_der_sym(p)
    else:
        raise NotImplementedError(f"Activation {select} not implemented")


def relu(x):
    """
    Rectified linear unit f(x) = max(0, x)
    """
    y = x.copy()
    if isinstance(x[0, 0], Z3SYMBOL):
        _If = z3.If
        for idx in range(len(y)):
            y[idx, 0] = z3.simplify(_If(y[idx, 0] > 0, y[idx, 0], 0))

    elif isinstance(x[0, 0], DRSYMBOL):
        _max = dr.Max
        for idx in range(len(y)):
            y[idx, 0] = _max(y[idx, 0], 0)

    elif isinstance(x[0, 0], SPSYMBOL):
        for idx in range(len(y)):
            y[idx, 0] = sympy.Max(y[idx, 0], 0)
    return y


def square_sym(x):
    """
    Square activation f(x) = x^2
    """
    return np.power(x, 2)


def lin_square_sym(x):
    """
    Linear - quadratic activation f(x) = [x, x^2]
    """
    h = int(len(x) / 2)
    x1, x2 = x[:h], x[h:]
    return np.vstack((x1, np.power(x2, 2)))


def relu_square_sym(x):
    """
    ReLU - square activation f(x) = [max(0, x), x^2]
    """
    h = int(len(x) / 2)
    x1, x2 = x[:h], x[h:]
    return np.vstack((relu(x1), np.power(x2, 2)))


def requ_sym(x):
    """
    Requ is f(x) = x^2/(1 + x^2)
    """
    return np.multiply(x, relu(x))


def hyper_tan(x):
    """
    Hyperbolic tangent f(x) = (e^x - e^-x)/(e^x + e^-x)
    """
    y = x.copy()
    if isinstance(x[0, 0], Z3SYMBOL):
        raise NotImplementedError("Z3 not implemented for hyperbolic tangent")
    elif isinstance(x[0, 0], DRSYMBOL):
        for idx in range(len(y)):
            y[idx, 0] = dr.tanh(y[idx, 0])
    return y

def hard_hyper_tan(x):
    """
    Hard hyperbolic tangent f(x) = max(-1, min(1, x))
    """
    fns = get_solver_fns(x[0])
    y = x.copy()
    _If = fns["If"]

    for idx in range(len(y)):
        y[idx, 0] = _If(
            y[idx, 0] > 1.0,
            1.0,
            _If(y[idx, 0] < -1.0, -1.0, y[idx, 0]),
        )
    return y


def sigm(x):
    """
    Sigmoid f(x) = 1/(1 + e^-x)
    """
    y = x.copy()
    if isinstance(x[0, 0], Z3SYMBOL):
        raise NotImplementedError("Z3 not implemented for sigmoid")
    elif isinstance(x[0, 0], DRSYMBOL):
        for idx in range(len(y)):
            y[idx, 0] = 1 / (1 + dr.exp(-y[idx, 0]))
    return y


def hard_sigm(x):
    """
    Hard Sigmoid f(x) = 1 if x > 3, 0 if x < -3, else x / 6 + 0.5
    """
    fns = get_solver_fns(x[0])
    y = x.copy()
    _If = fns["If"]

    for idx in range(len(y)):
        y[idx, 0] = _If(
            y[idx, 0] > 3.0,
            1.0,
            _If(y[idx, 0] < -3.0, 0.0, y[idx, 0] / 6 + 0.5),
        )
    return y


def softplus(x):
    """
    Softplus is f(x) = ln(1 + e^x)
    """
    y = x.copy()
    if isinstance(x[0, 0], Z3SYMBOL):
        raise NotImplementedError("Z3 not implemented for softplus")
    elif isinstance(x[0, 0], DRSYMBOL):
        for idx in range(len(y)):
            y[idx, 0] = dr.log(1 + dr.exp(y[idx, 0]))
    return y


def cosh(x):
    """
    Hyperbolic cosine f(x) = (e^x + e^-x)/2
    """
    y = x.copy()
    if isinstance(x[0, 0], Z3SYMBOL):
        raise NotImplementedError("Z3 not implemented for hyperbolic cosine")
    elif isinstance(x[0, 0], DRSYMBOL):
        for idx in range(len(y)):
            y[idx, 0] = dr.cosh(y[idx, 0]) - 1
    return y




def rational_sym(x):
    """
    Rational activation f(x) = 1/(1 + x^2)
    """
    return x / (1 + (x**2) ** 0.5)


##############################
# DERIVATIVE
##############################


def step(x):
    y = x.copy()
    original_shape = y.shape
    y = y.reshape(max(y.shape[0], y.shape[1]), 1)
    if isinstance(x[0, 0], z3.ArithRef):
        _If = z3.If
        for idx in range(y.shape[0]):
            y[idx, 0] = z3.simplify(
                _If(y[idx, 0] > 0.0, 1.0, 0.0)
            )  # using 0.0 and 1.0 avoids int/float issues

    else:
        _If = dr.if_then_else
        for idx in range(y.shape[0]):
            y[idx, 0] = _If(
                y[idx, 0] > 0.0, 1.0, 0.0
            )  # using 0.0 and 1.0 avoids int/float issues

    return y.reshape(original_shape)




def requ_der_z3(x):
    return 2 * relu(x)


def hyper_tan_der(x):
    """
    Derivative of hyperbolic tangent f'(x) = 1 - (tanh(x))^2 = 1 / cosh(x)^2

    For z3, we use hard tanh instead. f'(x) = 0 if x > 1 or x < -1, else 1
    """
    y = x.copy()
    if isinstance(x[0, 0], Z3SYMBOL):
        raise NotImplementedError("Z3 not implemented this activation")
    elif isinstance(x[0, 0], DRSYMBOL):
        for idx in range(len(y)):
            y[idx, 0] = 1 / dr.pow(dr.cosh(y[idx, 0]), 2)
    return y

def hard_hyper_tan_der(x):
    y = x.copy()

    fns = get_solver_fns(x[0])
    _If = fns["If"]
    for idx in range(len(y)):
        y[idx, 0] = _If(
            y[idx, 0] > 1.0,
            0.0,
            _If(y[idx, 0] < -1.0, 0.0, 1),
        )

    return y


def sinh(x):
    if isinstance(x[0, 0], Z3SYMBOL):
        raise NotImplementedError("Z3 not implemented this activation")
    y = x.copy()
    for idx in range(len(y)):
        y[idx, 0] = dr.sinh(y[idx, 0])
    return y


def sigm_der(x):
    y = x.copy()
    if isinstance(x[0, 0], Z3SYMBOL):
        raise NotImplementedError("Z3 not implemented this activation")
    elif isinstance(x[0, 0], DRSYMBOL):
        for idx in range(len(y)):
            y[idx, 0] = dr.exp(-y[idx, 0]) / dr.pow((1 + dr.exp(-y[idx, 0])), 2)
    return y


def hard_sigm_der(x):
    fns = get_solver_fns(x[0])
    _If = fns["If"]
    y = x.copy()

    for idx in range(len(y)):
        y[idx, 0] = _If(
            y[idx, 0] > 3.0,
            0.0,
            _If(y[idx, 0] < -3.0, 0.0, 1 / 6),
        )

    return y




def rational_der_sym(x):
    return 1 / (1 + x**2)
