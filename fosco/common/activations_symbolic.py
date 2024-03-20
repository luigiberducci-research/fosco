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
    elif select == consts.ActivationType.POLY_2:
        return lin_square_sym(p)
    elif select == consts.ActivationType.RELU_SQUARE:
        return relu_square_sym(p)
    elif select == consts.ActivationType.REQU:
        return requ_sym(p)
    elif select == consts.ActivationType.TANH:
        return hyper_tan(p)
    elif select == consts.ActivationType.SIGMOID:
        return sigm(p)
    elif select == consts.ActivationType.SOFTPLUS:
        return softplus(p)
    elif select == consts.ActivationType.COSH:
        return cosh(p)
    elif select == consts.ActivationType.POLY_3:
        return poly3_sym(p)
    elif select == consts.ActivationType.POLY_4:
        return poly4_sym(p)
    elif select == consts.ActivationType.POLY_5:
        return poly5_sym(p)
    elif select == consts.ActivationType.POLY_6:
        return poly6_sym(p)
    elif select == consts.ActivationType.POLY_7:
        return poly7_sym(p)
    elif select == consts.ActivationType.POLY_8:
        return poly8_sym(p)
    elif select == consts.ActivationType.EVEN_POLY_4:
        return even_poly4_sym(p)
    elif select == consts.ActivationType.EVEN_POLY_6:
        return even_poly6_sym(p)
    elif select == consts.ActivationType.EVEN_POLY_8:
        return even_poly8_sym(p)
    elif select == consts.ActivationType.EVEN_POLY_10:
        return even_poly10_sym(p)
    elif select == consts.ActivationType.RATIONAL:
        return rational_sym(p)
    else:
        raise NotImplementedError(f"Activation {select} not implemented")


def activation_der_sym(select, p):
    if select == consts.ActivationType.IDENTITY:
        return np.ones((p.shape))
    elif select == consts.ActivationType.RELU:
        return step(p)
    elif select == consts.ActivationType.LINEAR:
        return np.ones(p.shape)
    elif select == consts.ActivationType.SQUARE:
        return 2 * p
    elif select == consts.ActivationType.POLY_2:
        return poly2_der_sym(p)
    elif select == consts.ActivationType.RELU_SQUARE:
        return relu_square_der_z3(p)
    elif select == consts.ActivationType.REQU:
        return requ_der_z3(p)
    elif select == consts.ActivationType.TANH:
        return hyper_tan_der(p)
    elif select == consts.ActivationType.SIGMOID:
        return sigm_der(p)
    elif select == consts.ActivationType.COSH:
        return sinh(p)
    elif select == consts.ActivationType.POLY_3:
        return poly3_der_sym(p)
    elif select == consts.ActivationType.POLY_4:
        return poly4_der_sym(p)
    elif select == consts.ActivationType.POLY_5:
        return poly5_der_sym(p)
    elif select == consts.ActivationType.POLY_6:
        return poly6_der_sym(p)
    elif select == consts.ActivationType.POLY_7:
        return poly7_der_sym(p)
    elif select == consts.ActivationType.POLY_8:
        return poly8_der_sym(p)
    elif select == consts.ActivationType.EVEN_POLY_4:
        return even_poly4_der_sym(p)
    elif select == consts.ActivationType.EVEN_POLY_6:
        return even_poly6_der_sym(p)
    elif select == consts.ActivationType.EVEN_POLY_8:
        return even_poly8_der_sym(p)
    elif select == consts.ActivationType.EVEN_POLY_10:
        return even_poly10_der_sym(p)
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

    For z3, we use hard tanh instead.
    hard-tanh(x) = max(-1, min(1, x))
    """
    y = x.copy()
    if isinstance(x[0, 0], Z3SYMBOL):
        for idx in range(len(y)):
            y[idx, 0] = z3.If(
                y[idx, 0] > 1.0,
                1.0,
                z3.If(
                    y[idx, 0] < -1.0,
                    -1.0,
                    y[idx, 0]
                ),
            )
    elif isinstance(x[0, 0], DRSYMBOL):
        for idx in range(len(y)):
            y[idx, 0] = dr.tanh(y[idx, 0])
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


def poly3_sym(x):
    """
    Linear - quadratic - cubic activation
    """
    h = int(x.shape[0] / 3)
    x1, x2, x3 = x[:h], x[h: 2 * h], x[2 * h:]
    return np.vstack([x1, np.power(x2, 2), np.power(x3, 3)])


def poly4_sym(x):
    """
    Linear - quadratic - cubic - quartic activation
    """
    h = int(x.shape[0] / 4)
    x1, x2, x3, x4 = x[:h], x[h: 2 * h], x[2 * h: 3 * h], x[3 * h:]
    return np.vstack([x1, np.power(x2, 2), np.power(x3, 3), np.power(x4, 4)])


def poly5_sym(x):
    """
    Linear - quadratic - cubic - quartic - penta activation
    """
    h = int(x.shape[0] / 5)
    x1, x2, x3, x4, x5 = (
        x[:h],
        x[h: 2 * h],
        x[2 * h: 3 * h],
        x[3 * h: 4 * h],
        x[4 * h:],
    )
    return np.vstack(
        [x1, np.power(x2, 2), np.power(x3, 3), np.power(x4, 4), np.power(x5, 5)]
    )


def poly6_sym(x):
    """
    Linear - quadratic - cubic - quartic -penta activation
    """
    h = int(x.shape[0] / 6)
    x1, x2, x3, x4, x5, x6 = (
        x[:h],
        x[h: 2 * h],
        x[2 * h: 3 * h],
        x[3 * h: 4 * h],
        x[4 * h: 5 * h],
        x[5 * h:],
    )
    return np.vstack(
        [
            x1,
            np.power(x2, 2),
            np.power(x3, 3),
            np.power(x4, 4),
            np.power(x5, 5),
            np.power(x6, 6),
        ]
    )


def poly7_sym(x):
    """
    Linear - quadratic - cubic - quartic -penta activation
    """
    h = int(x.shape[0] / 7)
    x1, x2, x3, x4, x5, x6, x7 = (
        x[:h],
        x[h: 2 * h],
        x[2 * h: 3 * h],
        x[3 * h: 4 * h],
        x[4 * h: 5 * h],
        x[5 * h: 6 * h],
        x[6 * h:],
    )
    return np.vstack(
        [
            x1,
            np.power(x2, 2),
            np.power(x3, 3),
            np.power(x4, 4),
            np.power(x5, 5),
            np.power(x6, 6),
            np.power(x7, 7),
        ]
    )


def poly8_sym(x):
    """
    Linear - quadratic - cubic - quartic -penta activation
    """
    h = int(x.shape[0] / 8)
    x1, x2, x3, x4, x5, x6, x7, x8 = (
        x[:h],
        x[h: 2 * h],
        x[2 * h: 3 * h],
        x[3 * h: 4 * h],
        x[4 * h: 5 * h],
        x[5 * h: 6 * h],
        x[6 * h: 7 * h],
        x[7 * h:],
    )
    return np.vstack(
        [
            x1,
            np.power(x2, 2),
            np.power(x3, 3),
            np.power(x4, 4),
            np.power(x5, 5),
            np.power(x6, 6),
            np.power(x7, 7),
            np.power(x8, 8),
        ]
    )


def even_poly4_sym(x):
    """
    Square - quartic activation f(x) = [x^2, x^4]
    """
    h = int(x.shape[0] / 2)
    x1, x2 = (
        x[:h],
        x[h:],
    )
    return np.vstack([np.power(x1, 2), np.power(x2, 4), ])


def even_poly6_sym(x):
    """
    Square - quartic - sextic activation f(x) = [x^2, x^4, x^6]
    """
    h = int(x.shape[0] / 3)
    x1, x2, x3 = (
        x[:h],
        x[h: 2 * h],
        x[2 * h:],
    )
    return np.vstack([np.power(x1, 2), np.power(x2, 4), np.power(x3, 6), ])


def even_poly8_sym(x):
    """
    Square - quartic - sextic - octic activation f(x) = [x^2, x^4, x^6, x^8]
    """
    h = int(x.shape[0] / 4)
    x1, x2, x3, x4 = (
        x[:h],
        x[h: 2 * h],
        x[2 * h: 3 * h],
        x[3 * h:],
    )
    return np.vstack(
        [np.power(x1, 2), np.power(x2, 4), np.power(x3, 6), np.power(x4, 8), ]
    )


def even_poly10_sym(x):
    """
    Square - quartic - sextic - octic - decic activation f(x) = [x^2, x^4, x^6, x^8, x^10]
    """
    h = int(x.shape[0] / 5)
    x1, x2, x3, x4, x5 = (
        x[:h],
        x[h: 2 * h],
        x[2 * h: 3 * h],
        x[3 * h: 4 * h],
        x[4 * h:],
    )
    return np.vstack(
        [
            np.power(x1, 2),
            np.power(x2, 4),
            np.power(x3, 6),
            np.power(x4, 8),
            np.power(x5, 10),
        ]
    )


def rational_sym(x):
    """
    Rational activation f(x) = 1/(1 + x^2)
    """
    return x / (1 + (x ** 2) ** 0.5)


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


def poly2_der_sym(x):
    h = int(len(x) / 2)
    x1, x2 = x[:h], x[h:]
    return np.vstack((np.ones(x1.shape), 2 * x2))


def relu_square_der_z3(x):
    h = int(len(x) / 2)
    x1, x2 = x[:h], x[h:]
    return np.vstack((step(x1), 2 * x2))


def requ_der_z3(x):
    return 2 * relu(x)


def hyper_tan_der(x):
    """
    Derivative of hyperbolic tangent f'(x) = 1 - (tanh(x))^2 = 1 / cosh(x)^2

    For z3, we use hard tanh instead. f'(x) = 0 if x > 1 or x < -1, else 1
    """
    y = x.copy()
    if isinstance(x[0, 0], Z3SYMBOL):
        for idx in range(len(y)):
            y[idx, 0] = z3.If(
                y[idx, 0] > 1.0,
                0.0,
                z3.If(
                    y[idx, 0] < -1.0,
                    0.0,
                    1.0
                ),
            )
    elif isinstance(x[0, 0], DRSYMBOL):
        for idx in range(len(y)):
            y[idx, 0] = 1 / dr.pow(dr.cosh(y[idx, 0]), 2)
    return y


def sinh(x):
    y = x.copy()
    for idx in range(len(y)):
        y[idx, 0] = dr.sinh(y[idx, 0])
    return y


def sigm_der(x):
    y = x.copy()
    if isinstance(x[0, 0], Z3SYMBOL):
        raise NotImplementedError("Z3 not implemented for hyperbolic tangent")
    elif isinstance(x[0, 0], DRSYMBOL):
        for idx in range(len(y)):
            y[idx, 0] = dr.exp(-y[idx, 0]) / dr.pow((1 + dr.exp(-y[idx, 0])), 2)
    return y


def poly3_der_sym(x):
    # linear - quadratic - cubic activation
    h = int(x.shape[0] / 3)
    x1, x2, x3 = x[:h], x[h: 2 * h], x[2 * h:]
    return np.vstack([np.ones((h, 1)), 2 * x2, 3 * np.power(x3, 2)])


def poly4_der_sym(x):
    # # linear - quadratic - cubic - quartic activation
    h = int(x.shape[0] / 4)
    x1, x2, x3, x4 = x[:h], x[h: 2 * h], x[2 * h: 3 * h], x[3 * h:]
    return np.vstack(
        [np.ones((h, 1)), 2 * x2, 3 * np.power(x3, 2), 4 * np.power(x4, 3)]
    )  # torch.pow(x, 2)


def poly5_der_sym(x):
    # # linear - quadratic - cubic - quartic - penta activation
    h = int(x.shape[0] / 5)
    x1, x2, x3, x4, x5 = (
        x[:h],
        x[h: 2 * h],
        x[2 * h: 3 * h],
        x[3 * h: 4 * h],
        x[4 * h:],
    )
    return np.vstack(
        [
            np.ones((h, 1)),
            2 * x2,
            3 * np.power(x3, 2),
            4 * np.power(x4, 3),
            5 * np.power(x5, 4),
        ]
    )


def poly6_der_sym(x):
    # # linear - quadratic - cubic - quartic - penta activation
    h = int(x.shape[0] / 6)
    x1, x2, x3, x4, x5, x6 = (
        x[:h],
        x[h: 2 * h],
        x[2 * h: 3 * h],
        x[3 * h: 4 * h],
        x[4 * h: 5 * h],
        x[5 * h:],
    )
    return np.vstack(
        [
            np.ones((h, 1)),
            2 * x2,
            3 * np.power(x3, 2),
            4 * np.power(x4, 3),
            5 * np.power(x5, 4),
            6 * np.power(x6, 5),
        ]
    )


def poly7_der_sym(x):
    # # linear - quadratic - cubic - quartic - penta activation
    h = int(x.shape[0] / 7)
    x1, x2, x3, x4, x5, x6, x7 = (
        x[:h],
        x[h: 2 * h],
        x[2 * h: 3 * h],
        x[3 * h: 4 * h],
        x[4 * h: 5 * h],
        x[5 * h: 6 * h],
        x[6 * h:],
    )
    return np.vstack(
        [
            np.ones((h, 1)),
            2 * x2,
            3 * np.power(x3, 2),
            4 * np.power(x4, 3),
            5 * np.power(x5, 4),
            6 * np.power(x6, 5),
            7 * np.power(x7, 6),
        ]
    )


def poly8_der_sym(x):
    # # linear - quadratic - cubic - quartic - penta activation
    h = int(x.shape[0] / 8)
    x1, x2, x3, x4, x5, x6, x7, x8 = (
        x[:h],
        x[h: 2 * h],
        x[2 * h: 3 * h],
        x[3 * h: 4 * h],
        x[4 * h: 5 * h],
        x[5 * h: 6 * h],
        x[6 * h: 7 * h],
        x[7 * h:],
    )
    return np.vstack(
        [
            np.ones((h, 1)),
            2 * x2,
            3 * np.power(x3, 2),
            4 * np.power(x4, 3),
            5 * np.power(x5, 4),
            6 * np.power(x6, 5),
            7 * np.power(x7, 6),
            8 * np.power(x8, 7),
        ]
    )


def even_poly4_der_sym(x):
    h = int(x.shape[0] / 2)
    x1, x2 = (
        x[:h],
        x[h:],
    )
    return np.vstack([2 * x1, 4 * np.power(x2, 3), ])


def even_poly6_der_sym(x):
    h = int(x.shape[0] / 3)
    x1, x2, x3 = (
        x[:h],
        x[h: 2 * h],
        x[2 * h:],
    )
    return np.vstack([2 * x1, 4 * np.power(x2, 3), 6 * np.power(x3, 5), ])


def even_poly8_der_sym(x):
    h = int(x.shape[0] / 4)
    x1, x2, x3, x4 = (
        x[:h],
        x[h: 2 * h],
        x[2 * h: 3 * h],
        x[3 * h:],
    )
    return np.vstack(
        [2 * x1, 4 * np.power(x2, 3), 6 * np.power(x3, 5), 8 * np.power(x4, 7), ]
    )


def even_poly10_der_sym(x):
    h = int(x.shape[0] / 5)
    x1, x2, x3, x4, x5 = (
        x[:h],
        x[h: 2 * h],
        x[2 * h: 3 * h],
        x[3 * h: 4 * h],
        x[4 * h:],
    )
    return np.vstack(
        [
            2 * x1,
            4 * np.power(x2, 3),
            6 * np.power(x3, 5),
            8 * np.power(x4, 7),
            10 * np.power(x5, 9),
        ]
    )


def rational_der_sym(x):
    return 1 / (1 + x ** 2)
