import torch

from fosco.common import consts


# Activation function
def activation(select: consts.ActivationType, p):
    """
    :param select: enum selects the type of activation
    :param p: the layer
    :return: calls the activation fcn and returns the layer after activation
    """
    if select == consts.ActivationType.IDENTITY:
        return identity(p)
    elif select == consts.ActivationType.RELU:
        return relu(p)
    elif select == consts.ActivationType.LINEAR:
        return p
    elif select == consts.ActivationType.SQUARE:
        return square(p)
    elif select == consts.ActivationType.REQU:
        return requ(p)
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
        return rational(p)
    else:
        raise ValueError(f"Activation {select} not implemented")


def activation_der(select: consts.ActivationType, p):
    """
    :param select: enum selects the type of activation
    :param p: the layer
    :return: calls the activation fcn and returns the layer after activation
    """
    if select == consts.ActivationType.IDENTITY:
        return identity_der(p)
    elif select == consts.ActivationType.RELU:
        return step(p)
    elif select == consts.ActivationType.LINEAR:
        return torch.ones(p.shape)
    elif select == consts.ActivationType.SQUARE:
        return 2 * p
    elif select == consts.ActivationType.REQU:
        return 2 * relu(p)
    elif select == consts.ActivationType.TANH:
        return hyper_tan_der(p)
    elif select == consts.ActivationType.HTANH:
        return hard_hyper_tan_der(p)
    elif select == consts.ActivationType.SIGMOID:
        return sigm_der(p)
    elif select == consts.ActivationType.HSIGMOID:
        return hard_sigm_der(p)
    elif select == consts.ActivationType.SOFTPLUS:
        return softplus_der(p)
    elif select == consts.ActivationType.COSH:
        return sinh(p)
    elif select == consts.ActivationType.RATIONAL:
        return rational_der(p)
    else:
        raise ValueError(f"Activation {select} not implemented")


##################################################################
# ACTIVATIONS
##################################################################


def identity(x):
    return x


def relu(x):
    return torch.relu(x)


def square(x):
    return torch.pow(x, 2)


# ReQU: Rectified Quadratic Unit
def requ(x):
    return x * torch.relu(x)


def hyper_tan(x):
    return torch.tanh(x)


def hard_hyper_tan(x):
    return torch.nn.functional.hardtanh(x)


def sigm(x):
    return torch.sigmoid(x)


def hard_sigm(x):
    return torch.nn.functional.hardsigmoid(x)


def softplus(x):
    return torch.nn.functional.softplus(x)


def cosh(x):
    return torch.cosh(x) - 1


def rational(x):
    # tanh approximation
    return x / (1 + torch.sqrt(torch.pow(x, 2)))


##################################################################
# DERIVATIVES
##################################################################


def identity_der(x):
    return torch.ones(x.shape)


def step(x):
    sign = torch.sign(x)
    return torch.relu(sign)


def poly2_der(x):
    h = int(x.shape[1] / 2)
    x1, x2 = x[:, :h], x[:, h:]
    return torch.cat([torch.ones(x1.shape), 2 * x2], dim=1)


def relu_square_der(x):
    h = int(x.shape[1] / 2)
    x1, x2 = x[:, :h], x[:, h:]
    return torch.cat([step(x1), 2 * x2], dim=1)  # torch.pow(x, 2)


def hyper_tan_der(x):
    return torch.ones(x.shape) - torch.pow(torch.tanh(x), 2)


def hard_hyper_tan_der(x):
    y_mask = (x < -1.0) | (x > 1.0)
    dydx = torch.ones(x.shape)
    dydx[y_mask] = 0
    return dydx


def sigm_der(x):
    y = sigm(x)
    return y * (torch.ones(x.shape) - y)


def hard_sigm_der(x):
    y_mask = (x < -3.0) | (x > 3.0)
    dydx = 1 / 6 * torch.ones(x.shape)
    dydx[y_mask] = 0
    return dydx


def softplus_der(x):
    return torch.sigmoid(x)


def sinh(x):
    return torch.sinh(x)


def rational_der(x):
    return 1 / (1 + torch.pow(x, 2))
