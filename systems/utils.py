from typing import Callable

import numpy as np


def lie_derivative_fn(certificate, f, ctrl) -> Callable[[np.ndarray], np.ndarray]:
    def lie_derivative(x):
        _, grad_net = certificate.compute_net_gradnet(x)
        xdot = f(x, ctrl(x))
        grad_net = grad_net.reshape(-1, 1, grad_net.shape[-1])  # (batch, 1, dim)
        xdot = xdot.reshape(-1, xdot.shape[-1], 1)  # (batch, dim, 1)
        # (batch, 1, dim) @ (batch, dim, 1) = (batch, 1, 1)
        return (grad_net @ xdot).reshape(-1, 1)

    return lie_derivative
