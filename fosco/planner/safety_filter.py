import numpy as np
import torch

import cvxpy as cp


class CBFSafetyFilter:
    def __init__(self, env, h_model: torch.nn.Module):
        self._input_lb = env.action_space.low
        self._input_ub = env.action_space.high
        assert len(self._input_lb) == len(
            self._input_ub
        ), "action lower and upper bounds must have same dimension"

        self.h = h_model  # todo: replace torch model with TorchMLP
        self.f = env._base_system.fx_torch
        self.g = env._base_system.gx_torch

        self.prob = self.setup_problem()

    def setup_problem(self) -> cp.Problem:
        # todo: infer number of actions from env spec
        self.u_nom = cp.Parameter(2)

        self.Lfhx = cp.Parameter(1)
        self.Lghx = cp.Parameter(2)
        self.alpha_hx = cp.Parameter(1)

        # todo: infer number of actions from env spec
        self.u = cp.Variable(2)
        self.d = cp.Variable()
        constraints = []

        # input constraints
        # todo: infer input bounds from env spec
        constraints += [self.u >= -5.0, self.u <= 5.0]  # todo

        # hard cbf constraint
        constraints += [self.Lfhx + self.Lghx @ self.u + self.alpha_hx + self.d >= 0]

        objective = cp.Minimize(cp.norm(self.u) + 1e4 * self.d ** 2)
        self.prob = cp.Problem(objective, constraints)

    def __call__(
        self, observation, action
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_joint = observation

        self.u_nom.value = action

        tx_joint = torch.from_numpy(x_joint).float()
        dhdx = self.h.compute_net_gradnet(tx_joint)[1].detach().numpy()[0]
        hx = self.h(tx_joint).detach().numpy()[0]
        alpha = 1.0  # todo: take alpha from model

        self.Lfhx.value = dhdx @ self.f(x_joint)
        self.Lghx.value = dhdx @ self.g(x_joint)
        self.alpha_hx.value = alpha * hx

        self.prob.solve(verbose=False, solver=cp.CLARABEL)

        if self.prob.status != cp.OPTIMAL:
            print("WARNING: CBF QP not solved to optimality!")

        if self.u.value is None:
            u, d = np.zeros(2), 100.0
        else:
            u, d = self.u.value, self.d.value

        return u, d
