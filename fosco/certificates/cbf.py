import math
from typing import Generator, Callable

import numpy as np
import torch
from torch.optim import Optimizer

from fosco.config import CegisConfig
from fosco.certificates.certificate import Certificate, TrainableCertificate
from fosco.common.domains import Set, Rectangle
from fosco.common.consts import DomainName, LossReLUType, TimeDomain
from fosco.common.utils import _set_assertion
from fosco.models import TorchSymDiffFn
from fosco.verifier.types import SYMBOL
from fosco.systems import ControlAffineDynamics

XD = DomainName.XD.value
XI = DomainName.XI.value
XU = DomainName.XU.value
UD = DomainName.UD.value


class ControlBarrierFunction(Certificate):
    """
    Certifies Safety for continuous time controlled systems with control affine dynamics.

    Arguments:
        system {ControlAffineDynamics}: control affine dynamics
        vars {dict}: dictionary of symbolic variables
        domains {dict}: dictionary of (string,domain) pairs
        config {CegisConfig}: configuration object
        verbose {int}: verbosity level
    """

    def __init__(
        self,
        system: ControlAffineDynamics,
        variables: dict[str, list],
        domains: dict[str, Set],
        verbose: int = 0,
    ) -> None:
        # todo rename vars to x, u
        self.x_vars = variables["v"]
        self.u_vars = variables["u"]

        self.x_domain: SYMBOL = domains[XD].generate_domain(self.x_vars)
        self.u_set: Rectangle = domains[UD]
        self.u_domain: SYMBOL = domains[UD].generate_domain(self.u_vars)
        self.initial_domain: SYMBOL = domains[XI].generate_domain(self.x_vars)
        self.unsafe_domain: SYMBOL = domains[XU].generate_domain(self.x_vars)

        self.n_vars = len(self.x_vars)
        self.n_controls = len(self.u_vars)

        super(ControlBarrierFunction, self).__init__(
            system=system, variables=variables, domains=domains, verbose=verbose
        )

    def _assert_state(self) -> None:
        dn = DomainName
        domain_labels = set(self.domains.keys())

        _set_assertion(
            {dn.XD.value, dn.UD.value, dn.XI.value, dn.XU.value},
            domain_labels,
            "Symbolic Domains",
        )

        assert all(
            [sv in self.variables for sv in ["v", "u"]]
        ), f"Missing symbolic variables, got {self.variables}"

        assert isinstance(
            self.u_set, Rectangle
        ), f"CBF only works with rectangular input domains, got {self.u_set}"

    def get_constraints(
        self,
        verifier,
        B,
        B_constr,
        B_vars,
        sigma,
        sigma_constr,
        sigma_vars,
        Bdot,
        Bdot_constr,
        Bdot_vars,
        *args,
    ) -> Generator:
        """
        :param verifier: verifier object
        :param B: symbolic formula of the CBF
        :param sigma: symbolic formula of the compensator (not used here)
        :param Bdot: symbolic formula of the CBF derivative (not yet Lie derivative)
        :return: tuple of dictionaries of Barrier conditons
        """
        # todo extend signature with **kwargs
        assert isinstance(B_vars, list) and all(
            [isinstance(v, SYMBOL) for v in B_vars]
        ), f"Expected list of SYMBOL, got {B_vars}"
        assert isinstance(Bdot_vars, list) and all(
            [isinstance(v, SYMBOL) for v in Bdot_vars]
        ), f"Expected list of SYMBOL, got {Bdot_vars}"

        # Bx >= 0 if x \in initial
        # counterexample: B < 0 and x \in initial
        initial_vars = self.x_vars
        initial_aux_vars = [v for v in B_vars if v not in initial_vars]
        initial_constr = self._init_constraint_smt(
            verifier=verifier, B=B, B_constr=B_constr
        )

        # Bx < 0 if x \in unsafe
        # counterexample: B >= 0 and x \in unsafe
        unsafe_vars = self.x_vars
        unsafe_aux_vars = [v for v in B_vars if v not in unsafe_vars]
        unsafe_constr = self._unsafe_constraint_smt(
            verifier=verifier, B=B, B_constr=B_constr
        )

        # feasibility condition
        # exists u Bdot + alpha * Bx >= 0 if x \in domain
        # counterexample: x \in domain s.t. forall u Bdot + alpha * Bx < 0
        alpha = lambda x: x  # todo make it part of the cbf and pass it in input
        feasible_vars = self.x_vars + self.u_vars
        feasible_aux_vars = [v for v in B_vars + Bdot_vars if v not in feasible_vars]
        feasible_constr = self._feasibility_constraint_smt(
            verifier=verifier,
            B=B,
            B_constr=B_constr,
            Bdot=Bdot,
            Bdot_constr=Bdot_constr,
            alpha=alpha,
        )

        # self._logger.debug(f"initial_constr: {initial_constr}")
        # self._logger.debug(f"unsafe_constr: {unsafe_constr}")
        # self._logger.debug(f"lie_constr: {feasible_constr}")

        for cs in (
            {XD: (feasible_constr, feasible_vars, feasible_aux_vars)},
            {
                XI: (initial_constr, initial_vars, initial_aux_vars),
                XU: (unsafe_constr, unsafe_vars, unsafe_aux_vars),
            },
        ):
            yield cs

    def _init_constraint_smt(self, verifier, B, B_constr) -> SYMBOL:
        """
        Initial constraint for CBF: the barrier must be non-negative in the initial set.
        spec: Bx >= 0 if x \in initial
        counterexample: B < 0 and x \in initial
        """
        _And = verifier.solver_fncts()["And"]

        initial_constr = B < 0
        for c in B_constr:
            initial_constr = _And(initial_constr, c)
        initial_constr = _And(
            initial_constr, self.initial_domain
        )  # add initial domain constraints
        inital_constr = _And(
            initial_constr, self.x_domain
        )  # add state domain constraints (redundant)

        return inital_constr

    def _unsafe_constraint_smt(self, verifier, B, B_constr) -> SYMBOL:
        """
        Unsafe constraint for CBF: the barrier must be negative in the unsafe set.

        spec: Bx < 0 if x \in unsafe
        counterexample: B >= 0 and x \in unsafe
        """
        _And = verifier.solver_fncts()["And"]
        unsafe_constr = B >= 0

        for c in B_constr:
            unsafe_constr = _And(unsafe_constr, c)
        unsafe_constr = _And(unsafe_constr, self.unsafe_domain)
        unsafe_constr = _And(unsafe_constr, self.x_domain)
        return unsafe_constr

    def _feasibility_constraint_smt(
        self, verifier, B, B_constr, Bdot, Bdot_constr, alpha
    ) -> SYMBOL:
        """
        Feasibility constraint

        spec: exists u Bdot + alpha * Bx >= 0 if x \in domain
        counterexample: x \in domain s.t. forall u Bdot + alpha * Bx < 0

        Note: trick for tractable verification using vertices of input convex-hull
        counterexample: x \in domain and AND_v (u=v and Bdot + alpha * Bx < 0)
        """
        _And = verifier.solver_fncts()["And"]
        _Substitute = verifier.solver_fncts()["Substitute"]
        _RealVal = verifier.solver_fncts()["RealVal"]

        u_vertices = self.u_set.get_vertices()
        lie_constr = B >= 0
        for c in B_constr:
            lie_constr = _And(lie_constr, c)
        lie_constr = _And(lie_constr, self.x_domain)
        for u_vert in u_vertices:
            vertex_constr = Bdot + alpha(B) < 0
            for c in Bdot_constr:
                vertex_constr = _And(vertex_constr, c)
            for u_var, u_val in zip(self.u_vars, u_vert):
                vertex_constr = _Substitute(vertex_constr, (u_var, _RealVal(u_val)))
            lie_constr = _And(lie_constr, vertex_constr)

        return lie_constr


class TrainableCBF(TrainableCertificate, ControlBarrierFunction):

    @staticmethod
    def learn(
        learner,
        optimizers: Optimizer,
        datasets: dict,
        f_torch: callable,
        n_vars: int,
        n_controls: int,
        **kwargs,
    ) -> dict[str, float | np.ndarray | dict]:
        """
        Updates the CBF model.

        :param learner: LearnerNN object
        :param optimizers: torch optimizers
        :param datasets: dictionary of (string,torch.Tensor) pairs
        :param f_torch: callable
        """

        if not optimizers:
            return {}

        condition_old = False

        i1 = datasets[XD].shape[0]
        i2 = datasets[XI].shape[0]

        label_order = [XD, XI, XU]
        state_samples = torch.cat(
            [datasets[label][:, :n_vars] for label in label_order]
        )

        losses, accuracies, infos = {}, {}, {}
        for t in range(learner.epochs):
            optimizers["barrier"].zero_grad()

            # net gradient
            B = learner.net(state_samples)

            B_d = B[:i1, 0]
            B_i = B[i1 : i1 + i2, 0]
            B_u = B[i1 + i2 :, 0]

            # compute lie derivative
            Bdot_d = TrainableCBF._compute_barrier_difference(
                X_d=datasets[XD][:, :n_vars],
                U_d=datasets[XD][:, n_vars : n_vars + n_controls],
                barrier=learner.net,
                f_torch=f_torch,
            )

            loss, losses, accuracies = TrainableCBF._compute_loss(
                learner=learner, B_i=B_i, B_u=B_u, B_d=B_d, Bdot_d=Bdot_d, alpha=1.0
            )

            if t % math.ceil(learner.epochs / 10) == 0 or learner.epochs - t < 10:
                # log_loss_acc(t, loss, accuracy, learner.verbose)
                learner._logger.debug(f"Epoch {t}: loss={loss}, accuracy={accuracies}")

            # early stopping after 2 consecutive epochs with ~100% accuracy
            condition = all(acc >= 99.9 for name, acc in accuracies.items())
            if condition and condition_old:
                break
            condition_old = condition

            loss.backward()
            optimizers["barrier"].step()
            infos = {}

        return {
            "loss": losses,
            "accuracy": accuracies,
            "info": infos,
        }

    @staticmethod
    def _compute_barrier_difference(
        X_d: torch.Tensor, U_d: torch.Tensor, barrier: TorchSymDiffFn, f_torch: Callable
    ) -> torch.Tensor:
        """
        Computes the change over time of the barrier function subject to the system dynamics.
        This is the Lie derivative of the barrier function for continuous-time systems.

        Args:
            X_d (torch.Tensor): batch of states of shape (batch_size, n_vars)
            U_d (torch.Tensor): batch of inputs of shape (batch_size, n_controls)
            barrier (TorchSymDiffFn): barrier function
            f_torch (Callable): system dynamics, xdot=f(x,u) for ct, x_{k+1}=f(x_k,u_k) for dt
        """
        assert (
            X_d.shape[0] == U_d.shape[0]
        ), f"expected pairs of state,input data. Got {X_d.shape[0], U_d.shape[0]}"

        # Lie derivative of B: dB/dt = dB/dx * dx/dt
        db_dx = barrier.gradient(X_d)
        dx_dt = f_torch(X_d, U_d)
        db_dt = torch.sum(torch.mul(db_dx, dx_dt), dim=1)

        return db_dt

    @staticmethod
    def _compute_loss(
        learner,
        B_i: torch.Tensor,
        B_u: torch.Tensor,
        B_d: torch.Tensor,
        Bdot_d: torch.Tensor,
        alpha: torch.Tensor | float,
    ) -> tuple[torch.Tensor, dict, dict]:
        """Computes loss function for CBF and its accuracy w.r.t. the batch of data.

        Args:
            B_i (torch.Tensor): Barrier values for initial set
            B_u (torch.Tensor): Barrier values for unsafe set
            B_d (torch.Tensor): Barrier values for domain
            Bdot_d (torch.Tensor): Barrier time-difference for domain set
            alpha (torch.Tensor): coeff. linear class-k function, f(x) = alpha * x, for alpha in R_+

        Returns:
            tuple[torch.Tensor, float]: loss and accuracy
        """
        assert (
            Bdot_d is None or B_d.shape == Bdot_d.shape
        ), f"B_d and Bdot_d must have the same shape, got {B_d.shape} and {Bdot_d.shape}"
        assert isinstance(
            learner.loss_margins, dict
        ), f"Expected loss margins as dict, got {type(learner.loss_margins)}"
        assert isinstance(
            learner.loss_weights, dict
        ), f"Expected loss weights as dict, got {type(learner.loss_weights)}"

        margin_init = learner.loss_margins["init"]
        margin_unsafe = learner.loss_margins["unsafe"]
        margin_lie = learner.loss_margins["lie"]

        weight_init = learner.loss_weights["init"]
        weight_unsafe = learner.loss_weights["unsafe"]
        weight_lie = learner.loss_weights["lie"]
        weight_conservative_b = learner.loss_weights["conservative_b"]

        accuracy_i = (B_i >= margin_init).count_nonzero().item()
        accuracy_u = (B_u < -margin_unsafe).count_nonzero().item()
        accuracy_d = (Bdot_d + alpha * B_d >= margin_lie).count_nonzero().item()

        percent_accuracy_init = 100 * accuracy_i / B_i.shape[0]
        percent_accuracy_unsafe = 100 * accuracy_u / B_u.shape[0]
        percent_accuracy_lie = 100 * accuracy_d / Bdot_d.shape[0]

        # penalize B_i < 0
        init_loss = weight_init * (learner.loss_relu(margin_init - B_i)).mean()
        # penalize B_u > 0
        unsafe_loss = weight_unsafe * (learner.loss_relu(B_u + margin_unsafe)).mean()
        # penalize when B_d > 0 and dB_d + alpha * B_d < 0
        loss_cond = torch.min(B_d - margin_lie, margin_lie - (Bdot_d + alpha * B_d))
        lie_loss = weight_lie * (learner.loss_relu(loss_cond)).mean()

        # regularization losses
        # penalize negative B (conservative)
        loss_B_neg = learner.loss_relu(-B_d).mean()  # penalize B_d < 0
        loss_B_conservative = weight_conservative_b * loss_B_neg

        loss = init_loss + unsafe_loss + lie_loss + loss_B_conservative

        losses = {
            "init_loss": init_loss.item(),
            "unsafe_loss": unsafe_loss.item(),
            "lie_loss": lie_loss.item(),
            "tot_loss": loss.item(),
        }

        accuracy = {
            "accuracy_init": percent_accuracy_init,
            "accuracy_unsafe": percent_accuracy_unsafe,
            "accuracy_derivative": percent_accuracy_lie,
        }

        # debug
        learner._logger.debug("Dataset Accuracy:")
        learner._logger.debug(
            "\n" + "\n".join([f"{k}:{v}" for k, v in accuracy.items()])
        )

        return loss, losses, accuracy
