import logging
import math
from typing import Generator

import numpy as np
import torch
from torch.optim import Optimizer

from fosco.config import CegisConfig
from fosco.certificates.certificate import Certificate, TrainableCertificate
from fosco.common.domains import Set, Rectangle
from fosco.common.consts import DomainNames
from fosco.common.utils import _set_assertion
from fosco.learner import LearnerNN
from fosco.verifier.verifier import SYMBOL
from fosco.logger import LOGGING_LEVELS
from systems import ControlAffineDynamics

XD = DomainNames.XD.value
XI = DomainNames.XI.value
XU = DomainNames.XU.value
UD = DomainNames.UD.value


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
            vars: dict[str, list],
            domains: dict[str, Set],
            config: CegisConfig,
            verbose: int = 0,
    ) -> None:
        # todo rename vars to x, u
        assert all(
            [sv in vars for sv in ["v", "u"]]
        ), f"Missing symbolic variables, got {vars}"
        self.x_vars = vars["v"]
        self.u_vars = vars["u"]

        self.x_domain: SYMBOL = domains[XD].generate_domain(self.x_vars)
        self.u_set: Rectangle = domains[UD]
        self.u_domain: SYMBOL = domains[UD].generate_domain(self.u_vars)
        self.initial_domain: SYMBOL = domains[XI].generate_domain(self.x_vars)
        self.unsafe_domain: SYMBOL = domains[XU].generate_domain(self.x_vars)

        assert isinstance(
            self.u_set, Rectangle
        ), f"CBF only works with rectangular input domains, got {self.u_set}"
        self.n_vars = len(self.x_vars)
        self.n_controls = len(self.u_vars)

        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(LOGGING_LEVELS[verbose])
        self._logger.debug("CBF initialized")

    def get_constraints(self, verifier, B, B_constr, sigma, sigma_constr, Bdot, Bdot_constr, *args) -> Generator:
        """
        :param verifier: verifier object
        :param B: symbolic formula of the CBF
        :param sigma: symbolic formula of the compensator (not used here)
        :param Bdot: symbolic formula of the CBF derivative (not yet Lie derivative)
        :return: tuple of dictionaries of Barrier conditons
        """
        # todo extend signature with **kwargs

        # Bx >= 0 if x \in initial
        # counterexample: B < 0 and x \in initial
        initial_constr = self._init_constraint_smt(verifier=verifier, B=B, B_constr=B_constr)

        # Bx < 0 if x \in unsafe
        # counterexample: B >= 0 and x \in unsafe
        unsafe_constr = self._unsafe_constraint_smt(verifier=verifier, B=B, B_constr=B_constr)

        # feasibility condition
        # exists u Bdot + alpha * Bx >= 0 if x \in domain
        # counterexample: x \in domain s.t. forall u Bdot + alpha * Bx < 0
        alpha = lambda x: x  # todo make it part of the cbf and pass it in input
        feasible_constr = self._feasibility_constraint_smt(verifier=verifier, B=B, B_constr=B_constr, Bdot=Bdot,
                                                              Bdot_constr=Bdot_constr, alpha=alpha)

        logging.debug(f"initial_constr: {initial_constr}")
        logging.debug(f"unsafe_constr: {unsafe_constr}")
        logging.debug(f"lie_constr: {feasible_constr}")

        for cs in (
                {XI: (initial_constr, self.x_vars), XU: (unsafe_constr, self.x_vars)},
                {XD: (feasible_constr, self.x_vars + self.u_vars)},
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
        initial_constr = _And(initial_constr, self.initial_domain)  # add initial domain constraints
        inital_constr = _And(initial_constr, self.x_domain)  # add state domain constraints (redundant)

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

    def _feasibility_constraint_smt(self, verifier, B, B_constr, Bdot, Bdot_constr, alpha) -> SYMBOL:
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

    @staticmethod
    def _assert_state(domains, data):
        dn = DomainNames
        domain_labels = set(domains.keys())
        data_labels = set(data.keys())
        _set_assertion(
            {dn.XD.value, dn.UD.value, dn.XI.value, dn.XU.value},
            domain_labels,
            "Symbolic Domains",
        )
        _set_assertion(
            {dn.XD.value, dn.XI.value, dn.XU.value}, data_labels, "Data Sets"
        )


class TrainableCBF(TrainableCertificate, ControlBarrierFunction):

    def __init__(
            self,
            system: ControlAffineDynamics,
            vars: dict[str, list],
            domains: dict[str, Set],
            config: CegisConfig,
            verbose: int = 0,
    ):
        super(TrainableCBF, self).__init__(system=system, vars=vars, domains=domains,
                                           config=config, verbose=verbose)

        # loss parameters
        self.loss_relu = config.LOSS_RELU
        self.epochs = config.N_EPOCHS

        # process loss margins
        loss_keys = ["init", "unsafe", "lie"]
        if isinstance(config.LOSS_MARGINS, float):
            loss_margins = {k: config.LOSS_MARGINS for k in loss_keys}
        else:
            assert all(
                [k in config.LOSS_MARGINS for k in loss_keys]
            ), f"Missing loss margin, got {config.LOSS_MARGINS}"
            loss_margins = config.LOSS_MARGINS
        self.loss_margins = loss_margins

        # process loss weights
        if isinstance(config.LOSS_WEIGHTS, float):
            loss_weights = {k: config.LOSS_WEIGHTS for k in loss_keys}
        else:
            assert all(
                [k in config.LOSS_WEIGHTS for k in loss_keys]
            ), f"Missing loss weight, got {config.LOSS_WEIGHTS}"
            loss_weights = config.LOSS_WEIGHTS
        self.loss_weights = loss_weights

        # regularization on net gradient
        self.loss_netgrad_weight = config.LOSS_NETGRAD_WEIGHT

    def learn(
            self,
            learner: LearnerNN,
            optimizers: Optimizer,
            datasets: dict,
            f_torch: callable,
    ) -> dict[str, float | np.ndarray]:
        """
        Updates the CBF model.

        :param learner: LearnerNN object
        :param optimizer: torch optimizer
        :param datasets: dictionary of (string,torch.Tensor) pairs
        :param f_torch: callable
        """
        # todo extend signature with **kwargs

        if not optimizers:
            return {}

        condition_old = False
        i1 = datasets[XD].shape[0]
        i2 = datasets[XI].shape[0]
        # samples = torch.cat([s for s in S.values()])
        label_order = [XD, XI, XU]
        state_samples = torch.cat(
            [datasets[label][:, : self.n_vars] for label in label_order]
        )
        U_d = datasets[XD][:, self.n_vars: self.n_vars + self.n_controls]

        losses, accuracies, infos = {}, {}, {}
        for t in range(self.epochs):
            optimizers["barrier"].zero_grad()

            # net gradient
            B = learner.net(state_samples)
            gradB = learner.net.gradient(state_samples)

            B_d = B[:i1, 0]
            B_i = B[i1: i1 + i2, 0]
            B_u = B[i1 + i2:, 0]

            # compute lie derivative
            assert (
                    B_d.shape[0] == U_d.shape[0]
            ), f"expected pairs of state,input data. Got {B_d.shape[0]} and {U_d.shape[0]}"
            X_d = state_samples[:i1]
            gradB_d = gradB[:i1]
            Sdot_d = f_torch(X_d, U_d)
            Bdot_d = torch.sum(torch.mul(gradB_d, Sdot_d), dim=1)

            loss, losses, accuracies = self.compute_loss(
                B_i, B_u, B_d, Bdot_d, alpha=1.0
            )

            # regularization net gradient
            netgrad_sos = torch.sum(torch.square(gradB))
            netgrad_loss = self.loss_netgrad_weight * netgrad_sos
            losses["netgrad_loss"] = netgrad_loss.item()
            loss += netgrad_loss

            # infos
            infos = {
                "netgrad_sos": netgrad_sos.item(),
            }

            if t % math.ceil(self.epochs / 10) == 0 or self.epochs - t < 10:
                # log_loss_acc(t, loss, accuracy, learner.verbose)
                logging.debug(f"Epoch {t}: loss={loss}, accuracy={accuracies}")

            # early stopping after 2 consecutive epochs with ~100% accuracy
            condition = all(acc >= 99.9 for name, acc in accuracies.items())
            if condition and condition_old:
                break
            condition_old = condition

            loss.backward()
            optimizers["barrier"].step()

        return {
            "loss": losses,
            "accuracy": accuracies,
            "info": infos,
        }

    def compute_loss(
            self,
            B_i: torch.Tensor,
            B_u: torch.Tensor,
            B_d: torch.Tensor,
            Bdot_d: torch.Tensor,
            alpha: torch.Tensor | float,
    ) -> tuple[torch.Tensor, dict, dict]:
        # todo make this private
        """Computes loss function for CBF and its accuracy w.r.t. the batch of data.

        Args:
            B_i (torch.Tensor): Barrier values for initial set
            B_u (torch.Tensor): Barrier values for unsafe set
            B_d (torch.Tensor): Barrier values for domain
            Bdot_d (torch.Tensor): Barrier derivative values for domain
            alpha (torch.Tensor): coeff. linear class-k function, f(x) = alpha * x, for alpha in R_+

        Returns:
            tuple[torch.Tensor, float]: loss and accuracy
        """
        assert (
                Bdot_d is None or B_d.shape == Bdot_d.shape
        ), f"B_d and Bdot_d must have the same shape, got {B_d.shape} and {Bdot_d.shape}"
        assert isinstance(
            self.loss_margins, dict
        ), f"Expected loss margins as dict, got {type(self.loss_margins)}"
        assert isinstance(
            self.loss_weights, dict
        ), f"Expected loss weights as dict, got {type(self.loss_weights)}"

        margin_init = self.loss_margins["init"]
        margin_unsafe = self.loss_margins["unsafe"]
        margin_lie = self.loss_margins["lie"]

        weight_init = self.loss_weights["init"]
        weight_unsafe = self.loss_weights["unsafe"]
        weight_lie = self.loss_weights["lie"]

        accuracy_i = (B_i >= margin_init).count_nonzero().item()
        accuracy_u = (B_u < -margin_unsafe).count_nonzero().item()
        accuracy_d = (Bdot_d + alpha * B_d >= margin_lie).count_nonzero().item()

        percent_accuracy_init = 100 * accuracy_i / B_i.shape[0]
        percent_accuracy_unsafe = 100 * accuracy_u / B_u.shape[0]
        percent_accuracy_lie = 100 * accuracy_d / Bdot_d.shape[0]

        # penalize B_i < 0
        init_loss = weight_init * (self.loss_relu(margin_init - B_i)).mean()
        # penalize B_u > 0
        unsafe_loss = weight_unsafe * (self.loss_relu(B_u + margin_unsafe)).mean()
        # penalize dB_d + alpha * B_d < 0
        lie_loss = (
                weight_lie * (self.loss_relu(margin_lie - (Bdot_d + alpha * B_d))).mean()
        )

        loss = init_loss + unsafe_loss + lie_loss

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
        logging.debug("Dataset Accuracy:")
        logging.debug("\n".join([f"{k}:{v}" for k, v in accuracy.items()]))

        return loss, losses, accuracy
