import logging
import math
from typing import Generator

import numpy as np
import torch
from torch.optim import Optimizer

from fosco.certificates.cbf import ControlBarrierFunction, TrainableCBF
from fosco.certificates.certificate import TrainableCertificate
from fosco.config import CegisConfig
from fosco.certificates import Certificate
from fosco.common.domains import Set, Rectangle
from fosco.common.consts import DomainNames
from fosco.common.utils import _set_assertion
from fosco.learner import LearnerCT
from fosco.verifier import SYMBOL
from logger import LOGGING_LEVELS
from systems import ControlAffineControllableDynamicalModel

XD = DomainNames.XD.value
XI = DomainNames.XI.value
XU = DomainNames.XU.value
UD = DomainNames.UD.value
ZD = DomainNames.ZD.value


class RobustControlBarrierFunction(ControlBarrierFunction):
    """
    Certifies Safety for continuous time controlled systems with control affine dynamics.

    Note: CBF use different conventions.
    B(Xi)>0, B(Xu)<0, Bdot(Xd) > -alpha(B(Xd)) for alpha class-k function

    Arguments:
        vars {dict}: dictionary of symbolic variables
        domains {dict}: dictionary of (string,domain) pairs
    """

    def __init__(
        self,
        system: ControlAffineControllableDynamicalModel,
        vars: dict[str, list],
        domains: dict[str, Set],
        config: CegisConfig,
        verbose: int = 0,
    ) -> None:
        assert all(
            [sv in vars for sv in ["v", "u", "z"]]
        ), f"Missing symbolic variables, got {vars}"

        super().__init__(system=system, vars=vars, domains=domains, config=config, verbose=verbose)

        self.z_vars = vars["z"]
        self.z_domain: SYMBOL = domains[ZD].generate_domain(self.z_vars)
        self.n_uncertain = len(self.z_vars)

    def get_constraints(self, verifier, B, sigma, Bdot, Bdotz) -> Generator:
        """
        :param verifier: verifier object
        :param B: symbolic formula of the CBF
        :param sigma: symbolic formula of compensator sigma
        :param Bdot: symbolic formula of the CBF derivative (not yet Lie derivative)
        :return: tuple of dictionaries of Barrier conditons
        """

        # todo extend signature with **kwargs
        _True = verifier.solver_fncts()["True"]
        _And = verifier.solver_fncts()["And"]
        _Or = verifier.solver_fncts()["Or"]
        _Not = verifier.solver_fncts()["Not"]
        _Exists = verifier.solver_fncts()["Exists"]
        _ForAll = verifier.solver_fncts()["ForAll"]
        _Substitute = verifier.solver_fncts()["Substitute"]
        _RealVal = verifier.solver_fncts()["RealVal"]

        alpha = lambda x: 1.0 * x

        # initial condition
        # Bx >= 0 if x \in initial
        # counterexample: B < 0 and x \in initial
        initial_constr = _And(B < 0, self.initial_domain)
        # add domain constraints
        inital_constr = _And(initial_constr, self.x_domain)

        # unsafe condition
        # Bx < 0 if x \in unsafe
        # counterexample: B >= 0 and x \in unsafe
        unsafe_constr = _And(B >= 0, self.unsafe_domain)
        # add domain constraints
        unsafe_constr = _And(unsafe_constr, self.x_domain)

        # feasibility condition
        # exists u Bdot + alpha * Bx >= 0 if x \in domain
        # counterexample: x \in domain s.t. forall u Bdot + alpha * Bx < 0
        #
        # note: smart trick for tractable verification using vertices of input convex-hull
        # counterexample: x \in domain and AND_v (u=v and Bdot + alpha * Bx < 0)
        u_vertices = self.u_set.get_vertices()
        lie_constr = self.x_domain
        for u_vert in u_vertices:
            vertex_constr = Bdot - sigma + alpha(B) < 0
            for u_var, u_val in zip(self.u_vars, u_vert):
                vertex_constr = _Substitute(vertex_constr, (u_var, _RealVal(u_val)))
            lie_constr = _And(lie_constr, vertex_constr)

        # robustness constraint
        # spec := forall z forall u (Bdot + alpha * Bx >= 0 implies Bdot(z) + alpha * B(z) >= 0)
        # counterexample: x \in xdomain and z \in zdomain and u \in udomain and
        #                 Bdot + alpha * Bx >= 0 and Bdot(z) + alpha * B(z) < 0
        is_nominal_safe = Bdot - sigma + alpha(B) >= 0
        is_uncertain_unsafe = Bdotz + alpha(B) < 0
        robust_constr = _And(is_nominal_safe, is_uncertain_unsafe)
        # add domain constraints
        robust_constr = _And(robust_constr, self.x_domain)
        robust_constr = _And(robust_constr, self.z_domain)
        robust_constr = _And(robust_constr, self.u_domain)

        logging.debug(f"inital_constr: {inital_constr}")
        logging.debug(f"unsafe_constr: {unsafe_constr}")
        logging.debug(f"lie_constr: {lie_constr}")
        logging.debug(f"robust_constr: {robust_constr}")

        for cs in (
            {XI: (inital_constr, self.x_vars), XU: (unsafe_constr, self.x_vars)},
            {
                XD: (lie_constr, self.x_vars + self.u_vars + self.z_vars),
                ZD: (robust_constr, self.x_vars + self.u_vars + self.z_vars),
            },
        ):
            yield cs

    @staticmethod
    def _assert_state(domains, data):
        dn = DomainNames
        domain_labels = set(domains.keys())
        data_labels = set(data.keys())
        _set_assertion(
            {dn.XD.value, dn.UD.value, dn.ZD.value, dn.XI.value, dn.XU.value},
            domain_labels,
            "Symbolic Domains",
        )
        _set_assertion(
            {dn.XD.value, dn.XI.value, dn.XU.value, dn.ZD.value},
            data_labels,
            "Data Sets",
        )


class TrainableRCBF(TrainableCBF, RobustControlBarrierFunction):

    def __init__(
        self,
        system: ControlAffineControllableDynamicalModel,
        vars: dict[str, list],
        domains: dict[str, Set],
        config: CegisConfig,
        verbose: int = 0,
    ) -> None:
        super(TrainableRCBF, self).__init__(
            system=system, vars=vars, domains=domains, config=config, verbose=verbose
        )

        # add extra loss margin for uncertainty loss
        if isinstance(config.LOSS_MARGINS, float):
            self.loss_margins["robust"] = config.LOSS_MARGINS
        else:
            assert "robust" in config.LOSS_MARGINS, f"Missing loss margin, got {config.LOSS_MARGINS}"
            self.loss_margins["robust"] = config.LOSS_MARGINS["robust"]

        # add extra loss weight for uncertainty loss
        if isinstance(config.LOSS_WEIGHTS, float):
            self.loss_weights["robust"] = config.LOSS_WEIGHTS
        else:
            assert "robust" in config.LOSS_WEIGHTS, f"Missing loss weight, got {config.LOSS_WEIGHTS}"
            self.loss_weights["robust"] = config.LOSS_WEIGHTS["robust"]

    def learn(
        self,
        learner: LearnerCT,
        optimizer: Optimizer,
        datasets: dict,
        f_torch: callable,
    ) -> dict[str, float | np.ndarray | dict]:
        """
        Updates the CBF model.

        :param learner: LearnerNN object
        :param optimizer: torch optimizer
        :param datasets: dictionary of (string,torch.Tensor) pairs
        :param f_torch: callable
        """

        # todo extend signature with **kwargs
        condition_old = False
        i1 = datasets[XD].shape[0]
        i2 = datasets[XI].shape[0]
        i3 = datasets[XU].shape[0]

        label_order = [XD, XI, XU, ZD]
        state_samples = torch.cat(
            [datasets[label][:, : self.n_vars] for label in label_order]
        )
        U_d = datasets[XD][:, self.n_vars : self.n_vars + self.n_controls]
        Z_d = datasets[XD][
            :,
            self.n_vars
            + self.n_controls : self.n_vars
            + self.n_controls
            + self.n_uncertain,
        ]

        X_dz = datasets[ZD][:, : self.n_vars]
        U_dz = datasets[ZD][:, self.n_vars : self.n_vars + self.n_controls]
        Z_dz = datasets[ZD][
            :,
            self.n_vars
            + self.n_controls : self.n_vars
            + self.n_controls
            + self.n_uncertain,
        ]

        losses, accuracies = {}, {}
        for t in range(self.epochs):
            optimizer.zero_grad()

            # net gradient
            B, gradB = learner.net.compute_net_gradnet(state_samples)
            sigma = learner.xsigma(state_samples)

            B_d = B[:i1, 0]
            B_i = B[i1 : i1 + i2, 0]
            B_u = B[i1 + i2 : i1 + i2 + i3, 0]

            # compute lie derivative on lie dataset
            assert (
                B_d.shape[0] == U_d.shape[0]
            ), f"expected pairs of state,input data. Got {B_d.shape[0]} and {U_d.shape[0]}"
            X_d = state_samples[:i1]
            gradB_d = gradB[:i1]
            sigma_d = sigma[:i1, 0]
            Sdot_d = f_torch(X_d, U_d, Z_d, only_nominal=True)
            Bdot_d = torch.sum(torch.mul(gradB_d, Sdot_d), dim=1)

            # compute lie derivative on uncertainty dataset
            B_dz = B[i1 + i2 + i3 :, 0]
            gradB_dz = gradB[i1 + i2 + i3 :]
            sigma_dz = sigma[i1 + i2 + i3 :, 0]
            Sdot_dz = f_torch(X_dz, U_dz, Z_dz, only_nominal=True)
            Sdotz_dz = f_torch(X_dz, U_dz, Z_dz)
            Bdot_dz = torch.sum(torch.mul(gradB_dz, Sdot_dz), dim=1)
            Bdotz_dz = torch.sum(torch.mul(gradB_dz, Sdotz_dz), dim=1)

            loss, losses, accuracies = self.compute_loss(
                B_i,
                B_u,
                B_d,
                sigma_d,
                Bdot_d,
                B_dz,
                sigma_dz,
                Bdot_dz,
                Bdotz_dz,
                alpha=1.0,
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
                logging.debug(f"Epoch {t}")
                logging.debug(f"accuracy={accuracies}")
                logging.debug(f"losses={losses}")
                logging.debug(f"infos={infos}")
                logging.debug("")

            # early stopping after 2 consecutive epochs with ~100% accuracy
            condition = all(acc >= 99.9 for name, acc in accuracies.items())
            if condition and condition_old:
                break
            condition_old = condition

            loss.backward()
            optimizer.step()

        logging.info(f"Epoch {t}: loss={loss}")
        logging.info(f"mean compensation: {sigma.mean().item()}")
        logging.info(f"losses={losses}")
        logging.info(f"accuracy={accuracies}")

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
        sigma_d: torch.Tensor,
        Bdot_d: torch.Tensor,
        B_dz: torch.Tensor,
        sigma_dz: torch.Tensor,
        Bdot_dz: torch.Tensor,
        Bdotz_dz: torch.Tensor,
        alpha: torch.Tensor | float,
    ) -> tuple[torch.Tensor, dict, dict]:
        """Computes loss function for CBF and its accuracy w.r.t. the batch of data.

        Args:
            B_i (torch.Tensor): Barrier values for initial set
            B_u (torch.Tensor): Barrier values for unsafe set
            B_d (torch.Tensor): Barrier values for domain
            sigma_d (torch.Tensor): Compensator values for domain
            Bdot_d (torch.Tensor): Barrier derivative values for domain according to nominal model
            B_dz (torch.Tensor): Barrier values for domain according to uncertain model
            sigma_dz (torch.Tensor): Compensator values for domain according to uncertain model
            Bdot_dz (torch.Tensor): Barrier derivative values for domain according to uncertain model
            Bdotz_d (torch.Tensor): Barrier derivative values for domain according to uncertain model
            alpha (torch.Tensor): coeff. linear class-k function, f(x) = alpha * x, for alpha in R_+

        Returns:
            tuple[torch.Tensor, float]: loss and accuracy
        """
        # todo make this private
        assert (
            Bdot_d is None or B_d.shape == Bdot_d.shape
        ), f"B_d and Bdot_d must have the same shape, got {B_d.shape} and {Bdot_d.shape}"
        assert (
            Bdot_dz is None or B_dz.shape == Bdot_dz.shape
        ), f"B_d and Bdot_dz must have the same shape, got {B_d.shape} and {Bdot_dz.shape}"
        assert (
            Bdotz_dz is None or B_dz.shape == Bdotz_dz.shape
        ), f"B_d and Bdotz_dz must have the same shape, got {B_d.shape} and {Bdotz_dz.shape}"
        assert isinstance(
            self.loss_margins, dict
        ), f"Expected loss margins as dict, got {type(self.loss_margins)}"
        assert isinstance(
            self.loss_weights, dict
        ), f"Expected loss weights as dict, got {type(self.loss_weights)}"

        margin_init = self.loss_margins["init"]
        margin_unsafe = self.loss_margins["unsafe"]
        margin_lie = self.loss_margins["lie"]
        margin_robust = self.loss_margins["robust"]

        weight_init = self.loss_weights["init"]
        weight_unsafe = self.loss_weights["unsafe"]
        weight_lie = self.loss_weights["lie"]
        weight_robust = self.loss_weights["robust"]

        accuracy_i = (B_i >= margin_init).count_nonzero().item()
        accuracy_u = (B_u < -margin_unsafe).count_nonzero().item()
        accuracy_d = (
            (Bdot_d - sigma_d + alpha * B_d >= margin_lie).count_nonzero().item()
        )

        accuracy_z = (
            torch.logical_or(
                Bdot_dz - sigma_dz + alpha * B_dz < -margin_robust,
                Bdotz_dz + alpha * B_dz >= margin_robust,
            )
            .count_nonzero()
            .item()
        )

        percent_accuracy_init = 100 * accuracy_i / B_i.shape[0]
        percent_accuracy_unsafe = 100 * accuracy_u / B_u.shape[0]
        percent_accuracy_lie = 100 * accuracy_d / Bdot_d.shape[0]
        percent_accuracy_robust = 100 * accuracy_z / Bdot_dz.shape[0]

        # penalize B_i < 0
        init_loss = weight_init * (self.loss_relu(margin_init - B_i)).mean()
        # penalize B_u > 0
        unsafe_loss = weight_unsafe * (self.loss_relu(B_u + margin_unsafe)).mean()
        # penalize dB_d - sigma_d + alpha * B_d < 0
        lie_loss = (
            weight_lie
            * (self.loss_relu(margin_lie - (Bdot_d - sigma_d + alpha * B_d))).mean()
        )
        # penalize dB_d - sigma_d + alpha * B_d >=0 and Bdotz_d + alpha * B_d < 0
        robust_loss = (
            weight_robust
            * (
                self.loss_relu(
                    torch.min(
                        (Bdot_dz - sigma_dz + alpha * B_dz) - margin_robust,
                        margin_robust - (Bdotz_dz + alpha * B_dz),
                    )
                )
            ).mean()
        )

        loss = init_loss + unsafe_loss + lie_loss + robust_loss

        losses = {
            "init_loss": init_loss.item(),
            "unsafe_loss": unsafe_loss.item(),
            "lie_loss": lie_loss.item(),
            "robust_loss": robust_loss.item(),
            "tot_loss": loss.item(),
        }

        accuracy = {
            "accuracy_init": percent_accuracy_init,
            "accuracy_unsafe": percent_accuracy_unsafe,
            "accuracy_derivative": percent_accuracy_lie,
            "accuracy_robust": percent_accuracy_robust,
        }

        # debug
        logging.debug("Dataset Accuracy:")
        logging.debug("\n".join([f"{k}:{v}" for k, v in accuracy.items()]))

        return loss, losses, accuracy