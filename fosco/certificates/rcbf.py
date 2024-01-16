import logging
import math
from typing import Generator

import torch
from torch.optim import Optimizer

from fosco.certificates import Certificate
from fosco.common.domains import Set, Rectangle
from fosco.common.consts import DomainNames
from fosco.common.utils import _set_assertion
from fosco.learner import LearnerCT
from fosco.verifier import SYMBOL
from logger import LOGGING_LEVELS

XD = DomainNames.XD.value
XI = DomainNames.XI.value
XU = DomainNames.XU.value
UD = DomainNames.UD.value
ZD = DomainNames.ZD.value


class RobustControlBarrierFunction(Certificate):
    """
    Certifies Safety for continuous time controlled systems with control affine dynamics.

    Note: CBF use different conventions.
    B(Xi)>0, B(Xu)<0, Bdot(Xd) > -alpha(B(Xd)) for alpha class-k function

    Arguments:
        vars {dict}: dictionary of symbolic variables
        domains {dict}: dictionary of (string,domain) pairs
    """

    def __init__(self, vars: dict[str, list], domains: dict[str, Set], verbose: int = 0) -> None:
        assert all([sv in vars for sv in ["v", "u", "z"]]), f"Missing symbolic variables, got {vars}"
        self.x_vars = vars["v"]
        self.u_vars = vars["u"]
        self.z_vars = vars["z"]

        self.x_domain: SYMBOL = domains[XD].generate_domain(self.x_vars)
        self.u_set: Set = domains[UD]
        self.u_domain: SYMBOL = domains[UD].generate_domain(self.u_vars)
        self.z_domain: SYMBOL = domains[ZD].generate_domain(self.z_vars)

        self.initial_domain: SYMBOL = domains[XI].generate_domain(self.x_vars)
        self.unsafe_domain: SYMBOL = domains[XU].generate_domain(self.x_vars)

        assert isinstance(
            self.u_set, Rectangle
        ), f"CBF only works with rectangular input domains, got {self.u_set}"
        self.n_vars = len(self.x_vars)
        self.n_controls = len(self.u_vars)
        self.n_uncertain = len(self.z_vars)

        # loss parameters
        # todo: bring it outside
        self.loss_relu = torch.nn.Softplus()  # torch.relu  # torch.nn.Softplus()
        self.margin = 0.0
        self.epochs = 1000

        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(LOGGING_LEVELS[verbose])
        self._logger.debug("RCBF initialized")

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
        margin = self.margin

        accuracy_i = (B_i >= margin).count_nonzero().item()
        accuracy_u = (B_u < -margin).count_nonzero().item()
        accuracy_d = (Bdot_d + alpha * B_d >= margin).count_nonzero().item()

        accuracy_z = torch.logical_or(
            Bdot_dz - sigma_dz + alpha * B_dz < margin,
            Bdotz_dz + alpha * B_dz >= margin
        ).count_nonzero().item()

        percent_accuracy_init = 100 * accuracy_i / B_i.shape[0]
        percent_accuracy_unsafe = 100 * accuracy_u / B_u.shape[0]
        percent_accuracy_belt = 100 * accuracy_d / Bdot_d.shape[0]
        percent_accuracy_robust = 100 * accuracy_z / Bdot_dz.shape[0]

        relu = self.loss_relu
        init_loss = (relu(margin - B_i)).mean()  # penalize B_i < 0
        unsafe_loss = (relu(B_u + margin)).mean()  # penalize B_u > 0
        lie_loss = (relu(margin - (Bdot_d - sigma_d + alpha * B_d))).mean()  # penalize dB_d - sigma_d + alpha * B_d < 0
        robust_loss = (relu(
            torch.min(
                (Bdot_dz - sigma_dz + alpha * B_dz) - margin,
                margin - (Bdotz_dz + alpha * B_dz)
            )
        )).mean()  # penalize dB_d - sigma_d + alpha * B_d >=0 and Bdotz_d + alpha * B_d < 0

        losses = {
            "init loss": init_loss.item(),
            "unsafe loss": unsafe_loss.item(),
            "lie loss": lie_loss.item(),
            "robust loss": robust_loss.item(),
        }
        loss = init_loss + unsafe_loss + lie_loss + robust_loss

        accuracy = {
            "acc init": percent_accuracy_init,
            "acc unsafe": percent_accuracy_unsafe,
            "acc derivative": percent_accuracy_belt,
            "acc robust": percent_accuracy_robust,
        }

        return loss, losses, accuracy

    def learn(
            self,
            learner: LearnerCT,
            optimizer: Optimizer,
            datasets: dict,
            f_torch: callable,
    ) -> dict:
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
        U_d = datasets[XD][:, self.n_vars: self.n_vars + self.n_controls]
        Z_d = datasets[XD][:, self.n_vars + self.n_controls: self.n_vars + self.n_controls + self.n_uncertain]

        X_dz = datasets[ZD][:, : self.n_vars]
        U_dz = datasets[ZD][:, self.n_vars: self.n_vars + self.n_controls]
        Z_dz = datasets[ZD][:, self.n_vars + self.n_controls: self.n_vars + self.n_controls + self.n_uncertain]

        for t in range(self.epochs):
            optimizer["net"].zero_grad()
            optimizer["sigma"].zero_grad()

            # net gradient
            B, gradB = learner.net.compute_net_gradnet(state_samples)
            sigma = learner.xsigma(state_samples)

            B_d = B[:i1, 0]
            B_i = B[i1: i1 + i2, 0]
            B_u = B[i1 + i2: i1 + i2 + i3, 0]

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
            B_dz = B[i1 + i2 + i3:, 0]
            gradB_dz = gradB[i1 + i2 + i3:]
            sigma_dz = sigma[i1 + i2 + i3:, 0]
            Sdot_dz = f_torch(X_dz, U_dz, Z_dz, only_nominal=True)
            Sdotz_dz = f_torch(X_dz, U_dz, Z_dz)
            Bdot_dz = torch.sum(torch.mul(gradB_dz, Sdot_dz), dim=1)
            Bdotz_dz = torch.sum(torch.mul(gradB_dz, Sdotz_dz), dim=1)

            loss, losses, accuracy = self.compute_loss(B_i, B_u, B_d, sigma_d, Bdot_d,
                                                       B_dz, sigma_dz, Bdot_dz, Bdotz_dz, alpha=1.0)

            if t % math.ceil(self.epochs / 10) == 0 or self.epochs - t < 10:
                # log_loss_acc(t, loss, accuracy, learner.verbose)
                logging.debug(f"Epoch {t}")
                logging.debug(f"accuracy={accuracy}")
                logging.debug(f"losses={losses}")
                logging.debug("")


            # early stopping after 2 consecutive epochs with ~100% accuracy
            condition = all(acc >= 99.9 for name, acc in accuracy.items())
            if condition and condition_old:
                break
            condition_old = condition

            loss.backward()
            optimizer["net"].step()
            optimizer["sigma"].step()

        logging.info(f"Epoch {t}: loss={loss}")
        logging.info(f"mean compensation: {sigma.mean().item()}")
        logging.info(f"losses={losses}")
        logging.info(f"accuracy={accuracy}")

        # todo return logging info like accuracy and loss

        return {
            "loss": loss.item(),
            "losses": losses,
            "accuracy": accuracy,
        }

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
                {
                    XI: (inital_constr, self.x_vars),
                    XU: (unsafe_constr, self.x_vars)
                },
                {
                    XD: (lie_constr, self.x_vars + self.u_vars + self.z_vars),
                    ZD: (robust_constr, self.x_vars + self.u_vars + self.z_vars)
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
            {dn.XD.value, dn.XI.value, dn.XU.value, dn.ZD.value}, data_labels, "Data Sets"
        )
