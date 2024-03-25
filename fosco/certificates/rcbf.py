import logging
import math
from typing import Generator

import numpy as np
import torch
from torch.optim import Optimizer

from fosco.certificates.cbf import ControlBarrierFunction, TrainableCBF
from fosco.config import CegisConfig
from fosco.common.domains import Set, Rectangle
from fosco.common.consts import DomainName
from fosco.common.utils import _set_assertion
from fosco.learner.learner_rcbf_ct import LearnerRobustCT
from fosco.verifier.verifier import SYMBOL
from fosco.systems import ControlAffineDynamics

XD = DomainName.XD.value
XI = DomainName.XI.value
XU = DomainName.XU.value
UD = DomainName.UD.value
ZD = DomainName.ZD.value


class RobustControlBarrierFunction(ControlBarrierFunction):
    """
    Certifies Safety for continuous time controlled systems with control affine dynamics.

    Parameters
    ----------
    system : ControlAffineDynamics
        The control affine dynamics of the system.
    variables : dict[str, list[SYMBOL]]
        The symbolic variables for the system.
    domains : dict[str, Set]
        The domains for the symbolic variables.
    verbose : int, optional
        The verbosity level, by default 0.

    Raises
    ------
    AssertionError
        If the arguments are not of the expected type or if the domains are not valid.
    """

    def __init__(
            self,
            system: ControlAffineDynamics,
            variables: dict[str, list[SYMBOL]],
            domains: dict[str, Set],
            verbose: int = 0,
    ) -> None:
        super().__init__(
            system=system, variables=variables, domains=domains, verbose=verbose
        )

        self.z_vars = variables["z"]
        self.z_domain: SYMBOL = domains[ZD].generate_domain(self.z_vars)
        self.n_uncertain = len(self.z_vars)

    def _assert_state(self) -> None:
        dn = DomainName
        domain_labels = set(self.domains.keys())

        _set_assertion(
            {dn.XD.value, dn.UD.value, dn.XI.value, dn.XU.value, dn.ZD.value},
            domain_labels,
            "Symbolic Domains",
        )

        assert all(
            [sv in self.variables for sv in ["v", "u", "z"]]
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
            Bdot_residual,
            Bdot_residual_constr,
            Bdot_residual_vars,
    ) -> Generator:
        """
        Returns the constraints for the CBF problem.

        Args:
            TODO

        Returns:
            generator: yields constraints for each domain
        """
        assert isinstance(B_vars, list) and all([isinstance(v, SYMBOL) for v in B_vars]), f"Expected list of SYMBOL, got {B_vars}"
        assert isinstance(sigma_vars, list) and all([isinstance(v, SYMBOL) for v in sigma_vars]), f"Expected list of SYMBOL, got {sigma_vars}"
        assert isinstance(Bdot_vars, list) and all([isinstance(v, SYMBOL) for v in Bdot_vars]), f"Expected list of SYMBOL, got {Bdot_vars}"
        assert isinstance(Bdot_residual_vars, list) and all([isinstance(v, SYMBOL) for v in Bdot_residual_vars]), f"Expected list of SYMBOL, got {Bdot_residual_vars}"

        # initial condition
        # Bx >= 0 if x \in initial
        # counterexample: B < 0 and x \in initial
        initial_vars = self.x_vars
        initial_aux_vars = [v for v in B_vars if v not in initial_vars]
        initial_constr = self._init_constraint_smt(
            verifier=verifier, B=B, B_constr=B_constr
        )

        # unsafe condition
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
        #
        # note: smart trick for tractable verification using vertices of input convex-hull
        # counterexample: x \in domain and AND_v (u=v and Bdot + alpha * Bx < 0)
        alpha = lambda x: x
        feasible_vars = self.x_vars + self.u_vars
        feasible_aux_vars = [v for v in B_vars + sigma_vars + Bdot_vars if v not in feasible_vars]
        feasibility_constr = self._feasibility_constraint_smt(
            verifier=verifier,
            B=B,
            B_constr=B_constr,
            sigma=sigma,
            sigma_constr=sigma_constr,
            Bdot=Bdot,
            Bdot_constr=Bdot_constr,
            alpha=alpha,
        )

        # robustness constraint
        robust_vars = self.x_vars + self.u_vars + self.z_vars
        robust_aux_vars = [v for v in B_vars + sigma_vars + Bdot_vars + Bdot_residual_vars if v not in robust_vars]
        robust_constr = self._robust_constraint_smt(
            verifier=verifier,
            B=B,
            B_constr=B_constr,
            sigma=sigma,
            sigma_constr=sigma_constr,
            Bdot=Bdot,
            Bdot_constr=Bdot_constr,
            Bdot_residual=Bdot_residual,
            Bdot_residual_constr=Bdot_residual_constr,
            alpha=alpha,
        )

        logging.debug(f"inital_constr: {initial_constr}")
        logging.debug(f"unsafe_constr: {unsafe_constr}")
        logging.debug(f"lie_constr: {feasibility_constr}")
        logging.debug(f"robust_constr: {robust_constr}")

        for cs in (
                # first check initial and unsafe conditions
                {XI: (initial_constr, initial_vars, initial_aux_vars),
                 XU: (unsafe_constr, unsafe_vars, unsafe_aux_vars)},
                # then check robustness to uncertainty
                {ZD: (robust_constr, robust_vars, robust_aux_vars)},
                # finally check feasibility
                {XD: (feasibility_constr, feasible_vars, feasible_aux_vars)},
        ):
            yield cs

    def _feasibility_constraint_smt(
            self, verifier, B, B_constr, sigma, sigma_constr, Bdot, Bdot_constr, alpha
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
            # this is different from vanilla cbf because of the compensator sigma
            vertex_constr = Bdot - sigma + alpha(B) < 0
            for c in Bdot_constr + sigma_constr + B_constr:
                vertex_constr = _And(vertex_constr, c)

            for u_var, u_val in zip(self.u_vars, u_vert):
                vertex_constr = _Substitute(vertex_constr, (u_var, _RealVal(u_val)))
            lie_constr = _And(lie_constr, vertex_constr)

        return lie_constr

    def _robust_constraint_smt(
            self,
            verifier,
            B,
            B_constr,
            sigma,
            sigma_constr,
            Bdot,
            Bdot_constr,
            Bdot_residual,
            Bdot_residual_constr,
            alpha,
    ) -> SYMBOL:
        """
        Robustness constraint

        spec := forall x in belt(B==0) forall u forall z  B(x)>0 -> (sigma(x, u, z) >= - (Bdotz - Bdot))
        counterexample: x \in xdomain and z \in zdomain and u \in udomain and
                        B(x) > 0 and B(x) < 0.5 and sigma(x,u,z) < - (Bdotz - Bdot)
        """
        _And = verifier.solver_fncts()["And"]

        # precondition: we are in the belt of the barrier (zero-level set)
        belt_constr = _And(B >= 0, B <= 0.25)  # B ~ 0
        for c in B_constr:
            belt_constr = _And(belt_constr, c)

        # precondition: the input satisfies the feasibility constraint
        feas_constr = Bdot - sigma + alpha(B) >= 0
        for c in B_constr + Bdot_constr:
            feas_constr = _And(feas_constr, c)

        pre_constr = _And(belt_constr, feas_constr)

        # sigma is not compensating enough, sigma < - Bdot_residual
        robust_constr = _And(pre_constr, sigma < -Bdot_residual)
        for c in Bdot_residual_constr + sigma_constr:
            robust_constr = _And(robust_constr, c)

        # add domain constraints
        robust_constr = _And(robust_constr, self.x_domain)
        robust_constr = _And(robust_constr, self.z_domain)
        robust_constr = _And(robust_constr, self.u_domain)

        return robust_constr


class TrainableRCBF(TrainableCBF, RobustControlBarrierFunction):
    """
    Trainable robust CBF (RCBF) for continuous time controlled systems with control affine dynamics.

    Parameters
    ----------
    config : CegisConfig
        The configuration for the CEGIS algorithm.
    kwargs : dict
        Other parameters as in TrainableCBF.

    Raises
    ------
    AssertionError
        If the arguments are not of the expected type or if the domains are not valid.
    """

    def __init__(
            self,
            system: ControlAffineDynamics,
            variables: dict[str, list],
            domains: dict[str, Set],
            config: CegisConfig,
            verbose: int = 0,
    ) -> None:
        super(TrainableRCBF, self).__init__(
            system=system, variables=variables, domains=domains, config=config, verbose=verbose
        )

        # add extra loss margin for uncertainty loss
        self.loss_keys = self.loss_keys + ["robust", "conservative_sigma"]
        if isinstance(config.LOSS_MARGINS, float):
            for loss_k in ["robust", "conservative_sigma"]:
                self.loss_margins[loss_k] = config.LOSS_MARGINS
        else:
            assert "robust" in config.LOSS_MARGINS, f"Missing loss margin 'robust', got {config.LOSS_MARGINS}"
            assert "conservative_sigma" in config.LOSS_MARGINS, f"Missing loss margin 'conservative_sigma', got {config.LOSS_MARGINS}"
            self.loss_margins["robust"] = config.LOSS_MARGINS["robust"]
            self.loss_margins["conservative_sigma"] = config.LOSS_MARGINS["conservative_sigma"]

        # add extra loss weight for uncertainty loss
        if isinstance(config.LOSS_WEIGHTS, float):
            for loss in ["robust", "conservative_sigma"]:
                self.loss_weights[loss] = config.LOSS_WEIGHTS
        else:
            for loss in ["robust", "conservative_sigma"]:
                assert loss in config.LOSS_WEIGHTS, f"Missing loss weight {loss}, got {config.LOSS_WEIGHTS}"
                self.loss_weights[loss] = config.LOSS_WEIGHTS[loss]

        # rerun assertion with extended loss terms
        self._assert_state()

    def learn(
            self,
            learner: LearnerRobustCT,
            optimizers: dict[str, Optimizer],
            datasets: dict,
            f_torch: callable,
    ) -> dict[str, float | np.ndarray | dict]:
        """
        Updates the CBF model.

        :param learner: LearnerNN object
        :param optimizer: dict of optimizers
        :param datasets: dictionary of (string,torch.Tensor) pairs
        :param f_torch: callable
        """

        if not optimizers:
            return {}
        assert "barrier" in optimizers, f"Missing optimizer 'barrier', got {optimizers}"
        assert "xsigma" in optimizers, f"Missing optimizer 'xsigma', got {optimizers}"

        condition_old = False
        i1 = datasets[XD].shape[0]
        i2 = datasets[XI].shape[0]
        i3 = datasets[XU].shape[0]

        states_d = torch.cat(
            [datasets[label][:, : self.n_vars] for label in [XD, XI, XU]]
        )
        input_d = datasets[XD][:, self.n_vars: self.n_vars + self.n_controls]


        states_dz = datasets[ZD][:, : self.n_vars]
        input_dz = datasets[ZD][:, self.n_vars: self.n_vars + self.n_controls]
        uncert_dz = datasets[ZD][
               :,
               self.n_vars
               + self.n_controls: self.n_vars
                                  + self.n_controls
                                  + self.n_uncertain,
               ]

        losses, accuracies, infos = {}, {}, {}
        for t in range(self.epochs):
            optimizers["barrier"].zero_grad()

            # compute output for barrier loss
            B = learner.net(states_d)
            gradB = learner.net.gradient(states_d)
            sigma = learner.xsigma(states_d)

            B_d = B[:i1, 0]
            B_i = B[i1: i1 + i2, 0]
            B_u = B[i1 + i2: i1 + i2 + i3, 0]

            assert (
                    B_d.shape[0] == input_d.shape[0]
            ), f"expected pairs of state,input data. Got {B_d.shape[0]} and {input_d.shape[0]}"
            X_d = states_d[:i1]
            gradB_d = gradB[:i1]
            sigma_d = sigma[:i1, 0]
            Sdot_d = f_torch(v=X_d, u=input_d, z=None, only_nominal=True)
            Bdot_d = torch.sum(torch.mul(gradB_d, Sdot_d), dim=1)

            barrier_loss, barrier_losses, barrier_accuracies = self.compute_loss(
                B_i=B_i,
                B_u=B_u,
                B_d=B_d,
                Bdot_d=Bdot_d - sigma_d,
                alpha=1.0,
            )

            barrier_loss.backward()
            optimizers["barrier"].step()

            # compute output for robust loss
            optimizers["xsigma"].zero_grad()

            B_dz = learner.net(states_dz)[:, 0]
            gradB_dz = learner.net.gradient(states_dz)
            sigma_dz = learner.xsigma(states_dz)[:, 0]

            Sdot_dz = f_torch(states_dz, input_dz, uncert_dz, only_nominal=True)
            Sdotz_dz = f_torch(states_dz, input_dz, uncert_dz)
            Bdot_dz = torch.sum(torch.mul(gradB_dz, Sdot_dz), dim=1)
            Bdotz_dz = torch.sum(torch.mul(gradB_dz, Sdotz_dz), dim=1)
            
            sigma_loss, sigma_losses, sigma_accuracies = self.compute_robust_loss(
                B_dz=B_dz,
                Bdotz_dz=Bdotz_dz,
                Bdot_dz=Bdot_dz,
                sigma_dz=sigma_dz,
                alpha=1.0,
            )

            losses = {**barrier_losses, **sigma_losses}
            accuracies = {**barrier_accuracies, **sigma_accuracies}

            # net gradient info
            netgrad_sos = torch.sum(torch.square(gradB))
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

            sigma_loss.backward()
            optimizers["xsigma"].step()

        return {
            "loss": losses,
            "accuracy": accuracies,
            "info": infos,
        }

    def compute_robust_loss(
            self,
            B_dz: torch.Tensor,
            Bdotz_dz: torch.Tensor,
            Bdot_dz: torch.Tensor,
            sigma_dz: torch.Tensor,
            alpha: torch.Tensor | float,
    ) -> tuple[torch.Tensor, dict, dict]:
        assert (
                Bdot_dz is None or B_dz.shape == Bdot_dz.shape
        ), f"B_d and Bdot_dz must have the same shape, got {B_dz.shape} and {Bdot_dz.shape}"
        assert (
                Bdotz_dz is None or B_dz.shape == Bdotz_dz.shape
        ), f"B_d and Bdotz_dz must have the same shape, got {B_dz.shape} and {Bdotz_dz.shape}"

        belt_margin = 0.5
        margin_robust = self.loss_margins["robust"]
        weight_robust = self.loss_weights["robust"]
        weight_conservative_s = self.loss_weights["conservative_sigma"]

        accuracy_z = torch.logical_and(
                B_dz < belt_margin,
                sigma_dz + Bdotz_dz - Bdot_dz >= 0,
            ).count_nonzero().item()

        percent_accuracy_robust = 100 * accuracy_z / Bdot_dz.shape[0]

        # robust loss to make sigma robust to uncertainty
        belt_mask = (B_dz < belt_margin).float()
        compensator_term = belt_mask * (margin_robust - (sigma_dz + Bdotz_dz - Bdot_dz))

        robust_loss = (
                weight_robust * self.loss_relu(compensator_term).mean()
        )

        # regularization losses
        # penalize high sigma (conservative)
        loss_sigma_pos = self.loss_relu(sigma_dz).mean()  # penalize sigma_dz > 0
        loss_sigma_conservative = (
                weight_conservative_s * loss_sigma_pos
        )

        sigma_loss = robust_loss + loss_sigma_conservative

        losses = {
            "robust_loss": robust_loss.item(),
            "conservative_sigma_loss": loss_sigma_conservative.item(),
            "sigma_loss": sigma_loss.item(),
        }

        accuracy = {
            "accuracy_robust": percent_accuracy_robust,
        }

        # debug
        logging.debug("Dataset Accuracy:")
        logging.debug("\n".join([f"{k}:{v}" for k, v in accuracy.items()]))

        return sigma_loss, losses, accuracy

