import datetime
import logging
import pathlib
from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Type

import matplotlib.pyplot as plt
import numpy as np
import torch

from fosco.certificates import make_certificate
from fosco.common.formatter import CustomFormatter
from fosco.common.plotting import benchmark_3d, benchmark_lie
from fosco.consolidator import make_consolidator
from fosco.common.consts import (
    CertificateType,
    TimeDomain,
    ActivationType,
    VerifierType,
)
from fosco.learner import make_learner, LearnerNN
from fosco.translator import make_translator
from fosco.verifier import make_verifier
from systems import ControlAffineControllableDynamicalModel
from systems.system import UncertainControlAffineControllableDynamicalModel

CegisResult = namedtuple("CegisResult", ["found", "net", "infos"])

DEBUG_PLOT = False


@dataclass
class CegisConfig:
    # system
    SYSTEM: Type[ControlAffineControllableDynamicalModel] = None
    DOMAINS: dict[str, Any] = None
    TIME_DOMAIN: TimeDomain = TimeDomain.CONTINUOUS
    # fosco
    CERTIFICATE: CertificateType = CertificateType.CBF
    VERIFIER: VerifierType = VerifierType.Z3
    CEGIS_MAX_ITERS: int = 10
    ROUNDING: int = 3
    # training
    DATA_GEN: dict[str, callable] = None
    N_DATA: int = 500
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    # net architecture
    N_HIDDEN_NEURONS: tuple[int, ...] = (10,)
    ACTIVATION: tuple[ActivationType, ...] = (ActivationType.SQUARE,)
    # seeding
    SEED: int = None

    def __getitem__(self, item):
        return getattr(self, item)




class Cegis:
    def __init__(self, config: CegisConfig, verbose: int = 0):
        self.config = config

        # seeding
        if self.config.SEED is None:
            self.config.SEED = torch.randint(0, 1000000, (1,)).item()
        torch.manual_seed(self.config.SEED)
        np.random.seed(self.config.SEED)

        # logging
        # todo implement logger to log to file and use logging for info/debug messages
        self.logger = self._initialise_logger(verbose=verbose)

        # intialization
        self.f = self.config.SYSTEM()
        self.x, self.x_map, self.domains = self._initialise_domains()
        self.xdot, self.xdotz = self._initialise_dynamics()
        self.datasets = self._initialise_data()

        # todo pass logger to components
        self.certificate = self._initialise_certificate()
        self.learner = self._initialise_learner()
        self.verifier = self._initialise_verifier()
        self.consolidator = self._initialise_consolidator()
        self.translator = self._initialise_translator()

        self._result = None

        self._assert_state()

    def _initialise_logger(self, verbose: int) -> logging.Logger:
        logger = logging.getLogger("CEGIS")
        ch = logging.StreamHandler()

        verbose = min(max(verbose, 0), 2)
        verbosity_levels = [logging.WARNING, logging.INFO, logging.DEBUG]

        logger.setLevel(verbosity_levels[verbose])
        ch.setLevel(verbosity_levels[verbose])

        ch.setFormatter(CustomFormatter())
        logger.addHandler(ch)

        return logger

    def _initialise_learner(self) -> LearnerNN:
        learner_type = make_learner(system=self.f, time_domain=self.config.TIME_DOMAIN)
        learner_instance = learner_type(
            state_size=self.f.n_vars,
            learn_method=self.certificate.learn,
            hidden_sizes=self.config.N_HIDDEN_NEURONS,
            activation=self.config.ACTIVATION,
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
        )
        return learner_instance

    def _initialise_verifier(self):
        verifier_type = make_verifier(self.config.VERIFIER)
        verifier_instance = verifier_type(
            solver_vars=self.x, constraints_method=self.certificate.get_constraints
        )
        return verifier_instance

    def _initialise_domains(self):
        # todo: create domains based on input instead of hardcoding x, u
        verifier_type = make_verifier(type=self.config.VERIFIER)
        x = verifier_type.new_vars(self.f.n_vars, base="x")
        u = verifier_type.new_vars(self.f.n_controls, base="u")

        if isinstance(self.f, UncertainControlAffineControllableDynamicalModel):
            z = verifier_type.new_vars(self.f.n_uncertain, base="z")
        else:
            z = None

        # create map id -> variable
        if z:
            x_map = {"v": x, "u": u, "z": z}
            x = x + u + z
        else:
            x_map = {"v": x, "u": u}
            x = x + u

        # create domains
        domains = {
            label: domain.generate_domain(x)
            for label, domain in self.config.DOMAINS.items()
        }

        self.logger.debug("Domains: {}".format(domains))
        return x, x_map, domains

    def _initialise_dynamics(self):
        if isinstance(self.f, UncertainControlAffineControllableDynamicalModel):
            xdot = self.f(**self.x_map, only_nominal=True)
            xdotz = self.f(**self.x_map)
        else:
            xdot = self.f(**self.x_map)
            xdotz = None

        return xdot, xdotz

    def _initialise_data(self):
        datasets = {}
        for label in self.config.DATA_GEN.keys():
            datasets[label] = self.config.DATA_GEN[label](self.config.N_DATA)
        return datasets

    def _initialise_certificate(self):
        certificate_type = make_certificate(certificate_type=self.config.CERTIFICATE)
        return certificate_type(vars=self.x_map, domains=self.config.DOMAINS)

    def _initialise_consolidator(self):
        return make_consolidator()

    def _initialise_translator(self):
        return make_translator(
            verifier_type=self.config.VERIFIER,
            time_domain=self.config.TIME_DOMAIN,
            rounding=self.config.ROUNDING,
        )

    def solve(self) -> CegisResult:
        logdir = f"logs/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

        state = self.init_state()

        iter = None

        tot_loss = []
        losses = {}
        accuracy = {}
        for iter in range(1, self.config.CEGIS_MAX_ITERS + 1):
            self.logger.debug(f"Iteration {iter}")

            # debug print
            if DEBUG_PLOT:
                pathlib.Path(logdir).mkdir(parents=True, exist_ok=True)

                plt.clf()
                domains = self.config.DOMAINS
                xrange = domains["lie"].lower_bounds[0], domains["lie"].upper_bounds[0]
                yrange = domains["lie"].lower_bounds[1], domains["lie"].upper_bounds[1]

                func = self.learner.net
                ax2 = benchmark_3d(
                    func,
                    domains,
                    [0.0],
                    xrange,
                    yrange,
                    title=f"CBF - Iter {iter}",
                )
                # ax2.view_init(azim=0, elev=90)
                plt.savefig(f"{logdir}/cbf_iter_{iter}.png")

                plt.clf()
                if isinstance(self.f, UncertainControlAffineControllableDynamicalModel):
                    func = lambda x: self.learner.xsigma(x)
                    ax2 = benchmark_3d(
                        func,
                        domains,
                        [0.0],
                        xrange,
                        yrange,
                        title=f"Compensator - Iter {iter}",
                    )
                    # ax2.view_init(azim=0, elev=90)
                    plt.savefig(f"{logdir}/sigma_iter_{iter}.png")

                    # plot losses
                    if len(losses) > 0:
                        plt.clf()
                        fig, axes = plt.subplots(1, len(losses) + 1, figsize=(5 * (len(losses) + 1), 5))

                        for ax, (key, value) in zip(axes, losses.items()):
                            ax.plot(value)
                            ax.set_title(key)
                            ax.set_xlabel("iteration")
                            ax.set_ylabel("loss")

                        ax = axes[-1]
                        ax.plot(tot_loss)
                        ax.set_title("total loss")
                        ax.set_xlabel("iteration")
                        ax.set_ylabel("loss")

                        plt.savefig(f"{logdir}/losses_iter_{iter}.png")


            # Learner component
            self.logger.debug("Learner")
            outputs = self.learner.update(**state)
            #state.update(outputs)
            # add losses and accuracy in state to losses
            if "losses" in outputs:
                for key, value in outputs["losses"].items():
                    if key not in losses:
                        losses[key] = []
                    losses[key].append(value)

            if "accuracy" in outputs:
                for key, value in outputs["accuracy"].items():
                    if key not in accuracy:
                        accuracy[key] = []
                    accuracy[key].append(value)

            if "loss" in outputs:
                tot_loss.append(outputs["loss"])

            # Translator component
            self.logger.debug("Translator")
            outputs = self.translator.translate(**state)
            state.update(outputs)

            # Verifier component
            self.logger.debug("Verifier")
            outputs = self.verifier.verify(**state)
            state.update(outputs)

            # Consolidator component
            self.logger.debug("Consolidator")
            outputs = self.consolidator.get(**state)
            state.update(outputs)

            if state["found"]:
                self.logger.debug("found valid certificate")
                break

        # state = self.process_timers(state)

        logging.info(f"CEGIS finished after {iter} iterations")




        infos = {"iter": iter}
        self._result = CegisResult(
            found=state["found"],
            net=state["V_net"],
            infos=infos
        )

        return self._result

    def init_state(self) -> dict:
        # todo: extend to multiplicative uncertainty too
        xsigma = self.learner.xsigma if hasattr(self.learner, "xsigma") else None

        state = {
            "found": False,     # whether a valid cbf was found
            "iter": 0,          # current iteration
            "system": self.f,   # system object

            "V_net": self.learner.net,  # cbf model as nn
            "sigma_net": xsigma,  # sigma model as nn

            "xdot_func": self.f._f_torch,   # numerical dynamics function
            "datasets": self.datasets,  # dictionary of datasets of training data

            "x_v_map": self.x_map,  # dictionary of symbolic variables
            "V_symbolic": None,     # symbolic expression of cbf
            "sigma_symbolic": None, # symbolic expression of compensator sigma
            "Vdot_symbolic": None,  # symbolic expression of lie derivative w.r.t. nominal dynamics
            "Vdotz_symbolic": None, # symbolic expression of lie derivative w.r.t. uncertain dynamics
            "xdot": self.xdot,      # symbolic expression of nominal dynamics
            "xdotz": self.xdotz,    # symbolic expression of uncertain dynamics

            "cex": None,    # counterexamples
            # CegisStateKeys.found: False,
            # CegisStateKeys.verification_timed_out: False,
            # CegisStateKeys.cex: None,
            # CegisStateKeys.trajectory: None,
            # CegisStateKeys.ENet: self.config.ENET,
        }

        return state

    @property
    def result(self):
        return self._result

    def _assert_state(self):
        assert self.config.LEARNING_RATE > 0
        assert self.config.CEGIS_MAX_ITERS > 0
        assert (
            self.x is self.verifier.xs
        ), "expected same variables in fosco and verifier"
        self.certificate._assert_state(self.domains, self.datasets)
