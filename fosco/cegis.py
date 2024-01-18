import logging
import numpy as np
import torch

from fosco.certificates import make_certificate
from fosco.common.domains import Rectangle
from fosco.config import CegisConfig, CegisResult
from fosco.consolidator import make_consolidator
from fosco.common.consts import DomainNames
from fosco.learner import make_learner, LearnerNN
from fosco.plotting.utils import plot_func_and_domains
from fosco.translator import make_translator
from fosco.verifier import make_verifier
from logger import make_logger, Logger, LOGGING_LEVELS
from systems.system import UncertainControlAffineControllableDynamicalModel
from systems.utils import lie_derivative_fn


class Cegis:
    def __init__(self, config: CegisConfig, verbose: int = 0):
        self.config = config

        # seeding
        if self.config.SEED is None:
            self.config.SEED = torch.randint(0, 1000000, (1,)).item()
        torch.manual_seed(self.config.SEED)
        np.random.seed(self.config.SEED)

        # logging
        self.verbose = min(max(verbose, 0), len(LOGGING_LEVELS) - 1)
        self.logger, self.tlogger = self._initialise_logger()

        # intialization
        self.f = self.config.SYSTEM()
        self.x, self.x_map, self.domains = self._initialise_domains()
        self.xdot, self.xdotz = self._initialise_dynamics()
        self.datasets = self._initialise_data()

        self.certificate = self._initialise_certificate()
        self.learner = self._initialise_learner()
        self.verifier = self._initialise_verifier()
        self.consolidator = self._initialise_consolidator()
        self.translator = self._initialise_translator()

        self._result = None

        self._assert_state()

    def _initialise_logger(self) -> Logger:
        config = self.config.dict()
        logger = make_logger(logger_type=self.config.LOGGER, config=config)

        logging.basicConfig()
        tlogger = logging.getLogger(__name__)
        tlogger.setLevel(LOGGING_LEVELS[self.verbose])

        return logger, tlogger

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

        self.tlogger.debug("\n".join(["Domains"] + [f"{k}: {v}" for k, v in domains.items()]) + "\n")

        return x, x_map, domains

    def _initialise_dynamics(self):
        if isinstance(self.f, UncertainControlAffineControllableDynamicalModel):
            xdot = self.f(**self.x_map, only_nominal=True)
            xdotz = self.f(**self.x_map)
        else:
            xdot = self.f(**self.x_map)
            xdotz = None

        self.tlogger.debug(f"Nominal Dynamics: {'initialized' if xdot else 'not initialized'}")
        self.tlogger.debug(f"Uncertain Dynamics: {'initialized' if xdotz else 'not initialized'}")

        return xdot, xdotz

    def _initialise_data(self):
        datasets = {}
        for label in self.config.DATA_GEN.keys():
            datasets[label] = self.config.DATA_GEN[label](self.config.N_DATA)

        self.tlogger.debug("\n".join(["Data Collection"] + [f"{k}: {v.shape}" for k, v in datasets.items()]) + "\n")

        return datasets

    def _initialise_certificate(self):
        certificate_type = make_certificate(certificate_type=self.config.CERTIFICATE)
        return certificate_type(vars=self.x_map, domains=self.config.DOMAINS, verbose=self.verbose, config=self.config)

    def _initialise_consolidator(self):
        return make_consolidator(verbose=self.verbose)

    def _initialise_translator(self):
        return make_translator(
            certificate_type=self.config.CERTIFICATE,
            verifier_type=self.config.VERIFIER,
            time_domain=self.config.TIME_DOMAIN,
            rounding=self.config.ROUNDING,
            verbose=self.verbose,
        )

    def solve(self) -> CegisResult:
        state = self.init_state()

        iter = None

        for iter in range(1, self.config.CEGIS_MAX_ITERS + 1):
            self.tlogger.info(f"Iteration {iter}")

            # logging learned functions
            in_domain = self.config.DOMAINS[DomainNames.XD.value]
            other_domains = {k: v for k, v in self.config.DOMAINS.items() if
                             k in [DomainNames.XI.value, DomainNames.XU.value]}
            fig = plot_func_and_domains(
                func=self.learner.net,
                in_domain=in_domain,
                levels=[0.0],
                domains=other_domains,
                dim_select=(0, 1),
            )
            self.logger.log_image(tag="barrier", image=fig, step=iter)

            for dim in range(self.f.n_vars):
                func = lambda x: self.learner.net.compute_net_gradnet(x)[1][:, dim]
                fig = plot_func_and_domains(
                    func=func,
                    in_domain=in_domain,
                    levels=[0.0],
                    domains=other_domains,
                    dim_select=(0, 1),
                )
                self.logger.log_image(tag=f"barrier_grad", image=fig, step=iter, context={"dimension": dim})

            u_domain = self.config.DOMAINS[DomainNames.UD.value]
            assert isinstance(u_domain, Rectangle), "only rectangular domains are supported for u"
            lb, ub = np.array(u_domain.lower_bounds), np.array(u_domain.upper_bounds)
            for u_norm in np.linspace(-1, 1, 5):
                # denormalize u to the domain
                u = (lb + ub) / 2.0 + u_norm * (ub - lb) / 2.0
                ctrl = lambda x: torch.ones((x.shape[0], self.f.n_controls)) * torch.tensor(u).float()
                if isinstance(self.f, UncertainControlAffineControllableDynamicalModel):
                    f = lambda x, u: self.f._f_torch(x, u, z=torch.zeros((x.shape[0], self.f.n_uncertain)))
                else:
                    f = lambda x, u: self.f._f_torch(x, u)

                # lie derivative
                func = lambda x: lie_derivative_fn(certificate=self.learner.net, f=f, ctrl=ctrl)(x)
                fig = plot_func_and_domains(
                    func=func,
                    in_domain=in_domain,
                    levels=[0.0],
                    domains=other_domains,
                    dim_select=(0, 1),
                )
                self.logger.log_image(tag=f"lie_derivative", image=fig, step=iter, context={"u_norm": str(u)})

                # cbf condition
                alpha = lambda x: 1.0 * x
                if isinstance(self.f, UncertainControlAffineControllableDynamicalModel):
                    func = lambda x: lie_derivative_fn(certificate=self.learner.net, f=f, ctrl=ctrl)(
                        x) - self.learner.xsigma(x) + alpha(self.learner.net(x))
                else:
                    func = lambda x: lie_derivative_fn(certificate=self.learner.net, f=f, ctrl=ctrl)(x) + alpha(
                        self.learner.net(x))
                fig = plot_func_and_domains(
                    func=func,
                    in_domain=in_domain,
                    levels=[0.0],
                    domains=other_domains,
                    dim_select=(0, 1),
                )
                self.logger.log_image(tag=f"cbf_condition", image=fig, step=iter, context={"u_norm": str(u)})

            if isinstance(self.f, UncertainControlAffineControllableDynamicalModel):
                fig = plot_func_and_domains(
                    func=self.learner.xsigma,
                    in_domain=in_domain,
                    levels=[0.0],
                    domains=other_domains,
                    dim_select=(0, 1),
                )
                self.logger.log_image(tag="compensator", image=fig, step=iter)

            # Learner component
            self.tlogger.debug("Learner")
            outputs = self.learner.update(**state)
            for context, dict_metrics in outputs.items():
                self.logger.log_scalar(tag=None, value=dict_metrics, step=iter, context={context: True})

            # Translator component
            self.tlogger.debug("Translator")
            outputs = self.translator.translate(**state)
            state.update(outputs)

            # Verifier component
            self.tlogger.debug("Verifier")
            outputs = self.verifier.verify(**state)
            state.update(outputs)

            # Consolidator component
            self.tlogger.debug("Consolidator")
            outputs = self.consolidator.get(**state)
            state.update(outputs)

            if state["found"]:
                self.tlogger.debug("found valid certificate")
                break

        # state = self.process_timers(state)
        self.tlogger.info(f"CEGIS finished after {iter} iterations")

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
            "found": False,  # whether a valid cbf was found
            "iter": 0,  # current iteration
            "system": self.f,  # system object

            "V_net": self.learner.net,  # cbf model as nn
            "sigma_net": xsigma,  # sigma model as nn

            "xdot_func": self.f._f_torch,  # numerical dynamics function
            "datasets": self.datasets,  # dictionary of datasets of training data

            "x_v_map": self.x_map,  # dictionary of symbolic variables
            "V_symbolic": None,  # symbolic expression of cbf
            "sigma_symbolic": None,  # symbolic expression of compensator sigma
            "Vdot_symbolic": None,  # symbolic expression of lie derivative w.r.t. nominal dynamics
            "Vdotz_symbolic": None,  # symbolic expression of lie derivative w.r.t. uncertain dynamics
            "xdot": self.xdot,  # symbolic expression of nominal dynamics
            "xdotz": self.xdotz,  # symbolic expression of uncertain dynamics

            "cex": None,  # counterexamples
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
