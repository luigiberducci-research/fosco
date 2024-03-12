import logging
import pathlib
from datetime import datetime

import numpy as np
import torch

from barriers import make_barrier
from fosco.certificates import make_certificate
from fosco.common.domains import Rectangle
from fosco.config import CegisConfig, CegisResult
from fosco.consolidator import make_consolidator
from fosco.common.consts import DomainName, CertificateType
from fosco.learner import make_learner, LearnerNN
from fosco.plotting.data import scatter_datasets
from fosco.plotting.utils import (
    plot_func_and_domains,
    lie_derivative_fn,
    cbf_condition_fn,
)
from fosco.translator import make_translator
from fosco.verifier import make_verifier
from fosco.logger import make_logger, Logger, LOGGING_LEVELS
from systems.system import UncertainControlAffineDynamics


class Cegis:
    def __init__(self, config: CegisConfig, verbose: int = 0):
        self.config = config

        # seeding
        if self.config.SEED is None:
            self.config.SEED = torch.randint(0, 1000000, (1,)).item()
        torch.manual_seed(self.config.SEED)
        np.random.seed(self.config.SEED)

        # system intialization
        self.f = self.config.SYSTEM()

        # logging
        self.verbose = min(max(verbose, 0), len(LOGGING_LEVELS) - 1)
        self.logger, self.tlogger = self._initialise_logger()

        # domains, dynamics, data
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
        self.tlogger.info(f"Seed: {self.config.SEED}")

    def _initialise_logger(self) -> Logger:
        # make experiment name
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{self.f.id}_{self.config.CERTIFICATE.value}_{self.config.EXP_NAME}_Seed{self.config.SEED}_{datetime_str}"
        self.config.EXP_NAME = exp_name

        config = self.config.dict()
        logger = make_logger(
            logger_type=self.config.LOGGER, config=config, experiment=exp_name
        )

        logging.basicConfig()
        tlogger = logging.getLogger(__name__)
        tlogger.setLevel(LOGGING_LEVELS[self.verbose])

        return logger, tlogger

    def _initialise_learner(self) -> LearnerNN:
        learner_type = make_learner(system=self.f, time_domain=self.config.TIME_DOMAIN)

        initial_models = {}
        if self.config.USE_INIT_MODELS:
            known_fns = make_barrier(system=self.f)
            initial_models["net"] = known_fns["barrier"]
            if self.config.CERTIFICATE == CertificateType.RCBF:
                initial_models["xsigma"] = known_fns["compensator"]

        learner_instance = learner_type(
            state_size=self.f.n_vars,
            learn_method=self.certificate.learn,
            hidden_sizes=self.config.N_HIDDEN_NEURONS,
            activation=self.config.ACTIVATION,
            optimizer=self.config.OPTIMIZER,
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
            initial_models=initial_models,
            verbose=self.verbose
        )
        return learner_instance

    def _initialise_verifier(self):
        verifier_type = make_verifier(self.config.VERIFIER)
        verifier_instance = verifier_type(
            solver_vars=self.x,
            constraints_method=self.certificate.get_constraints,
            rounding=self.config.ROUNDING,
            solver_timeout=self.config.VERIFIER_TIMEOUT,
            n_counterexamples=self.config.VERIFIER_N_CEX,
            verbose=self.verbose,
        )
        return verifier_instance

    def _initialise_domains(self):
        verifier_type = make_verifier(type=self.config.VERIFIER)
        x = verifier_type.new_vars(var_names=[f"x{i}" for i in range(self.f.n_vars)])
        u = verifier_type.new_vars(
            var_names=[f"u{i}" for i in range(self.f.n_controls)]
        )

        if isinstance(self.f, UncertainControlAffineDynamics):
            z = verifier_type.new_vars(
                var_names=[f"z{i}" for i in range(self.f.n_uncertain)]
            )
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

        self.tlogger.debug(
            "\n".join(["Domains"] + [f"{k}: {v}" for k, v in domains.items()]) + "\n"
        )

        return x, x_map, domains

    def _initialise_dynamics(self):
        if isinstance(self.f, UncertainControlAffineDynamics):
            xdot = self.f(**self.x_map, only_nominal=True)
            xdotz = self.f(**self.x_map)
        else:
            xdot = self.f(**self.x_map)
            xdotz = None

        self.tlogger.debug(
            f"Nominal Dynamics: {'initialized' if xdot is not None else 'not initialized'}"
        )
        self.tlogger.debug(
            f"Uncertain Dynamics: {'initialized' if xdotz is not None else 'not initialized'}"
        )

        return xdot, xdotz

    def _initialise_data(self):
        datasets = {}
        for label in self.config.DATA_GEN.keys():
            datasets[label] = self.config.DATA_GEN[label](self.config.N_DATA)

        self.tlogger.debug(
            "\n".join(
                ["Data Collection"] + [f"{k}: {v.shape}" for k, v in datasets.items()]
            )
            + "\n"
        )

        return datasets

    def _initialise_certificate(self):
        certificate_type = make_certificate(certificate_type=self.config.CERTIFICATE)
        return certificate_type(
            system=self.f,
            vars=self.x_map,
            domains=self.config.DOMAINS,
            verbose=self.verbose,
            config=self.config,
        )

    def _initialise_consolidator(self):
        return make_consolidator(verbose=self.verbose)

    def _initialise_translator(self):
        return make_translator(
            certificate_type=self.config.CERTIFICATE,
            verifier_type=self.config.VERIFIER,
            time_domain=self.config.TIME_DOMAIN,
            verbose=self.verbose,
        )

    def solve(self) -> CegisResult:
        state = self.init_state()

        iter = None

        # todo pretrain supervised-learning

        for iter in range(1, self.config.CEGIS_MAX_ITERS + 1):
            self.tlogger.info(f"Iteration {iter}")

            # Log training distribution
            context = "dataset"
            for name, dataset in self.datasets.items():
                self.logger.log_scalar(
                    tag=f"{name}_data",
                    value=len(dataset),
                    step=iter,
                    context={context: name},
                )

            # Learner component
            self.tlogger.debug("Learner")
            outputs, elapsed_time = self.learner.update(**state)
            for context, dict_metrics in outputs.items():
                self.logger.log_scalar(
                    tag=None, value=dict_metrics, step=iter, context={context: True}
                )
            self.logger.log_scalar(tag="time_learner", value=elapsed_time, step=iter)

            # Translator component
            self.tlogger.debug("Translator")
            outputs, elapsed_time = self.translator.translate(**state)
            state.update(outputs)
            self.logger.log_scalar(tag="time_translator", value=elapsed_time, step=iter)

            # Verifier component
            self.tlogger.debug("Verifier")
            outputs, elapsed_time = self.verifier.verify(**state)
            state.update(outputs)
            self.logger.log_scalar(tag="time_verifier", value=elapsed_time, step=iter)

            # Consolidator component
            self.tlogger.debug("Consolidator")
            outputs, elapsed_time = self.consolidator.get(**state)
            state.update(outputs)
            self.logger.log_scalar(
                tag="time_consolidator", value=elapsed_time, step=iter
            )

            # Logging
            # logging data distribution
            # for each of them, scatter the counter-examples with different color than the rest of the data
            fig = scatter_datasets(
                datasets=self.datasets, counter_examples=state["cex"]
            )
            self.logger.log_image(tag="datasets", image=fig, step=iter)

            # logging learned functions
            in_domain = self.config.DOMAINS[DomainName.XD.value]
            other_domains = {
                k: v
                for k, v in self.config.DOMAINS.items()
                if k in [DomainName.XI.value, DomainName.XU.value]
            }
            fig = plot_func_and_domains(
                func=self.learner.net,
                in_domain=in_domain,
                levels=[0.0],
                domains=other_domains,
                dim_select=(0, 1),
            )
            self.logger.log_image(tag="barrier", image=fig, step=iter)

            for dim in range(self.f.n_vars):
                func = lambda x: self.learner.net.gradient(x)[:, dim]
                fig = plot_func_and_domains(
                    func=func,
                    in_domain=in_domain,
                    levels=[0.0],
                    domains=other_domains,
                    dim_select=(0, 1),
                )
                self.logger.log_image(
                    tag=f"barrier_grad",
                    image=fig,
                    step=iter,
                    context={"dimension": dim},
                )

            u_domain = self.config.DOMAINS[DomainName.UD.value]
            assert isinstance(
                u_domain, Rectangle
            ), "only rectangular domains are supported for u"
            lb, ub = np.array(u_domain.lower_bounds), np.array(u_domain.upper_bounds)
            for u_norm in np.linspace(-1, 1, 5):
                # denormalize u to the domain
                u = (lb + ub) / 2.0 + u_norm * (ub - lb) / 2.0
                ctrl = (
                    lambda x: torch.ones((x.shape[0], self.f.n_controls))
                    * torch.tensor(u).float()
                )
                if isinstance(self.f, UncertainControlAffineDynamics):
                    f = lambda x, u: self.f._f_torch(
                        x, u, z=torch.zeros((x.shape[0], self.f.n_uncertain))
                    )
                else:
                    f = lambda x, u: self.f._f_torch(x, u)

                # lie derivative
                func = lambda x: lie_derivative_fn(
                    certificate=self.learner.net, f=f, ctrl=ctrl
                )(x)
                fig = plot_func_and_domains(
                    func=func,
                    in_domain=in_domain,
                    levels=[0.0],
                    domains=other_domains,
                    dim_select=(0, 1),
                )
                self.logger.log_image(
                    tag=f"lie_derivative", image=fig, step=iter, context={"u": str(u)}
                )

                # cbf condition
                alpha = lambda x: 1.0 * x
                if isinstance(self.f, UncertainControlAffineDynamics):
                    sigma = self.learner.xsigma
                else:
                    sigma = None
                func = lambda x: cbf_condition_fn(
                    certificate=self.learner.net,
                    alpha=alpha,
                    f=f,
                    ctrl=ctrl,
                    sigma=sigma,
                )(x)
                fig = plot_func_and_domains(
                    func=func,
                    in_domain=in_domain,
                    levels=[0.0],
                    domains=other_domains,
                    dim_select=(0, 1),
                )
                self.logger.log_image(
                    tag=f"cbf_condition", image=fig, step=iter, context={"u": str(u)}
                )

            if isinstance(self.f, UncertainControlAffineDynamics):
                fig = plot_func_and_domains(
                    func=self.learner.xsigma,
                    in_domain=in_domain,
                    levels=[0.0],
                    domains=other_domains,
                    dim_select=(0, 1),
                )
                self.logger.log_image(tag="compensator", image=fig, step=iter)

            # Check termination
            if state["found"]:
                self.tlogger.debug("found valid certificate")
                break

        self.tlogger.info(f"CEGIS finished after {iter} iterations")
        self.logger.log_model(tag="learner", model=self.learner, step=iter)


        infos = {"iter": iter}
        self._result = CegisResult(
            found=state["found"], net=state["V_net"], infos=infos
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
            "V_symbolic_constr": None,  # extra constraints to use with V_symbolic (eg., extra vars)
            "sigma_symbolic": None,  # symbolic expression of compensator sigma
            "sigma_symbolic_constr": None,  # extra constraints to use with sigma_symbolic (eg., extra vars)
            "Vdot_symbolic": None,  # symbolic expression of lie derivative w.r.t. nominal dynamics
            "Vdot_symbolic_constr": None,  # extra constraints to use with Vdot_symbolic (eg., extra vars)
            "Vdotz_symbolic": None,  # symbolic expression of lie derivative w.r.t. uncertain dynamics
            "Vdotz_symbolic_constr": None,  # extra constraints to use with Vdotz_symbolic (eg., extra vars)
            "xdot": self.xdot,  # symbolic expression of nominal dynamics
            "xdotz": self.xdotz,  # symbolic expression of uncertain dynamics
            "cex": None,  # counterexamples
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
