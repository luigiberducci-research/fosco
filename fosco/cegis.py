import logging
from datetime import datetime
from typing import Callable

import numpy as np
import torch

from barriers import make_barrier, make_compensator
from fosco.certificates import make_certificate, Certificate
from fosco.common.domains import Rectangle, Set
from fosco.config import CegisConfig, CegisResult
from fosco.consolidator import make_consolidator, Consolidator
from fosco.common.consts import DomainName, CertificateType
from fosco.learner import make_learner, LearnerNN
from fosco.plotting.data import scatter_datasets
from fosco.plotting.functions import (
    plot_torch_function,
    plot_torch_function_grads,
    plot_lie_derivative,
    plot_cbf_condition,
)
from fosco.translator import make_translator, Translator
from fosco.verifier import make_verifier, Verifier
from fosco.logger import make_logger, Logger, LOGGING_LEVELS
from fosco.systems import UncertainControlAffineDynamics, ControlAffineDynamics
from fosco.verifier.types import SYMBOL


class Cegis:
    def __init__(
        self,
        system: ControlAffineDynamics,
        domains: dict[str, Set],
        config: CegisConfig,
        data_gen: dict[str, Callable[[int], torch.Tensor]],
        verbose: int = 0,
    ):  
        self.f = system
        self.domains = domains
        self.config = config
        self.data_gen = data_gen
        self.verbose = min(max(verbose, 0), len(LOGGING_LEVELS) - 1)

        # seeding
        if self.config.SEED is None:
            self.config.SEED = torch.randint(0, 1000000, (1,)).item()
        torch.manual_seed(self.config.SEED)
        np.random.seed(self.config.SEED)

        # logging
        self.logger, self.tlogger = self._initialise_logger()

        # domains, dynamics, data
        self.x, self.x_map = self._initialise_variables()
        self.xdot, self.xdotz, self.xdot_residual = self._initialise_dynamics()
        self.datasets = self._initialise_data()

        # cegis components
        self.certificate = self._initialise_certificate()
        self.learner = self._initialise_learner()
        self.verifier = self._initialise_verifier()
        self.consolidator = self._initialise_consolidator()
        self.translator = self._initialise_translator()

        # sanity check
        self._result = None
        self._assert_state()
        self.tlogger.info(f"Seed: {self.config.SEED}")

    def _initialise_logger(self) -> Logger:
        # make experiment name
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{self.f.id}_{self.config.CERTIFICATE}_{self.config.EXP_NAME}_Seed{self.config.SEED}_{datetime_str}"
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
        if self.config.BARRIER_TO_LOAD is not None:
            initial_models["net"] = make_barrier(
                system=self.f, model_to_load=self.config.BARRIER_TO_LOAD
            )
        if self.config.SIGMA_TO_LOAD is not None:
            initial_models["xsigma"] = make_compensator(
                system=self.f, model_to_load=self.config.SIGMA_TO_LOAD
            )

        learner_instance = learner_type(
            state_size=self.f.n_vars,
            learn_method=self.certificate.learn,
            hidden_sizes=self.config.N_HIDDEN_NEURONS,
            activation=self.config.ACTIVATION,
            optimizer=self.config.OPTIMIZER,
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
            initial_models=initial_models,
            verbose=self.verbose,
        )
        return learner_instance

    def _initialise_verifier(self):
        verifier_type = make_verifier(type=self.config.VERIFIER)
        verifier_instance = verifier_type(
            solver_vars=self.x,
            constraints_method=self.certificate.get_constraints,
            rounding=self.config.ROUNDING,
            solver_timeout=self.config.VERIFIER_TIMEOUT,
            verbose=self.verbose,
        )
        return verifier_instance

    def _initialise_variables(self):
        verifier_type = make_verifier(type=self.config.VERIFIER)
        x = verifier_type.new_vars(var_names=self.f.vars)
        u = verifier_type.new_vars(var_names=self.f.controls)

        if isinstance(self.f, UncertainControlAffineDynamics):
            z = verifier_type.new_vars(var_names=self.f.uncertain_vars)
        else:
            z = None

        # create map id -> variable
        if z:
            x_map = {"v": x, "u": u, "z": z}
            x = x + u + z
        else:
            x_map = {"v": x, "u": u}
            x = x + u

        return x, x_map

    def _initialise_dynamics(self):
        if isinstance(self.f, UncertainControlAffineDynamics):
            xdot = self.f(**self.x_map, only_nominal=True)
            xdotz = None  # self.f(**self.x_map)
            v, u, z = self.x_map["v"], self.x_map["u"], self.x_map["z"]
            xdot_residual = self.f.fz_smt(v, z) + self.f.gz_smt(v, z) @ u
        else:
            xdot = self.f(**self.x_map)
            xdotz = None
            xdot_residual = None

        self.tlogger.debug(
            f"Nominal Dynamics: {'initialized' if xdot is not None else 'not initialized'}"
        )
        self.tlogger.debug(
            f"Uncertain Dynamics: {'initialized' if xdot_residual is not None else 'not initialized'}"
        )

        return xdot, xdotz, xdot_residual

    def _initialise_data(self) -> dict[str, torch.Tensor]:
        datasets = {}
        for label, generator in self.data_gen.items():
            datasets[label] = generator(self.config.N_DATA)

        self.tlogger.debug(
            "\n".join(
                ["Data Collection"] + [f"{k}: {v.shape}" for k, v in datasets.items()]
            )
            + "\n"
        )

        return datasets

    def _initialise_certificate(self) -> Certificate:
        certificate_type = make_certificate(certificate_type=self.config.CERTIFICATE)
        return certificate_type(
            system=self.f,
            variables=self.x_map,
            domains=self.domains,
            config=self.config,
            verbose=self.verbose,
        )

    def _initialise_consolidator(self) -> Consolidator:
        return make_consolidator(
            resampling_n=self.config.RESAMPLING_N,
            resampling_stddev=self.config.RESAMPLING_STDDEV,
            verbose=self.verbose,
        )

    def _initialise_translator(self) -> Translator:
        return make_translator(
            certificate_type=self.config.CERTIFICATE,
            verifier_type=self.config.VERIFIER,
            time_domain=self.config.TIME_DOMAIN,
            verbose=self.verbose,
        )

    def solve(self) -> CegisResult:
        state = self.init_state()

        it = None

        # todo pretrain supervised-learning

        for it in range(1, self.config.CEGIS_MAX_ITERS + 1):
            self.tlogger.info(f"Iteration {it}")

            # Log training distribution
            context = "dataset"
            for name, dataset in self.datasets.items():
                self.logger.log_scalar(
                    tag=f"{name}_data",
                    value=len(dataset),
                    step=it,
                    context={context: name},
                )

            # Learner component
            self.tlogger.debug("Learner")
            outputs, elapsed_time = self.learner.update(**state)
            for context, dict_metrics in outputs.items():
                self.logger.log_scalar(
                    tag=None, value=dict_metrics, step=it, context={context: True}
                )
            self.logger.log_scalar(tag="time_learner", value=elapsed_time, step=it)

            # Translator component
            self.tlogger.debug("Translator")
            outputs, elapsed_time = self.translator.translate(**state)
            state.update(outputs)
            self.logger.log_scalar(tag="time_translator", value=elapsed_time, step=it)

            # Verifier component
            self.tlogger.debug("Verifier")
            outputs, elapsed_time = self.verifier.verify(**state)
            state.update(outputs)
            self.logger.log_scalar(tag="time_verifier", value=elapsed_time, step=it)

            # Consolidator component
            self.tlogger.debug("Consolidator")
            outputs, elapsed_time = self.consolidator.get(**state)
            state.update(outputs)
            self.logger.log_scalar(
                tag="time_consolidator", value=elapsed_time, step=it
            )

            # Debug plot - Learned functions
            self._plot_all(state=state, iteration=it)

            self.logger.log_model(tag="learner", model=self.learner, step=it)

            # Check termination
            if state["found"]:
                self.tlogger.debug("found valid certificate")
                break

        self.tlogger.info(f"CEGIS finished after {it} iterations")
        self.logger.log_model(tag="learner_final", model=self.learner, step=it)

        infos = {"iter": it}
        self._result = CegisResult(
            found=state["found"],
            barrier=state["V_net"],
            compensator=state["sigma_net"],
            infos=infos,
        )

        return self._result

    def _plot_all(self, state: dict, iteration: int) -> None:
        """
        Plot learned functions, gradients, lie derivatives, and CBF conditions.
        """

        # data distr: for each of them, scatter the counter-examples with different color than the rest of the data
        fig = scatter_datasets(datasets=self.datasets, counter_examples=state["cex"])
        self.logger.log_image(tag="datasets", image=fig, step=iteration)

        # logging learned functions
        fig = plot_torch_function(domains=self.domains, function=self.learner.net)
        self.logger.log_image(tag="barrier", image=fig, step=iteration)

        figs, titles = plot_torch_function_grads(
            domains=self.domains, function=self.learner.net
        )
        for title, fig in zip(titles, figs):
            self.logger.log_image(
                tag="barrier_grad", image=fig, step=iteration, context={"dimension": title}
            )

        figs, titles = plot_lie_derivative(
            function=self.learner.net, system=self.f, domains=self.domains
        )
        for title, fig in zip(titles, figs):
            self.logger.log_image(
                tag=f"lie_derivative", image=fig, step=iteration, context={"u": title}
            )

        if isinstance(self.f, UncertainControlAffineDynamics):
            fig = plot_torch_function(
                function=self.learner.xsigma,
                domains=self.domains,
            )
            self.logger.log_image(tag="compensator", image=fig, step=iteration)

            sigma = self.learner.xsigma
        else:
            sigma = None

        figs, titles = plot_cbf_condition(
            barrier=self.learner.net,
            system=self.f,
            domains=self.domains,
            compensator=sigma,
        )
        for title, fig in zip(titles, figs):
            self.logger.log_image(
                tag=f"cbf_condition", image=fig, step=iteration, context={"u": title}
            )

    def init_state(self) -> dict:
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
            "V_symbolic_vars": None,  # symbolic variables to use with V_symbolic_constr
            "sigma_symbolic": None,  # symbolic expression of compensator sigma
            "sigma_symbolic_constr": None,  # extra constraints to use with sigma_symbolic (eg., extra vars)
            "sigma_symbolic_vars": None,  # symbolic variables to use with sigma_symbolic_constr
            "Vdot_symbolic": None,  # symbolic expression of lie derivative w.r.t. nominal dynamics
            "Vdot_symbolic_constr": None,  # extra constraints to use with Vdot_symbolic (eg., extra vars)
            "Vdot_symbolic_vars": None,  # symbolic variables to use with Vdot_symbolic_constr
            "Vdot_residual_symbolic": None,  # symbolic expression of lie derivative residual (Vdotz - Vdot)
            "Vdot_residual_symbolic_constr": None,  # extra constraints to use with Vdot_residual_symbolic
            "Vdot_residual_symbolic_vars": None,  # symbolic variables to use with Vdot_residual_symbolic_constr
            "xdot": self.xdot,  # symbolic expression of nominal dynamics
            "xdot_residual": self.xdot_residual,  # symbolic expression of dynamics residual (xdotz - xdot)
            "cex": None,  # counterexamples
        }

        return state

    @property
    def result(self):
        return self._result

    def _assert_state(self):
        assert isinstance(
            self.f, ControlAffineDynamics
        ), f"expected control affine dynamics, got {type(self.f)}"
        assert isinstance(
            self.domains, dict
        ), f"expected dictionary of domains, got {type(self.domains)}"
        assert all(
            [isinstance(dom, Set) for dom in self.domains.values()]
        ), f"expected dictionary of Set, got {self.domains}"
        assert all(
            [isinstance(v, SYMBOL) for v in self.x]
        ), f"expected symbolic variables, got {self.x}"

        assert isinstance(
            self.certificate, Certificate
        ), f"expected Certificate, got {type(self.certificate)}"
        assert isinstance(
            self.verifier, Verifier
        ), f"expected Verifier, got {type(self.verifier)}"
        assert isinstance(
            self.learner, LearnerNN
        ), f"expected Verifier, got {type(self.learner)}"
        assert isinstance(
            self.translator, Translator
        ), f"expected Translator, got {type(self.translator)}"
        assert isinstance(
            self.consolidator, Consolidator
        ), f"expected Consolidator, got {type(self.consolidator)}"

        assert (
            self.config.LEARNING_RATE > 0
        ), f"expected positive learning rate, got {self.config.LEARNING_RATE}"
        assert (
            self.config.CEGIS_MAX_ITERS > 0
        ), f"expected positive max iterations, got {self.config.CEGIS_MAX_ITERS}"
        assert (
            self.config.N_DATA >= 0
        ), f"expected non-negative number of data samples, got {self.config.N_DATA}"
        assert (
            self.x is self.verifier.xs
        ), "expected same variables in fosco and verifier"
