import torch

import fosco.cegis
from fosco.common.consts import ActivationType, LossReLUType
from fosco.common.consts import CertificateType, TimeDomain, VerifierType
from fosco.logger import LoggerType
from systems import make_system, make_domains
from systems.uncertainty import add_uncertainty


def main(args):
    exp_name = args.exp_name
    seed = args.seed
    system_type = args.system
    uncertainty_type = args.uncertainty
    certificate_type = CertificateType[args.certificate.upper()]
    use_init_models = args.use_init_models
    activations = tuple([ActivationType[a.upper()] for a in args.activations])
    n_hidden_neurons = args.n_hiddens
    n_data_samples = args.n_data_samples
    optimizer = args.optimizer
    loss_act_type = LossReLUType[args.loss_act.upper()]
    loss_netgrad_weight = args.loss_netgrad_weight
    loss_init_weight = args.loss_init_weight
    loss_unsafe_weight = args.loss_unsafe_weight
    loss_lie_weight = args.loss_lie_weight
    loss_robust_weight = args.loss_robust_weight
    loss_conservative_b_weight = args.loss_conservative_b_weight
    loss_conservative_sigma_weight = args.loss_conservative_sigma_weight
    max_iters = args.max_iters
    n_epochs = args.n_epochs
    verbose = args.verbose

    assert len(n_hidden_neurons) == len(
        activations
    ), "Number of hidden layers must match number of activations"
    assert (
            uncertainty_type is None or certificate_type == CertificateType.RCBF
    ), "Uncertainty only supported for RCBF certificates"

    base_system = make_system(system_id=system_type)
    system = add_uncertainty(uncertainty_type=uncertainty_type, system_fn=base_system)
    sets = make_domains(system_id=system_type)

    if certificate_type == CertificateType.CBF:
        sets = {
            k: s for k, s in sets.items() if k in ["lie", "input", "init", "unsafe"]
        }

    # data generator
    data_gen = {
        "init": lambda n: sets["init"].generate_data(n),
        "unsafe": lambda n: sets["unsafe"].generate_data(n),
    }

    if certificate_type == CertificateType.CBF:
        data_gen["lie"] = lambda n: torch.concatenate(
            [sets["lie"].generate_data(n), sets["input"].generate_data(n)], dim=1
        )
    else:
        data_gen["lie"] = lambda n: torch.concatenate(
            [
                sets["lie"].generate_data(n),
                torch.zeros(n, sets["input"].dimension),
                sets["uncertainty"].generate_data(n),
            ],
            dim=1,
        )
        data_gen["uncertainty"] = lambda n: torch.concatenate(
            [
                sets["lie"].generate_data(n),
                sets["input"].generate_data(n),
                sets["uncertainty"].generate_data(n),
            ],
            dim=1,
        )

    config = fosco.cegis.CegisConfig(
        EXP_NAME=exp_name,
        SYSTEM=system,
        UNCERTAINTY=uncertainty_type,
        DOMAINS=sets,
        DATA_GEN=data_gen,
        CERTIFICATE=certificate_type,
        USE_INIT_MODELS=use_init_models,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.Z3,
        VERIFIER_N_CEX=20,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        CEGIS_MAX_ITERS=max_iters,
        N_DATA=n_data_samples,
        SEED=seed,
        LOGGER=LoggerType.AIM,
        N_EPOCHS=n_epochs,
        OPTIMIZER=optimizer,
        LOSS_MARGINS={"init": 0.0, "unsafe": 0.0, "lie": 0.0, "robust": 0.0},
        LOSS_WEIGHTS={"init": loss_init_weight, "unsafe": loss_unsafe_weight,
                      "lie": loss_lie_weight, "robust": loss_robust_weight,
                      "conservative_b": loss_conservative_b_weight,
                      "conservative_sigma": loss_conservative_sigma_weight},
        LOSS_RELU=loss_act_type,
        LOSS_NETGRAD_WEIGHT=loss_netgrad_weight,
    )
    cegis = fosco.cegis.Cegis(config=config, verbose=verbose)

    result = cegis.solve()


if __name__ == "__main__":
    import argparse
    from systems.system import SYSTEM_REGISTRY
    from systems.uncertainty.uncertainty_wrapper import UNCERTAINTY_REGISTRY

    systems = SYSTEM_REGISTRY.keys()
    uncertainties = UNCERTAINTY_REGISTRY.keys()

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="exp")
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--system", type=str, default="SingleIntegrator", choices=systems)
    parser.add_argument("--uncertainty", type=str, default=None, choices=uncertainties)

    parser.add_argument("--certificate", type=str, default="cbf")

    parser.add_argument("--use-init-models", action="store_true")
    parser.add_argument(
        "--activations", type=str, nargs="+", default=["square", "linear"]
    )
    parser.add_argument("--n-hiddens", type=int, nargs="+", default=[5, 5])
    parser.add_argument("--n-data-samples", type=int, default=1000)
    parser.add_argument("--max-iters", type=int, default=100)
    parser.add_argument("--n-epochs", type=int, default=1000)
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--loss-act", type=str, default="softplus")

    parser.add_argument("--loss-netgrad-weight", type=float, default=0.0)
    parser.add_argument("--loss-init-weight", type=float, default=1.0)
    parser.add_argument("--loss-unsafe-weight", type=float, default=1.0)
    parser.add_argument("--loss-lie-weight", type=float, default=1.0)
    parser.add_argument("--loss-robust-weight", type=float, default=1.0)
    parser.add_argument("--loss-conservative-b-weight", type=float, default=1.5)
    parser.add_argument("--loss-conservative-sigma-weight", type=float, default=0.0)

    parser.add_argument("--verbose", type=int, default=1)

    args = parser.parse_args()
    main(args)
