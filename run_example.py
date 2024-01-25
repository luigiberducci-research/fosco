import torch

import fosco.cegis
from fosco.common.consts import ActivationType, LossReLUType
from fosco.common.consts import CertificateType, TimeDomain, VerifierType
from fosco.logger import LoggerType
from systems import make_system, make_domains, add_uncertainty


def main(args):
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
        SYSTEM=system,
        DOMAINS=sets,
        DATA_GEN=data_gen,
        CERTIFICATE=certificate_type,
        USE_INIT_MODELS=use_init_models,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.Z3,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        CEGIS_MAX_ITERS=max_iters,
        N_DATA=n_data_samples,
        SEED=seed,
        LOGGER=LoggerType.AIM,
        N_EPOCHS=n_epochs,
        OPTIMIZER=optimizer,
        LOSS_MARGINS={"init": 0.0, "unsafe": 0.0, "lie": 0.0, "robust": 0.0},
        LOSS_WEIGHTS={"init": 1.0, "unsafe": 1.0, "lie": 1.0, "robust": 1.0,
                      "conservative_b": 0.10, "conservative_sigma": 0.0},
        LOSS_RELU=loss_act_type,
        LOSS_NETGRAD_WEIGHT=loss_netgrad_weight,
    )
    cegis = fosco.cegis.Cegis(config=config, verbose=verbose)

    result = cegis.solve()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--system", type=str, default="single_integrator")
    parser.add_argument("--uncertainty", type=str, default=None)
    parser.add_argument("--certificate", type=str, default="cbf")
    parser.add_argument("--use-init-models", action="store_true")
    parser.add_argument(
        "--activations", type=str, nargs="+", default=["relu", "linear"]
    )
    parser.add_argument("--n-hiddens", type=int, nargs="+", default=[5, 5])
    parser.add_argument("--n-data-samples", type=int, default=1000)
    parser.add_argument("--max-iters", type=int, default=100)
    parser.add_argument("--n-epochs", type=int, default=1000)
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--loss-act", type=str, default="softplus")
    parser.add_argument("--loss-netgrad-weight", type=float, default=0.0)
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    main(args)
