import torch

import fosco.cegis
from fosco.systems import make_system
from fosco.systems.discrete_time.system_dt import EulerDTSystem
from fosco.systems.uncertainty import add_uncertainty


def main(args):
    exp_name = args.exp_name
    seed = args.seed
    system_type = args.system
    dt = args.dt
    uncertainty_type = args.uncertainty
    certificate_type = args.certificate
    verifier_type = args.verifier
    resampling_n = args.resampling_n
    resampling_std = args.resampling_std
    barrier_to_load = args.barrier_to_load
    sigma_to_load = args.sigma_to_load
    activations = args.activations or []
    n_hidden_neurons = args.n_hiddens or []
    n_data_samples = args.n_data_samples
    optimizer = args.optimizer
    learning_rate = args.lr
    weight_decay = args.wd
    loss_act_type = args.loss_act
    loss_init_weight = args.loss_init_weight
    loss_unsafe_weight = args.loss_unsafe_weight
    loss_lie_weight = args.loss_lie_weight
    loss_robust_weight = args.loss_robust_weight
    loss_conservative_b_weight = args.loss_conservative_b_weight
    loss_conservative_sigma_weight = args.loss_conservative_sigma_weight
    max_iters = args.max_iters
    n_epochs = args.n_epochs
    logger_type = "aim"
    verbose = args.verbose

    assert len(n_hidden_neurons) == len(
        activations
    ), "Number of hidden layers must match number of activations"
    assert (
        uncertainty_type is None or certificate_type.upper() == "RCBF"
    ), "Uncertainty only supported for RCBF certificates"

    base_system = make_system(system_id=system_type)()
    if dt is not None:
        base_system = EulerDTSystem(system=base_system, dt=dt)
    system = add_uncertainty(uncertainty_type=uncertainty_type, system=base_system)

    sets = system.domains

    # data generator
    data_gen = {
        "init": lambda n: sets["init"].generate_data(n),
        "unsafe": lambda n: sets["unsafe"].generate_data(n),
    }

    if certificate_type.upper() == "CBF":
        data_gen["lie"] = lambda n: torch.concatenate(
            [sets["lie"].generate_data(n), sets["input"].generate_data(n)], dim=1
        )
    elif certificate_type.upper() == "RCBF":
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
    else:
        raise ValueError(f"Unknown certificate type {certificate_type}")

    config = fosco.cegis.CegisConfig(
        EXP_NAME=exp_name,
        CERTIFICATE=certificate_type,
        BARRIER_TO_LOAD=barrier_to_load,
        SIGMA_TO_LOAD=sigma_to_load,
        VERIFIER=verifier_type,
        RESAMPLING_N=resampling_n,
        RESAMPLING_STDDEV=resampling_std,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        CEGIS_MAX_ITERS=max_iters,
        N_DATA=n_data_samples,
        SEED=seed,
        LOGGER=logger_type,
        N_EPOCHS=n_epochs,
        OPTIMIZER=optimizer,
        LEARNING_RATE=learning_rate,
        WEIGHT_DECAY=weight_decay,
        LOSS_WEIGHTS={
            "init": loss_init_weight,
            "unsafe": loss_unsafe_weight,
            "lie": loss_lie_weight,
            "robust": loss_robust_weight,
            "conservative_b": loss_conservative_b_weight,
            "conservative_sigma": loss_conservative_sigma_weight,
        },
        LOSS_RELU=loss_act_type,
    )
    cegis = fosco.cegis.Cegis(
        system=system, domains=sets, config=config, data_gen=data_gen, verbose=verbose
    )

    result = cegis.solve()


if __name__ == "__main__":
    import argparse
    from fosco.systems.core.system import SYSTEM_REGISTRY
    from fosco.systems.uncertainty import UNCERTAINTY_REGISTRY

    systems = SYSTEM_REGISTRY.keys()
    uncertainties = UNCERTAINTY_REGISTRY.keys()

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="exp")
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument(
        "--system", type=str, default="SingleIntegrator", choices=systems
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Discretization time step for DT systems (None for CT)",
    )
    parser.add_argument("--uncertainty", type=str, default=None, choices=uncertainties)

    parser.add_argument("--certificate", type=str, default="cbf")
    parser.add_argument("--barrier-to-load", type=str, default=None)
    parser.add_argument("--sigma-to-load", type=str, default=None)

    parser.add_argument("--verifier", type=str, default="z3")
    parser.add_argument("--resampling-n", type=int, default=100)
    parser.add_argument("--resampling-std", type=float, default=5e-3)

    parser.add_argument("--activations", type=str, nargs="+", default=None)
    parser.add_argument("--n-hiddens", type=int, nargs="+", default=None)
    parser.add_argument("--n-data-samples", type=int, default=5000)
    parser.add_argument("--max-iters", type=int, default=100)
    parser.add_argument("--n-epochs", type=int, default=1000)
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--loss-act", type=str, default="softplus")

    parser.add_argument("--loss-init-weight", type=float, default=1.0)
    parser.add_argument("--loss-unsafe-weight", type=float, default=1.0)
    parser.add_argument("--loss-lie-weight", type=float, default=1.0)
    parser.add_argument("--loss-robust-weight", type=float, default=1.0)
    parser.add_argument("--loss-conservative-b-weight", type=float, default=0.0)
    parser.add_argument("--loss-conservative-sigma-weight", type=float, default=0.0)

    parser.add_argument("--verbose", type=int, default=1)

    args = parser.parse_args()
    main(args)
