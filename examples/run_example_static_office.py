import torch

from fosco.config import CegisConfig
from fosco.cegis import Cegis
from fosco.common.consts import CertificateType, VerifierType, ActivationType
from fosco.logger import LoggerType
from fosco.systems import make_system


def main(args):
    system_type = "OfficeSingleIntegrator"

    exp_name = args.exp_name
    seed = args.seed
    verifier_type = VerifierType[args.verifier.upper()]
    verifier_n_cex = args.n_cex
    activations = tuple([ActivationType[a.upper()] for a in args.activations])
    n_hidden_neurons = args.n_hiddens
    n_data_samples = args.n_data_samples
    optimizer = args.optimizer
    learning_rate = args.lr
    weight_decay = args.wd
    max_iters = args.max_iters
    n_epochs = args.n_epochs
    verbose = args.verbose

    system = make_system(system_id=system_type)
    sets = system().domains

    # data generator
    data_gen = {
        "init": lambda n: sets["init"].generate_data(n),
        "unsafe": lambda n: sets["unsafe"].generate_data(n),
        "lie": lambda n: torch.concatenate(
            [sets["lie"].generate_data(n), sets["input"].generate_data(n)], dim=1
        ),
    }

    config = CegisConfig(
        EXP_NAME=exp_name,
        SYSTEM=system,
        DOMAINS=sets,
        DATA_GEN=data_gen,
        VERIFIER=verifier_type,
        RESAMPLING_N=verifier_n_cex,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        CEGIS_MAX_ITERS=max_iters,
        N_DATA=n_data_samples,
        SEED=seed,
        LOGGER=LoggerType.AIM,
        N_EPOCHS=n_epochs,
        OPTIMIZER=optimizer,
        LEARNING_RATE=learning_rate,
        WEIGHT_DECAY=weight_decay,
    )
    cegis = Cegis(config=config, verbose=verbose)
    result = cegis.solve()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="exp")
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--verifier", type=str, default="dreal")
    parser.add_argument("--n-cex", type=int, default=20)

    parser.add_argument(
        "--activations", type=str, nargs="+", default=["sigmoid", "linear"]
    )
    parser.add_argument("--n-hiddens", type=int, nargs="+", default=[5, 5])
    parser.add_argument("--n-data-samples", type=int, default=1000)
    parser.add_argument("--max-iters", type=int, default=100)
    parser.add_argument("--n-epochs", type=int, default=1000)
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)

    parser.add_argument("--verbose", type=int, default=1)

    args = parser.parse_args()
    main(args)
