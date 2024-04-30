import logging
import math
import time
from datetime import datetime
from functools import partial
import random

import gymnasium
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from matplotlib.backends.backend_agg import FigureCanvasAgg
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

from fosco.certificates import make_certificate
from fosco.certificates.cbf import TrainableCBF
from fosco.common.domains import Sphere
from fosco.config import CegisConfig
from fosco.learner import make_learner
from fosco.plotting.domains import plot_sphere
from fosco.plotting.functions import plot_torch_function
from fosco.systems import make_system, MultiParticle, UncertainControlAffineDynamics
from fosco.systems.gym_env.system_env import SystemEnv
from fosco.systems.uncertainty import add_uncertainty
from fosco.verifier import make_verifier

XD, XI, XU, ZD = "lie", "init", "unsafe", "robust"

logging.basicConfig(level=logging.DEBUG)


class TransitionDataset(Dataset):
    def __init__(self, generators: dict, n_samples: int):
        self.data = {
            k: generators[k](n_samples) for k in generators
        }
        self.n_samples = n_samples

    def __getitem__(self, item):
        return torch.cat([self.data[k][item] for k in self.data], dim=0)

    def __len__(self):
        return self.n_samples

class ConcatDataset(Dataset):
    def __init__(self, **datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return {kd: d[i] for kd, d in self.datasets.items()}

    def __len__(self):
        return min(len(d) for d in self.datasets)


def make_env():
    sys = make_system("MultiParticleSingleIntegrator")()
    sys = add_uncertainty("DynamicAgents", system=sys)

    env = SystemEnv(system=sys, dt=0.1, max_steps=100, render_mode="human")
    return env


def setup_learner(system, config):
    verbose = 1
    learner_type = make_learner(system=system)

    # variables
    verifier_type = make_verifier(type="z3")
    x = verifier_type.new_vars(var_names=system.vars)
    u = verifier_type.new_vars(var_names=system.controls)
    z = verifier_type.new_vars(var_names=system.uncertain_vars)
    x_map = {"v": x, "u": u, "z": z}

    # certificate
    certificate_type = make_certificate(certificate_type="rcbf")
    certificate = certificate_type(
        system=system, variables=x_map, domains=system.domains, verbose=verbose,
    )

    # learner
    learner_instance = learner_type(
        state_size=system.n_vars,
        hidden_sizes=config.N_HIDDEN_NEURONS,
        activation=config.ACTIVATION,
        optimizer=config.OPTIMIZER,
        epochs=config.N_EPOCHS,
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        loss_margins=config.LOSS_MARGINS,
        loss_weights=config.LOSS_WEIGHTS,
        loss_relu=config.LOSS_RELU,
        verbose=verbose,
    )

    n_uncertain = system.n_uncertain

    # learn binding
    learner_instance.learn_method = partial(
        certificate.learn,
        n_vars=system.n_vars,
        n_controls=system.n_controls,
        n_uncertain=n_uncertain,
        f_torch=system._f_torch,
    )

    return certificate, learner_instance


def prepare_data(domains, n_data: int, train_perc: float):
    sets = domains
    datasets = {
        "init": TransitionDataset(generators={"state": sets["init"].generate_data}, n_samples=n_data),
        "unsafe": TransitionDataset(generators={"state": sets["unsafe"].generate_data}, n_samples=n_data),
        "lie": TransitionDataset(generators={
            "state": sets["lie"].generate_data,
            "control": sets["input"].generate_data,
            "uncertainty": sets["uncertainty"].generate_data
        }, n_samples=n_data),
        "robust": TransitionDataset(generators={
            "state": sets["lie"].generate_data,
            "control": sets["input"].generate_data,
            "uncertainty": sets["uncertainty"].generate_data
        }, n_samples=n_data),
    }
    concat_dataset = ConcatDataset(**datasets)

    n_train = int(train_perc * len(concat_dataset))
    n_val = len(concat_dataset) - n_train
    train_dataset, val_dataset = random_split(concat_dataset, [n_train, n_val])

    return train_dataset, val_dataset


def run_inference(learner, datasets, n_vars, n_controls, n_uncertain, f_torch):
    net = learner.net
    xsigma = learner.xsigma

    i1 = datasets[XD].shape[0]
    i2 = datasets[XI].shape[0]
    i3 = datasets[XU].shape[0]

    states_d = torch.cat([datasets[label][:, :n_vars] for label in [XD, XI, XU]])
    input_d = datasets[XD][:, n_vars: n_vars + n_controls]

    B = net(states_d)
    sigma = xsigma(states_d)

    B_d = B[:i1, 0]
    B_i = B[i1: i1 + i2, 0]
    B_u = B[i1 + i2: i1 + i2 + i3, 0]

    assert (
            B_d.shape[0] == input_d.shape[0]
    ), f"expected pairs of state,input data. Got {B_d.shape[0]} and {input_d.shape[0]}"
    sigma_d = sigma[:i1, 0]
    Bdot_d = TrainableCBF._compute_barrier_difference(
        X_d=datasets[XD][:, :n_vars],
        U_d=datasets[XD][:, n_vars: n_vars + n_controls],
        barrier=net,
        f_torch=partial(f_torch, z=None, only_nominal=True),
    )

    return {
        "B_d": B_d,
        "B_i": B_i,
        "B_u": B_u,
        "Bdot_d": Bdot_d,
        "sigma_d": sigma_d,
    }


def run_inference_compensator(learner, datasets, n_vars, n_controls, n_uncertain, f_torch):
    states_dz = datasets[ZD][:, :n_vars]
    input_dz = datasets[ZD][:, n_vars: n_vars + n_controls]
    uncert_dz = datasets[ZD][
                :, n_vars + n_controls: n_vars + n_controls + n_uncertain,
                ]

    B_dz = learner.net(states_dz)[:, 0]
    sigma_dz = learner.xsigma(states_dz)[:, 0]
    Bdot_dz = TrainableCBF._compute_barrier_difference(
        X_d=states_dz,
        U_d=input_dz,
        barrier=learner.net,
        f_torch=partial(f_torch, z=uncert_dz, only_nominal=True),
    )
    Bdotz_dz = TrainableCBF._compute_barrier_difference(
        X_d=states_dz,
        U_d=input_dz,
        barrier=learner.net,
        f_torch=partial(f_torch, z=uncert_dz, only_nominal=False),
    )

    return {
        "B_dz": B_dz,
        "Bdot_dz": Bdot_dz,
        "Bdotz_dz": Bdotz_dz,
        "sigma_dz": sigma_dz,
    }



def main():
    normalize_data = False
    train_perc = 0.8
    use_cuda = True
    debug = True
    batch_size = 32

    cfg = CegisConfig(
        EXP_NAME="multi-agent",
        CERTIFICATE="rcbf",
        VERIFIER="z3",
        ACTIVATION=["htanh", "htanh"],
        N_HIDDEN_NEURONS=[64, 64],
        N_DATA=1000000,
        SEED=42,
        N_EPOCHS=50000,
        OPTIMIZER="adam",
        LEARNING_RATE=1e-3,
        WEIGHT_DECAY=1e-4,
        LOSS_WEIGHTS={
            "init": 1.0,
            "unsafe": 1.0,
            "lie": 1.0,
            "robust": 1.0,
            "conservative_b": 0.0,
            "conservative_sigma": 0.0,
        },
        LOSS_RELU="softplus",
    )

    # device
    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logging.info(f"Device: {device}")

    # seeding
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    logging.info(f"Seed: {cfg.SEED}")

    # logging
    if not debug:
        datestr = datetime.now().strftime("%Y%m%d-%H%M%S")
        logdir = f"runs/pretraining/Seed{cfg.SEED}-{datestr}"
    else:
        logdir = None
    writer = SummaryWriter(logdir)

    #
    env = make_env()
    n_vars, n_controls, n_uncertain = env.system.n_vars, env.system.n_controls, env.system.n_uncertain
    f_torch = env.system.f

    # learner
    certificate, learner = setup_learner(system=env.system, config=cfg)

    # datasets
    train_datasets, val_datasets = prepare_data(domains=env.system.domains, n_data=cfg.N_DATA, train_perc=train_perc)
    print(train_datasets[0])
    logging.info(f"Data: {len(train_datasets)} training, {len(val_datasets)} validation")

    # data transform
    if normalize_data:
        raise NotImplementedError()
        states_d = torch.cat([train_datasets[label][:, :n_vars] for label in [XD, XI, XU]])
        state_mean = states_d.mean(dim=0)
        state_std = states_d.std(dim=0)
    else:
        state_mean = torch.zeros(n_vars)
        state_std = torch.ones(n_vars)

    transform = transforms.Compose([
        lambda xuz: torch.cat([(xuz[:n_vars] - state_mean) / state_std, xuz[n_vars:]])
    ])
    logging.info(f"Normalization: {state_mean} +/- {state_std}")

    #
    data_loader = DataLoader(
        train_datasets,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )

    # learn
    optimizers = learner.optimizers

    losses = {}
    accuracies = {}
    val_losses = {}
    val_accuracies = {}

    # validation before training
    """
    val_outputs = run_inference(
        learner=learner, datasets=val_datasets, n_vars=n_vars, n_controls=n_controls,
        n_uncertain=n_uncertain, f_torch=f_torch
    )
    val_outputs2 = run_inference_compensator(
        learner=learner, datasets=val_datasets, n_vars=n_vars, n_controls=n_controls,
        n_uncertain=n_uncertain, f_torch=f_torch
    )

    (
        barrier_loss_val,
        barrier_losses_val,
        barrier_accuracies_val,
    ) = TrainableCBF._compute_loss(
        learner=learner,
        B_i=val_outputs["B_i"],
        B_u=val_outputs["B_u"],
        B_d=val_outputs["B_d"],
        Bdot_d=val_outputs["Bdot_d"] - val_outputs["sigma_d"],
        alpha=1.0,
    )
    (
        sigma_loss_val,
        sigma_losses_val,
        sigma_accuracies_val,
    ) = certificate.compute_robust_loss(
        learner=learner,
        B_dz=val_outputs2["B_dz"],
        Bdotz_dz=val_outputs2["Bdotz_dz"],
        Bdot_dz=val_outputs2["Bdot_dz"],
        sigma_dz=val_outputs2["sigma_dz"],
    )

    for k, loss in barrier_losses_val.items():
        writer.add_scalar(f"val_loss/{k}", loss, 0)
    for k, loss in sigma_losses_val.items():
        writer.add_scalar(f"val_loss/{k}", loss, 0)
    for k, accu in barrier_accuracies_val.items():
        writer.add_scalar(f"val_accuracy/{k}", accu, 0)
    for k, accu in sigma_accuracies_val.items():
        writer.add_scalar(f"val_accuracy/{k}", accu, 0)

    logging.info(
        f"Epoch 0: barrier: val loss: {barrier_loss_val}, "
        f"compensator: val loss: {sigma_loss_val:.3f}"
    )
    """

    for t in range(learner.epochs):

        for ib, datasets in enumerate(data_loader):

            optimizers["barrier"].zero_grad()

            # training
            t0 = time.time()
            outputs = run_inference(
                learner=learner, datasets=datasets, n_vars=n_vars, n_controls=n_controls,
                n_uncertain=n_uncertain, f_torch=f_torch
            )
            logging.debug("Inference time: %.3f", time.time() - t0)

            t0 = time.time()
            (
                barrier_loss,
                barrier_losses,
                barrier_accuracies,
            ) = TrainableCBF._compute_loss(
                learner=learner,
                B_i=outputs["B_i"],
                B_u=outputs["B_u"],
                B_d=outputs["B_d"],
                Bdot_d=outputs["Bdot_d"] - outputs["sigma_d"],
                alpha=1.0,
            )
            logging.debug("Barrier loss time: %.3f", time.time() - t0)

            t0 = time.time()
            barrier_loss.backward()
            optimizers["barrier"].step()
            logging.debug("Barrier backward time: %.3f", time.time() - t0)

            # compute output for robust loss
            optimizers["xsigma"].zero_grad()

            t0 = time.time()
            outputs2 = run_inference_compensator(
                learner=learner, datasets=datasets, n_vars=n_vars, n_controls=n_controls,
                n_uncertain=n_uncertain, f_torch=f_torch
            )
            logging.debug("Inference time: %.3f", time.time() - t0)

            t0 = time.time()
            (
                sigma_loss,
                sigma_losses,
                sigma_accuracies,
            ) = certificate.compute_robust_loss(
                learner=learner,
                B_dz=outputs2["B_dz"],
                Bdotz_dz=outputs2["Bdotz_dz"],
                Bdot_dz=outputs2["Bdot_dz"],
                sigma_dz=outputs2["sigma_dz"],
            )
            logging.debug("Compensator loss time: %.3f", time.time() - t0)

            t0 = time.time()
            sigma_loss.backward()
            optimizers["xsigma"].step()
            logging.debug("Compensator backward time: %.3f", time.time() - t0)

            # validation
            t0 = time.time()
            val_outputs = run_inference(
                learner=learner, datasets=val_datasets, n_vars=n_vars, n_controls=n_controls,
                n_uncertain=n_uncertain, f_torch=f_torch
            )
            val_outputs2 = run_inference_compensator(
                learner=learner, datasets=val_datasets, n_vars=n_vars, n_controls=n_controls,
                n_uncertain=n_uncertain, f_torch=f_torch
            )
            logging.debug("Validation inference time: %.3f", time.time() - t0)

            t0 = time.time()
            (
                barrier_loss_val,
                barrier_losses_val,
                barrier_accuracies_val,
            ) = TrainableCBF._compute_loss(
                learner=learner,
                B_i=val_outputs["B_i"],
                B_u=val_outputs["B_u"],
                B_d=val_outputs["B_d"],
                Bdot_d=val_outputs["Bdot_d"] - val_outputs["sigma_d"],
                alpha=1.0,
            )
            logging.debug("Validation barrier loss time: %.3f", time.time() - t0)

            t0 = time.time()
            (
                sigma_loss_val,
                sigma_losses_val,
                sigma_accuracies_val,
            ) = certificate.compute_robust_loss(
                learner=learner,
                B_dz=val_outputs2["B_dz"],
                Bdotz_dz=val_outputs2["Bdotz_dz"],
                Bdot_dz=val_outputs2["Bdot_dz"],
                sigma_dz=val_outputs2["sigma_dz"],
            )
            logging.debug("Validation compensator loss time: %.3f", time.time() - t0)

            t0 = time.time()
            for k, loss in barrier_losses.items():
                writer.add_scalar(f"train_loss/{k}", loss, t + 1)
            for k, loss in sigma_losses.items():
                writer.add_scalar(f"train_loss/{k}", loss, t + 1)
            for k, accu in barrier_accuracies.items():
                writer.add_scalar(f"train_accuracy/{k}", accu, t + 1)
            for k, accu in sigma_accuracies.items():
                writer.add_scalar(f"train_accuracy/{k}", accu, t + 1)

            for k, loss in barrier_losses_val.items():
                writer.add_scalar(f"val_loss/{k}", loss, t + 1)
            for k, loss in sigma_losses_val.items():
                writer.add_scalar(f"val_loss/{k}", loss, t + 1)
            for k, accu in barrier_accuracies_val.items():
                writer.add_scalar(f"val_accuracy/{k}", accu, t + 1)
            for k, accu in sigma_accuracies_val.items():
                writer.add_scalar(f"val_accuracy/{k}", accu, t + 1)
            logging.debug("Logging time: %.3f", time.time() - t0)

            if t % math.ceil(min(1000, learner.epochs / 10)) == 0:
                # log_loss_acc(t, loss, accuracy, learner.verbose)
                logging.info(
                    f"Epoch {t + 1}: barrier: loss: {barrier_loss:.3f}, val loss: {barrier_loss_val}, "
                    f"compensator: loss: {sigma_loss:.3f}, val loss: {sigma_loss_val:.3f}"
                )

                t0 = time.time()
                images = plot_functions(learner, env, datasets=val_datasets, seed=cfg.SEED, state_mean=state_mean,
                                        state_std=state_std)
                for name, img in images.items():
                    writer.add_image(name, img, t, dataformats="HWC")
                logging.debug("Plotting time: %.3f", time.time() - t0)

    writer.close()

    return {
        "loss": losses,
        "accuracy": accuracies,
        "val_loss": val_losses,
        "val_accuracy": val_accuracies,
    }


def plot_functions_old(learner, env):
    images = {}
    for name, fn in zip(["barrier", "compensator"], [learner.net, learner.xsigma]):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        canvas = FigureCanvasAgg(fig)

        fig = plot_torch_function(
            function=fn,
            domains={XD: env.system.domains[XD]},
            fig=fig
        )
        canvas.draw()
        s, (width, height) = canvas.print_to_buffer()

        image_from_plot = np.frombuffer(s, np.uint8).reshape((height, width, 4))
        images[name] = image_from_plot

    return images


def plot_functions(learner, env, datasets, seed, state_mean=0.0, state_std=1.0):
    images = {}

    n_samples = 5
    # states0, _ = env.reset(seed=seed, options={"batch_size": n_samples})
    # if states0.ndim == 1:
    #    states0 = states0[None, :]
    states0 = datasets[XD][:n_samples]

    domain = env.system.domains[XD]
    xy_lb = domain.lower_bounds[:2]
    xy_ub = domain.upper_bounds[:2]
    coll_radius = env.system._base_system._base_system._collision_distance

    x = np.linspace(xy_lb[0], xy_ub[0], 100)
    y = np.linspace(xy_lb[1], xy_ub[1], 100)
    X, Y = np.meshgrid(x, y)

    for i, state0 in enumerate(states0):
        # get state
        ego_xy, npc0_dxy, npc1_dxy = state0[:2], state0[2:4], state0[4:6]
        npc0_xy = ego_xy + npc0_dxy
        npc1_xy = ego_xy + npc1_dxy

        # compute relative coordinates for all ego x, y
        npc0_X, npc0_Y = npc0_xy[0] - X, npc0_dxy[1] - Y
        npc1_X, npc1_Y = npc1_xy[0] - X, npc1_dxy[1] - Y

        all_states = np.stack([X, Y, npc0_X, npc0_Y, npc1_X, npc1_Y], axis=-1).reshape(-1, 6)
        all_states = (all_states - state_mean) / state_std

        for name, fn in zip(["barrier", "compensator"], [learner.net, learner.xsigma]):
            with torch.no_grad():
                z_vals = fn(torch.tensor(all_states, dtype=torch.float32)).numpy()
            z_vals = z_vals.reshape(X.shape)

            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.view_init(elev=90, azim=90, roll=0)  # top view
            canvas = FigureCanvasAgg(fig)

            ax.plot_surface(
                X, Y, z_vals, cmap=cm.plasma, alpha=0.5
            )
            ax.contour(X, Y, z_vals, levels=[0.0], colors="k", linewidths=2)

            # plot npc as sphere
            for npc_xy in [npc0_xy, npc1_xy]:
                plot_sphere(
                    domain=Sphere(vars=["x", "y"], center=npc_xy, radius=coll_radius),
                    color="red",
                    fig=fig,
                )

            ax.set_xlim(xy_lb[0], xy_ub[0])
            ax.set_ylim(xy_lb[1], xy_ub[1])

            canvas.draw()
            s, (width, height) = canvas.print_to_buffer()

            image_from_plot = np.frombuffer(s, np.uint8).reshape((height, width, 4))
            images[f"{name}/state{i}"] = image_from_plot

            plt.close(fig)

    return images


if __name__ == "__main__":
    main()
