import warnings
from pathlib import Path
import uuid
import datetime
from functools import partial

#import mlflow
import tomli_w

import numpy as onp
from flax.core.lift import switch

import functools

import jax
from jax import tree_util, lax, random, nn, experimental

import jax.numpy as jnp

import matplotlib.pyplot as plt
from jax.example_libraries.optimizers import nesterov
from jax.experimental import mesh_utils

from jax_md import simulate, partition, space, util, energy, \
    quantity as snapshot_quantity, minimize

from jax_md_mod import custom_energy, custom_space, custom_quantity, custom_simulate
from jax_md_mod.model import layers, neural_networks

import optax

import haiku as hk

import e3nn_jax

from chemtrain.trainers import ForceMatching, Difftre
from chemtrain.ensemble import sampling
from chemtrain import util as chem_util
from jax_md_mod import custom_space, custom_partition
from jax_md import energy

from chemutils.models.nequip import nequip_neighborlist_pp
from chemutils.models.mace import mace_neighborlist_pp
from chemutils.models.allegro import allegro_neighborlist_pp
from chemutils.models.painn import painn_neighborlist_pp
from chemutils.visualize import molecule
from cycler import cycler


def define_model(config,
                 dataset=None,
                 nbrs_init=None,
                 max_edges=None,
                 per_particle=False,
                 avg_num_neighbors=1.0,
                 positive_species=False,
                 displacement_fn=None,
                 species=None,
                 **kwargs):
    """Initializes a concrete model for a system given path to model parameters."""

    if displacement_fn is None:
        print(f"Use non-periodic displacement")
        displacement_fn, _ = custom_space.nonperiodic_general(
            fractional_coordinates=False)

    # Requirement to capture all species in the dataset
    n_species = 55

    model_type = config["model"].get("type", "NequIP")
    print(f"Run model {model_type}")
    if model_type == "NequIP":
        init_fn, gnn_energy_fn = nequip_neighborlist_pp(
            displacement_fn, config["model"]["r_cutoff"], n_species,
            max_edges=max_edges, per_particle=per_particle,
            avg_num_neighbors=avg_num_neighbors, mode="energy",
            **config["model"]["model_kwargs"],
            positive_species=positive_species,
        )
    elif model_type == "MACE":
        init_fn, gnn_energy_fn = mace_neighborlist_pp(
            displacement_fn, config["model"]["r_cutoff"], n_species,
            max_edges=max_edges, output_irreps="1x0e",
            per_particle=per_particle,
            avg_num_neighbors=avg_num_neighbors, mode="energy",
            positive_species=positive_species,
            **config["model"]["model_kwargs"]
        )
    elif model_type == "Allegro":
        init_fn, gnn_energy_fn = allegro_neighborlist_pp(
            displacement_fn, config["model"]["r_cutoff"], n_species,
            max_edges=max_edges, output_irreps="1x0e",
            per_particle=per_particle,
            avg_num_neighbors=avg_num_neighbors, mode="energy",
            positive_species=positive_species,
            **config["model"]["model_kwargs"]
        )
    elif model_type == "PaiNN":
        init_fn, gnn_energy_fn = painn_neighborlist_pp(
            displacement_fn, config["model"]["r_cutoff"], n_species,
            max_edges=max_edges,
            per_particle=per_particle,
            avg_num_neighbors=avg_num_neighbors,
            positive_species=positive_species,
            mode="energy",
            **config["model"]["model_kwargs"]
        )
    else:
        raise NotImplementedError(f"Model {model_type} not implemented.")


    def energy_fn_template(energy_params):
        def energy_fn(pos, neighbor, mode=None, **dynamic_kwargs):
            assert 'species' in dynamic_kwargs.keys(), 'species not in dynamic_kwargs'

            if "mask" not in dynamic_kwargs:
                print(f"Add defaul all-positive mask.")
                dynamic_kwargs["mask"] = jnp.ones(pos.shape[0], dtype=jnp.bool_)

            if "box" in dynamic_kwargs:
                print(f"Found box in energy kwargs")

            return gnn_energy_fn(
                energy_params, pos, neighbor, **dynamic_kwargs
            )

        return energy_fn

    if dataset is None:
        return energy_fn_template

    # Set up NN model
    r_init = jnp.asarray(dataset['R'][0])
    species_init = jnp.asarray(dataset['species'][0])
    # mask_init = jnp.asarray(dataset['mask'][0])

    nbrs_init = nbrs_init.update(r_init)

    key = random.PRNGKey(config.get("seed", 0))

    try:
        top = molecule.topology_from_neighbor_list(nbrs_init, species_init)
        fig = molecule.plot_molecule(r_init, top)
        fig.savefig("molecule.pdf", bbox_inches="tight")

    except NotImplementedError:
        print("Could not plot molecule")

    # Load a pretrained model
    init_params = init_fn(
        key, r_init, nbrs_init, species=species_init,
    )

    # print(f"Init params: {init_params}")
    print(f"Initial energy is {jax.jit(energy_fn_template(init_params))(r_init, nbrs_init, species=species_init)}")
    # print(f"Initial forces are {jax.jit(jax.grad(energy_fn_template(init_params)))(r_init, nbrs_init, mask=mask_init, species=species_init)}")

    return energy_fn_template, init_params


def init_simulator(config, shift_fn):
    """Initializes simulator for ReSOLV"""
    sim_settings = config["simulator_settings"]
    simulator_template = functools.partial(
        custom_simulate.nvt_langevin_gsd, shift_fn=shift_fn,
        dt=sim_settings["dt"], kT=sim_settings["kT"],
        gamma=sim_settings["gamma"]
    )

    timings = sampling.process_printouts(
        sim_settings["dt"], sim_settings["total_time"],
        sim_settings["t_equilib"], sim_settings["print_every"]
    )

    return simulator_template, timings


def init_reference_state(key, simulator_template, timings, nbrs_init, dataset, energy_fn_template, init_params, mass_repartitioning=False):
    energy_fn = energy_fn_template(init_params)
    init_simulator, apply_simulator = simulator_template(energy_fn)
    mass_multiplier = 2.0

    gen_traj_fn = sampling.trajectory_generator_init(
        simulator_template, energy_fn_template, timings,
        quantities={"energy": custom_quantity.energy_wrapper(energy_fn_template)},
        vmap_batch=1
    )

    mesh = jax.sharding.Mesh(jax.devices(), axis_names=('batch'))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('batch',))

    def n_clusters(state, neighbor=None, mask=None, **kwargs):
        _, num = custom_partition.find_clusters(neighbor, mask)
        return num

    @jax.vmap
    def _init_ref_state(split, sample):
        # Performa mass repartitioning to hydrogen atoms
        senders, receivers = sample["bonds"]

        init_mass = jnp.asarray(sample["mass"])
        init_species = jnp.asarray(sample["species"])

        # Edge must be valid, sender must be a heavy atom, receiver must be a light atom
        heavy_sender = (init_species[senders] > 1) & (senders < init_mass.size)
        light_receiver = (init_species[receivers] == 1) & (receivers < init_mass.size)
        send_mass = jnp.float_(heavy_sender & light_receiver)

        if mass_repartitioning:
            # Mass repartitioning (1 u from heavy to light atom)
            init_mass -= mass_multiplier * jax.ops.segment_sum(send_mass, senders, init_mass.size)
            init_mass += mass_multiplier * jax.ops.segment_sum(send_mass, receivers, init_mass.size)

        nbrs = nbrs_init.update(sample["R"], mask=sample["mask"])
        reference_state = sampling.SimulatorState(
            sim_state=init_simulator(
                split, sample["R"],
                neighbor=nbrs, mass=init_mass,
                species=init_species, mask=sample["mask"],),
            nbrs=nbrs,
        )

        # Generate a reference trajectory for testing purposes
        ref_traj = gen_traj_fn(init_params, reference_state, species=init_species, mask=sample["mask"], id=sample["id"])
        quantities = sampling.quantity_traj(
            ref_traj, quantities={"nclusters": n_clusters},
            energy_params=init_params
        )

        final_clusters = custom_partition.find_clusters(ref_traj.sim_state.nbrs, mask=sample["mask"])

        return ref_traj, final_clusters, quantities

    @functools.partial(jax.jit, in_shardings=sharding, out_shardings=sharding)
    def init_ref_state(split, dataset):
        jax.debug.visualize_array_sharding(dataset["mask"])
        return _init_ref_state(split, dataset)

    sharded_args = random.split(key, dataset["R"].shape[0]), dataset
    n_samples = dataset["R"].shape[0]

    remainder = n_samples % jax.device_count()
    if remainder > 0:
        # Need to pad the dataset to make it divisible by the number of devices
        pad = jax.device_count() - remainder
        sharded_args = tree_util.tree_map(
            lambda x: jnp.concatenate([x, x[:pad]], axis=0), sharded_args)

    sharded_args = jax.device_put(sharded_args, sharding)
    jax.debug.visualize_array_sharding(sharded_args[1]["mask"])
    trajstates, (clusters, nclusters), quants = init_ref_state(*sharded_args)
    stable = (nclusters == 1)

    # Remove the padded entries
    if remainder > 0:
        pad = jax.device_count() - remainder
        trajstates, stable, quants = tree_util.tree_map(
            lambda x: x[:-pad], (trajstates, stable, quants)
        )

    print(f"Removed {onp.sum(~stable)} of {stable.size} entries from the dataset.")
    print(f"Remaining IDs in the dataset: {dataset['id'][stable]}")
    print(f"Removed IDs from the dataset: {dataset['id'][~stable]}")

    # Sort out all samples that broke during the initial simulation
    successful = tree_util.tree_map(
        lambda x: onp.asarray(x[stable]), (trajstates, dataset, quants)
    )
    failed = tree_util.tree_map(
        lambda x: onp.asarray(x[~stable]), (trajstates, dataset, quants)
    )

    return successful, failed


def init_optimizer(config, dataset):

    num_samples = 1
    if 'U' in dataset['training']:
        num_samples = dataset['training']['U'].shape[0]
    elif 'dF' in dataset['training']:
        num_samples = dataset['training']['dF'].shape[0]
    else:
        exit()

    transition_steps = int(
        config["optimizer"]["epochs"] * num_samples
    ) // config["optimizer"]["batch"]

    lr_schedule_fm = optax.polynomial_schedule(
        config["optimizer"]["init_lr"],
        config["optimizer"]["lr_decay"] * config["optimizer"]["init_lr"],
        config["optimizer"]["power"],
        transition_steps,
    )
    # lr_schedule_fm = optax.exponential_decay(
    #     config["optimizer"]["init_lr"],
    #     transition_steps,
    #     config["optimizer"]["lr_decay"] * config["optimizer"]["init_lr"],
    # )
    # lr_schedule_fm = optax.polynomial_schedule(
    #     config["optimizer"]["init_lr"],
    #     config["optimizer"]["lr_decay"] * config["optimizer"]["init_lr"],
    #     config["optimizer"]["power"],
    #     transition_steps,
    # )


    if config["optimizer"]["type"] == "ADAM":
        opt_transform = optax.scale_by_adam(
            b1=config["optimizer"]["optimizer_kwargs"]["b1"],
            b2=config["optimizer"]["optimizer_kwargs"]["b2"],
            eps=config["optimizer"]["optimizer_kwargs"]["eps"],
            eps_root=config["optimizer"]["optimizer_kwargs"]["eps"] ** 0.5,
            nesterov=True,
        )
    elif config["optimizer"]["type"] == "AdaBelief":
        opt_transform = optax.scale_by_belief(
            b1=config["optimizer"]["optimizer_kwargs"]["b1"],
            b2=config["optimizer"]["optimizer_kwargs"]["b2"],
            eps=config["optimizer"]["optimizer_kwargs"]["eps"],
            eps_root=config["optimizer"]["optimizer_kwargs"]["eps"] ** 0.5,
        )
    else:
        raise NotImplementedError(f"Optimizer {config['optimizer']['type']} not implemented.")

    weight_decay = config["optimizer"].get("weight_decay")
    if weight_decay is None:
        print(f"No weight decay")
        weight_decay = optax.identity()
    else:
        weight_decay = optax.transforms.add_decayed_weights(weight_decay)
    optimizer_fm = optax.chain(
        opt_transform,
        weight_decay,
        optax.scale_by_learning_rate(lr_schedule_fm, flip_sign=True),
    )

    return optimizer_fm


def create_out_dir(config, tag=None, log_mlflow=False):
    now = datetime.datetime.now()
    if tag is not None:
        tag = f"_{tag}"
    else:
        tag = ""

    model = config["model"].get("type", "NequIP")
    name = f"SPICE{tag}_{model}_{now.year}_{now.month}_{now.day}_{uuid.uuid4()}"

    out_dir = Path("output") / name
    out_dir.mkdir(exist_ok=False, parents=True)

    # Save the config values
    with open(out_dir / "config.toml", "wb") as f:
        tomli_w.dump(config, f)

    if not log_mlflow: return out_dir

    client = mlflow.MlflowClient()
    model = client.create_registered_model(name)
    model_version = client.create_model_version(name, source="")
    run_id = model_version.run_id

    return out_dir, run_id


def save_training_results(config, out_dir, trainer: ForceMatching, run_id=None):
    # Save the config values
    with open(out_dir / "config.toml", "wb") as f:
        tomli_w.dump(config, f)

    # Save all the outputs
    trainer.save_energy_params(out_dir / "best_params.pkl", ".pkl", best=True)
    trainer.save_energy_params(out_dir / "final_params.pkl", ".pkl", best=False)
    trainer.save_trainer(out_dir / "trainer.pkl", ".pkl")

    if run_id is None: return

    try:
        mlflow.log_artifact(out_dir / "training.log", "", run_id)
        mlflow.log_artifact(out_dir / "trainer.pkl", "", run_id)
        mlflow.log_artifact(out_dir / "best_params.pkl", "", run_id)
        mlflow.log_artifact(out_dir / "final_params.pkl", "", run_id)
        mlflow.log_dict(config, "", run_id)
    except Exception:
        print(f"Could not connect to MLFlow server.")



def save_resolv_training(config, out_dir, trainer: Difftre):
    # Save the config values
    with open(out_dir / "config.toml", "wb") as f:
        tomli_w.dump(config, f)

    # Save all the outputs
    trainer.save_energy_params(out_dir / "final_params.pkl", ".pkl", best=False)
    trainer.save_trainer(out_dir / "trainer.pkl", ".pkl")


def save_predictions(out_dir, name, predictions, run_id=None):
    predictions = tree_util.tree_map(
        onp.asarray, predictions
    )

    onp.savez(out_dir / f"{name}.npz", **predictions)


def plot_convergence(trainer, out_dir, run_id=None):
    fig, ax1 = plt.subplots(1, 1, figsize=(5, 5),
                                        layout="constrained")

    ax1.set_title("Loss")
    ax1.semilogy(trainer.train_losses, label="Training")
    ax1.semilogy(trainer.val_losses, label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    fig.savefig(out_dir / f"convergence.pdf", bbox_inches="tight")

    if run_id is None: return

    try:
        mlflow.log_artifact(out_dir / "convergence.pdf", "", run_id)
    except:
        print(f"Could not connect to MLFlow server.")


def plot_resolv_convergence(trainer: Difftre, out_dir):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(5, 5), layout="constrained", sharex=True)

    batches = onp.arange(len(trainer.batch_losses)) / len(trainer.batch_losses) * len(trainer.epoch_losses) + 0.5
    epochs = onp.arange(len(trainer.epoch_losses)) + 1

    ax1.semilogy(batches, trainer.batch_losses, label="Batch")
    ax1.semilogy(epochs, trainer.epoch_losses, label="Epoch")

    ax2.semilogy(epochs, trainer.gradient_norm_history)
    ax2.xaxis.set_major_formatter('{x:.0f}')

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("Gradient norm")

    fig.savefig(out_dir / f"convergence.pdf", bbox_inches="tight")


def difference_learning(energy_fn_template, energy_params):
    """Instead of updating the parameters directly, we update the difference to the initial parameters."""

    init_params = tree_util.tree_map(jnp.zeros_like, energy_params)

    def absolute_params(params):
        return tree_util.tree_map(jnp.add, params, energy_params)

    @functools.wraps(energy_fn_template)
    def wrapped_energy_fn_template_wrapper(params):
        if params is None:
            # I.e., the prior model
            return energy_fn_template(energy_params)
        else:
            return energy_fn_template(absolute_params(params))

    return wrapped_energy_fn_template_wrapper, absolute_params, init_params


class ConnectivityChecker:
    """Helper class that reverts the last update if molecules broke."""

    def __init__(self):
        self.trainer_state = None
        self.trajectory_states = None

    @property
    def prepare_task(self):
        def fn(trainer, *args, **kwargs):
            self._prepare_task(trainer, *args, **kwargs)
        return fn

    @property
    def check_task(self):
        def fn(trainer, *args, **kwargs):
            self._check_task(trainer, *args, **kwargs)
        return fn

    def _prepare_task(self, trainer: Difftre, *args, **kwargs):
        self.trainer_state = trainer.state
        self.trajectory_states = trainer.traj_states

    def _check_task(self, trainer: Difftre, batch, *args, **kwargs):
        trajstates = chem_util.tree_take(trainer.traj_states, batch, on_cpu=False)
        statepoints = chem_util.tree_take(trainer.statepoints, batch, on_cpu=False)

        @jax.jit
        @jax.vmap
        def _not_connected(trajstate, statepoint):
            last_neighbor_list = trajstate.sim_state.nbrs
            mask = statepoint["mask"]
            return ~custom_partition.check_connectivity(last_neighbor_list, mask)

        if onp.any(_not_connected(trajstates, statepoints)):
            warnings.warn(
                f"Broken molecules detected during validation. "
                f"Reverting last update."
            )

            # Revert the optimizer update and the broken simulations
            trainer.state = self.trainer_state
            trainer.traj_states = self.trajectory_states
        else:
            print(f"[Connectivity Checker] No broken molecules detected.")


def plot_predictions(predictions, reference_data, subsets, out_dir, name, run_id=None):
    # Simplifies comparison to reported values
    scale_energy = 96.485  # [eV] -> [kJ/mol]
    scale_pos = 0.1  # [Å] -> [nm]

    cmap = plt.get_cmap('tab20')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5),
                                        layout="constrained")

    fig.suptitle("Predictions")
    pred_u_per_a = predictions['U'] / onp.sum(reference_data['mask'], axis=1) / scale_energy
    ref_u_per_a = reference_data['U'] / onp.sum(reference_data['mask'], axis=1) / scale_energy

    mae = onp.mean(onp.abs(pred_u_per_a - ref_u_per_a))
    print(f'Energy MEA: {mae * 1000:.1f} meV/atom')
    ax1.set_title(f"Energy (MAE: {mae * 1000:.1f} meV/atom)")
    ax1.set_prop_cycle(cycler(color=plt.get_cmap('tab20c').colors))
    for subset, label in subsets.items():
        mask = reference_data['subset'] == subset
        ax1.plot(ref_u_per_a[mask] , pred_u_per_a[mask], "*", label=label)
    ax1.set_xlabel("Ref. U [eV/atom]")
    ax1.set_ylabel("Pred. U [eV/atom]")

    # Select only the atoms that are not masked
    subs = onp.tile(reference_data['subset'], (1, *predictions['F'].shape[1:]))
    subs = subs.reshape((-1, 3))[reference_data['mask'].ravel(), :]
    pred_F = predictions['F'].reshape((-1, 3))[reference_data['mask'].ravel(), :] / scale_energy * scale_pos
    ref_F = reference_data['F'].reshape((-1, 3))[reference_data['mask'].ravel(), :] / scale_energy * scale_pos

    mae = onp.mean(onp.abs(pred_F - ref_F))
    ax2.set_title(f"Force (MAE: {mae * 1000:.1f} meV/A)")
    print(f'Force MEA: {mae * 1000:.1f} meV/A')
    ax2.set_prop_cycle(cycler(color=plt.get_cmap('tab20c').colors))
    for subset, label in subsets.items():
        mask = subs == subset
        ax2.plot(ref_F[mask].ravel(), pred_F[mask].ravel(), "*", label=label)
    ax2.set_xlabel("Ref. F [eV/A]")
    ax2.set_ylabel("Pred. F [eV/A]")
    ax2.legend(loc="lower right", prop={'size': 5})

    fig.savefig(out_dir / f"{name}.tiff", bbox_inches="tight")

    if run_id is None: return

    mlflow.log_artifact(out_dir / f"{name}.tiff", f"", run_id)
