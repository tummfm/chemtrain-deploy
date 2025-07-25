import warnings
from pathlib import Path
import uuid
import datetime

import tomli_w

import numpy as onp

import functools

import jax
from jax import tree_util, random

import jax.numpy as jnp

import matplotlib.pyplot as plt

from jax_md import space
from jax_md import simulate

from jax_md_mod import custom_quantity, custom_simulate
from chemtrain import quantity

import optax

from chemtrain.trainers import ForceMatching, Difftre
from chemtrain.ensemble import sampling
from jax_md_mod import custom_partition

from chemutils.models.nequip import nequip_neighborlist_pp
from chemutils.models.mace import mace_neighborlist_pp
from chemutils.models.allegro import allegro_neighborlist_pp
from chemutils.models.painn import painn_neighborlist_pp

# from chemutils.models import import sys
from jax_md_mod.model import neural_networks
from chemutils.visualize import molecule


def define_model(config,
                 dataset=None,
                 nbrs_init=None,
                 max_edges=None,
                 per_particle=False,
                 avg_num_neighbors=1.0,
                 positive_species=False,
                 displacement_fn=None,
                 max_triplets=None):
    """Initializes a concrete model for a system given path to model parameters."""

    if displacement_fn is None:
        fractional = True
        displacement_fn, _ = space.periodic_general(
            jnp.asarray(dataset['training']['box'][0]),
            fractional_coordinates=fractional)

    n_species = 2

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
    
    elif model_type == "DimeNetPP":
        init_fn, gnn_energy_fn = neural_networks.dimenetpp_neighborlist(
            displacement_fn, config["model"]["r_cutoff"], n_species=n_species,
            max_edges=max_edges, max_triplets=max_triplets,
            n_global=1, n_local=0,
            **config["model"]["model_kwargs"]
        )

    else:
        raise NotImplementedError(f"Model {model_type} not implemented.")


    def energy_fn_template(energy_params):
        def energy_fn(pos, neighbor, mode=None, **dynamic_kwargs):
            # assert 'species' in dynamic_kwargs.keys(), 'species not in dynamic_kwargs'

            if "mask" not in dynamic_kwargs:
                print(f"Add defaul all-positive mask.")
                dynamic_kwargs["mask"] = jnp.ones(pos.shape[0], dtype=jnp.bool_)

            if "box" in dynamic_kwargs:
                print(f"Found box in energy kwargs")

            if per_particle:
                num_atoms = jnp.sum(dynamic_kwargs["mask"])
            else:
                num_atoms = 1.0 # return total energy
            if config["model"]["type"] == "DimeNetPP":
                energy = gnn_energy_fn(
                    energy_params, pos, neighbor, **dynamic_kwargs
                ).squeeze() # test with squeeze
                return energy
            else:
                return gnn_energy_fn(
                    energy_params, pos, neighbor, **dynamic_kwargs
                )

        return energy_fn

    if dataset is None:
        return energy_fn_template

    r_init = jnp.asarray(dataset['R'][0])
    species_init = jnp.asarray(dataset['species'][0])
    box_init = jnp.asarray(dataset['box'][0])

    nbrs_init = nbrs_init.update(r_init, box=box_init)

    key = random.PRNGKey(11)

    try:
        top = molecule.topology_from_neighbor_list(nbrs_init, species_init)
        fig = molecule.plot_molecule(r_init, top)
        fig.savefig("molecule.pdf", bbox_inches="tight")

    except NotImplementedError:
        print("Could not plot molecule")

    init_params = init_fn(
        key, r_init, nbrs_init, box=box_init, species=species_init,
    )
    print(f"Initial energy is {jax.jit(energy_fn_template(init_params))(r_init, nbrs_init, species=species_init)}")

    return energy_fn_template, init_params


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
        senders, receivers = sample["bonds"]
        init_mass = jnp.asarray(sample["mass"])
        init_species = jnp.asarray(sample["species"])

        heavy_sender = (init_species[senders] > 1) & (senders < init_mass.size)
        light_receiver = (init_species[receivers] == 1) & (receivers < init_mass.size)
        send_mass = jnp.float_(heavy_sender & light_receiver)

        if mass_repartitioning:
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
        pad = jax.device_count() - remainder
        sharded_args = tree_util.tree_map(
            lambda x: jnp.concatenate([x, x[:pad]], axis=0), sharded_args)

    sharded_args = jax.device_put(sharded_args, sharding)
    jax.debug.visualize_array_sharding(sharded_args[1]["mask"])
    trajstates, (clusters, nclusters), quants = init_ref_state(*sharded_args)
    stable = (nclusters == 1)

    if remainder > 0:
        pad = jax.device_count() - remainder
        trajstates, stable, quants = tree_util.tree_map(
            lambda x: x[:-pad], (trajstates, stable, quants)
        )

    print(f"Removed {onp.sum(~stable)} of {stable.size} entries from the dataset.")
    print(f"Remaining IDs in the dataset: {dataset['id'][stable]}")
    print(f"Removed IDs from the dataset: {dataset['id'][~stable]}")

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
        2.0,
        transition_steps,
    )

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


def create_out_dir(config):
    now = datetime.datetime.now()

    model = config["model"].get("type", "NequIP")
    tag = config["tag"]
    if tag is None:
        tag = uuid.uuid4()
    name = f"water_{model}_r_cutoff_{config['model']['r_cutoff']}_{now.year}_{now.month}_{now.day}_{tag}"

    out_dir = Path("output") / name
    out_dir.mkdir(exist_ok=False, parents=True)

    # Save the config values
    with open(out_dir / "config.toml", "wb") as f:
        tomli_w.dump(config, f)

    return out_dir


def save_training_results(config, out_dir, trainer: ForceMatching):
    # Save the config values
    with open(out_dir / "config.toml", "wb") as f:
        tomli_w.dump(config, f)

    # Save all the outputs
    trainer.save_energy_params(out_dir / "best_params.pkl", ".pkl", best=True)
    trainer.save_energy_params(out_dir / "final_params.pkl", ".pkl", best=False)
    trainer.save_trainer(out_dir / "trainer.pkl", ".pkl")


def save_resolv_training(config, out_dir, trainer: Difftre):
    # Save the config values
    with open(out_dir / "config.toml", "wb") as f:
        tomli_w.dump(config, f)

    # Save all the outputs
    trainer.save_energy_params(out_dir / "final_params.pkl", ".pkl", best=False)
    trainer.save_trainer(out_dir / "trainer.pkl", ".pkl")


def save_predictions(out_dir, name, predictions):
    predictions = tree_util.tree_map(
        onp.asarray, predictions
    )

    onp.savez(out_dir / f"{name}.npz", **predictions)


def plot_convergence(trainer, out_dir):
    fig, ax1 = plt.subplots(1, 1, figsize=(5, 5),
                                        layout="constrained")

    ax1.set_title("Loss")
    ax1.semilogy(trainer.train_losses, label="Training")
    ax1.semilogy(trainer.val_losses, label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    fig.savefig(out_dir / f"convergence.pdf", bbox_inches="tight")


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

    def _check_task(self, trainer: Difftre, batch, *args, **kwargs):
        broken_molecules = False
        for sim_key in batch:
            try:
                last_neighbor_list = trainer.trajectory_states[sim_key].sim_state.nbrs
                last_position = trainer.trajectory_states[sim_key].sim_state.sim_state.position
            except AttributeError:
                *_, ref_state = trainer.trajectory_states[sim_key].args
                last_neighbor_list = ref_state.nbrs
                last_position = ref_state.sim_state.position

            mask = trainer.state_dicts[sim_key]["mask"]
            is_connected = custom_partition.check_connectivity(
                last_neighbor_list, mask
            )
            if not is_connected:
                print(f"[Validation] Neighbor list of statepoint {sim_key} is ->not<- connected.")
                broken_molecules = True
                warnings.warn(f"Neighbor list for {sim_key} is not connected.")
            else:
                print(f"[Validation] Neighbor list of statepoint {sim_key} is connected.")

            # top = molecule.topology_from_neighbor_list(last_neighbor_list, mask)
            # fig = molecule.plot_molecule(init_pos, top)
            # fig.savefig(out_dir / f"molecule_{sim_key}.pdf", bbox_inches="tight")
            # plt.close(fig)

        if broken_molecules:
            warnings.warn(
                f"Broken molecules detected during validation. "
                f"Reverting last update."
            )
            trainer.state = self.trainer_state

def plot_predictions(predictions, reference_data, out_dir, name):

    scale_energy = 96.4853722 # [eV] -> [kJ/mol]
    scale_pos = 0.1  # [Å] -> [nm]
    
    num_atoms = reference_data["F"].shape[1]
    pred_U_per_atom = predictions["U"] / num_atoms
    ref_U_per_atom = reference_data["U"] / num_atoms # not sure if I should compare num_atoms

    # energies
    mae_U = onp.mean(onp.abs(pred_U_per_atom - ref_U_per_atom) / scale_energy)
    min_U = min(min(pred_U_per_atom), min(ref_U_per_atom)) / scale_energy
    max_U = max(max(pred_U_per_atom), max(ref_U_per_atom)) / scale_energy
    line_U = onp.linspace(min_U, max_U)

    # forces
    mae_F = onp.mean(onp.abs(predictions['F'] - reference_data[
        'F'])) / scale_energy * scale_pos
    min_F = min(min(predictions["F"][::50].ravel()), min(reference_data["F"][::50].ravel())) / scale_energy * scale_pos
    max_F = max(max(predictions["F"][::50].ravel()), max(reference_data["F"][::50].ravel())) / scale_energy * scale_pos
    line_F = onp.linspace(min_F, max_F)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5),
                                        layout="constrained")

    fig.suptitle("Predictions")
    
    ax1.set_title(f"Energy (MAE: {mae_U * 1000:.1f} meV/atom)")
    ax1.plot(ref_U_per_atom / scale_energy,
             pred_U_per_atom / scale_energy, "*")    
    ax1.plot(line_U, line_U, "k--")
    ax1.set_xlabel("Ref. U [eV/atom]")
    ax1.set_ylabel("Pred. U [eV/atom]")

    ax2.set_title(f"Force (MAE: {mae_F * 1000:.1f} meV/A)")
    ax2.plot(reference_data['F'][::50].ravel() / scale_energy * scale_pos,
             predictions['F'][::50].ravel() / scale_energy * scale_pos,
             "*")
    ax2.plot(line_F, line_F, "k--")
    ax2.set_xlabel("Ref. F [eV/A]")
    ax2.set_ylabel("Pred. F [eV/A]")

    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
