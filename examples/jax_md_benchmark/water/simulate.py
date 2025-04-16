import os
import sys

import argparse
import time

if len(sys.argv) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

# os.environ["CUDA_VISIBLE_DEVICES"] = "1" # set your cuda device here instead od passing as argument

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

import jax
# jax.config.update("jax_debug_nans", True)

import jax
from chemtrain.deploy import exporter, graphs as export_graphs

from jax_md import partition, space, simulate
from jax_md_mod.model import neural_networks
from jax import numpy as jnp, tree_util
import numpy as onp

from collections import OrderedDict

from chemtrain.data import graphs
from chemtrain import trainers

from chemutils.datasets import water

import train_utils

from ase.io.lammpsdata import read_lammps_data
import pickle



def get_default_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("device", type=str, default="-1")
    parser.add_argument("tag", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=128) # original 64
    args = parser.parse_args()

    print(f"Run on device {args.device}")
    return OrderedDict(
        tag=args.tag,
        model=OrderedDict(
            r_cutoff=0.50, # dimenet train 0.25
            edge_multiplier=1.25,
            # type="Allegro",
            # model_kwargs=OrderedDict(
            #     hidden_irreps="32x0e + 16x1e + 16x1o + 8x2e + 8x2o",
            #     max_ell=3, # original 2
            #     num_layers=2, # original 1
            #     mlp_n_hidden=64,
            #     embed_n_hidden=(8, 16, 32),
            # ),
            type="MACE",
            model_kwargs=OrderedDict(
                hidden_irreps="32x0e + 32x1o",
                embed_dim=64,
                readout_mlp_irreps="16x0e",
                max_ell=2,
                num_interactions=2,
                correlation=3,
            ),
            # type="DimeNetPP",
            # model_kwargs=OrderedDict(
            #     embed_size=64, # origianl 128
            #     n_interaction_blocks=2, # original 3
            #     out_embed_size=192, # original 192
            #     num_rbf=6,
            #     num_sbf=7,
            #     num_residual_before_skip=1,
            #     num_residual_after_skip=2,
            #     num_dense_out=3,
            # ),
            # type="PaiNN",
            # model_kwargs=OrderedDict(
            #     # hidden_size=128,
            #     hidden_size=192,
            #     n_layers=4,
            # ),
        ),
        optimizer=OrderedDict(
            init_lr=1e-4, # original 1e-3
            # init_lr=1e-2,
            # lr_decay=5e-2,
            lr_decay=1e-05,
            epochs=args.epochs,
            batch=args.batch,
            cache=25,
            # weight_decay=-1e-2,
            type="ADAM",
            optimizer_kwargs=OrderedDict(
                b1=0.9,
                b2=0.99,
                eps=1e-8
            )
        ),
        gammas=OrderedDict(
            U=1e-6,
            F=1e-2,
        ),
    )


def init_count_edges_fn(displacement_fn, r_cutoff):
    metric = jax.vmap(space.metric(displacement_fn))

    def count(position, neighbor):
        senders, receivers = neighbor.idx
        dists = metric(position[senders], position[receivers])
        return jnp.sum(dists < r_cutoff)

    return count

def init_simulator(step_fn, count_edges_fn, neighbor_fn: partition.NeighborListFunctions, steps_to_printout, printout_steps, state_kwargs):

    def run_step(state, _):
        sim_state, nbrs = state
        nbrs = neighbor_fn.update(sim_state.position, neighbor=nbrs)
        sim_state = step_fn(sim_state, nbrs, **state_kwargs)

        # Count number of valid edges

        return (sim_state, nbrs), count_edges_fn(sim_state.position, nbrs)

    def run_printout(state, _):
        state, edges = jax.lax.scan(run_step, state, jnp.arange(steps_to_printout))
        sim_state, _ = state

        return state, (sim_state, jnp.max(edges))

    @jax.jit
    def run(state):
        state, (traj, edges) = jax.lax.scan(run_printout, state, jnp.arange(printout_steps))

        return state, traj, jnp.max(edges)

    return run


def main():

    config = get_default_config()
    out_dir = train_utils.create_out_dir(config)

    atoms = read_lammps_data("final_config_filtered_13056_atoms.data",
                             style='atomic')

    positions = atoms.get_positions() / 10  # convert to nm
    box = atoms.cell
    species = atoms.get_atomic_numbers()
    box_lengths = jnp.array([box[0, 0], box[1, 1], box[2, 2]]) / 10
    R = jnp.array(positions) / box_lengths[None, :]
    species = jnp.array(species) - 1

    masses = jnp.zeros_like(species, dtype=jnp.float32)
    masses = jnp.where(species == 0, 15.9994, masses)
    masses = jnp.where(species == 1, 1.008, masses)

    init_sample = {
        "R": R,
        "box": box_lengths,
        "species": species,
        "mass": masses,
    }

    dataset = {key: jnp.expand_dims(value, axis=0) for key, value in init_sample.items()}

    displacement_fn, shift_fn = space.periodic_general(
        box_lengths, fractional_coordinates=True)

    _, (max_neighbors, max_edges, avg_num_neighbors) = graphs.allocate_neighborlist(
        dataset, displacement_fn, None, config["model"]["r_cutoff"], box_key="box", mask_key=None,
        format=partition.Sparse, capacity_multiplier=config["model"]["edge_multiplier"],
    )

    neighbor_fn = partition.neighbor_list(
        displacement_fn, init_sample["box"], config["model"]["r_cutoff"], dr_threshold=0.0, # TODO: Set threshold with config file
        disable_cell_list=False, fractional_coordinates=True,
        format=partition.Sparse,
    )

    max_edges = int(max_edges * config["model"]["edge_multiplier"])
    nbrs_init = neighbor_fn.allocate(dataset["R"], capacity_multiplier=config["model"]["edge_multiplier"])

    print(f"Neighbors: {nbrs_init}")
    print(f"Max neighbors: {max_neighbors}, max edges: {max_edges}")

    # Positive species not required -> check again
    energy_fn_template, init_params = train_utils.define_model(
        config, dataset, nbrs_init, max_edges, per_particle=False, # per_particle=False for trainig
        avg_num_neighbors=avg_num_neighbors, positive_species=False,
        displacement_fn=displacement_fn
    )
    energy_params = onp.load(
        "best_params.pkl",
        allow_pickle=True,
    )

    energy_params = tree_util.tree_map(
        jnp.asarray, energy_params
    )
    energy_fn = energy_fn_template(energy_params)

    # TODO: Initialize the simulator from above

    count_edge_fn = init_count_edges_fn(displacement_fn, config["model"]["r_cutoff"])


    init_fn, step_fn = simulate.nvt_nose_hoover(...) # TODO: Initialize
    simulator_fn = init_simulator(step_fn, count_edge_fn, neighbor_fn, steps_to_printout, printout_steps, state_kwargs={"species": init_sample["species"]}) # TODO: Set timings

    init_state = (
        init_fn(init_sample["R"], species=init_sample["species"], mass=init_sample["mass"]), nbrs_init
    )

    t_start = time.time()
    (_, final_nbrs), traj, sim_max_edges = simulator_fn(init_state)
    t_end = time.time() - t_start
    print(f"Simulation time: {t_end / 60} min")


    # TODO: Check final neighbor list error code
    final_nbrs.error

    # TODO: Check whether simulations max edges remained below actual max edges
    assert sim_max_edges <= max_edges, "Simulation exceeded max edges"

if __name__ == "__main__":
    main()



