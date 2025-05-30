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
from chemtrain import quantity

from jax_md import partition, space, simulate

from jax import random
from jax import numpy as jnp, tree_util
import numpy as onp

from collections import OrderedDict

from chemtrain.data import graphs

import train_utils

from ase.io.lammpsdata import read_lammps_data

def get_default_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("device", type=str, default="-1")
    args = parser.parse_args()

    print(f"Run on device {args.device}")
    return OrderedDict(
        model=OrderedDict(
            r_cutoff=0.50, # dimenet train 0.25
            edge_multiplier=1.1,
            type="Allegro",
            model_kwargs=OrderedDict(
                hidden_irreps="32x0e + 16x1e + 16x1o + 8x2e + 8x2o",
                # hidden_irreps="32x0e + 32x1e + 16x1o + 8x2e + 8x2o",
                max_ell=3, # original 2
                num_layers=1, # original 1
                mlp_n_hidden=64,
                embed_n_hidden=(8, 16, 32),
            ),
            # type="MACE",
            # model_kwargs=OrderedDict(
            #     hidden_irreps="32x0e + 32x1o",
            #     embed_dim=64,
            #     max_ell=2,
            #     num_interactions=2,
            #     correlation=3,
            # ),
            # type="PaiNN",
            # model_kwargs=OrderedDict(
            #     hidden_size=128,
            #     # hidden_size=192,
            #     n_layers=4,
            # ),
        ),
        simulator=OrderedDict(
            dt=0.001,
            T=300,
            dr_threshold=0,
            steps_to_printout=1, # original 1000
            printout_steps=250,
        ),
    )


def init_simulator(step_fn, neighbor_fn: partition.NeighborListFns, steps_to_printout, printout_steps, state_kwargs):

    def run_step(state, _):
        sim_state, nbrs = state
        nbrs = neighbor_fn.update(sim_state.position, neighbors=nbrs)
        sim_state = step_fn(sim_state, neighbor=nbrs, **state_kwargs)

        overflowed = nbrs.did_buffer_overflow

        # Count number of valid edges

        return (sim_state, nbrs), overflowed

    def run_printout(state, _):
        state, overflow_flags = jax.lax.scan(run_step, state, jnp.arange(steps_to_printout))
        # sim_state, _ = state

        return state, overflow_flags

    @jax.jit
    def run(state):
        state, overflow_summary = jax.lax.scan(run_printout, state, jnp.arange(printout_steps))

        return state, overflow_summary

    return run

def main():
    key = random.PRNGKey(0)
    key, split = random.split(key)
    config = get_default_config()

    atoms = read_lammps_data(f"replicated_initial_config_{config['model']['type'].lower()}.data",
                             style='atomic')

    positions = atoms.get_positions() / 10  # convert to nm
    box = atoms.cell
    print(f"Box: {box}")
    species = atoms.get_atomic_numbers()
    box_lengths = jnp.array([box[0, 0], box[1, 1], box[2, 2]]) / 10
    print(f"Box lengths: {box_lengths}") 
    R = jnp.array(positions) / box_lengths[None, :]
    species = jnp.array(species) - 1

    masses = jnp.zeros_like(species, dtype=jnp.float32)
    masses = jnp.where(species == 0, 15.9994, masses)
    masses = jnp.where(species == 1, 1.008, masses)

    init_sample = {
        "R": R,
        "box": jnp.diag(box_lengths),
        "species": species,
        "mass": masses,
    }

    dataset = {key: jnp.expand_dims(value, axis=0) for key, value in init_sample.items()}

    displacement_fn, shift_fn = space.periodic_general(
        box_lengths, fractional_coordinates=True)

    # _, (max_neighbors, max_edges, avg_num_neighbors) = graphs.allocate_neighborlist(
    #     dataset, displacement_fn, None, config["model"]["r_cutoff"], box_key="box", mask_key=None,
    #     format=partition.Sparse, capacity_multiplier=config["model"]["edge_multiplier"],
    # )

    neighbor_fn = partition.neighbor_list(
        displacement_fn, init_sample["box"], config["model"]["r_cutoff"], dr_threshold=config["simulator"]["dr_threshold"], # TODO: Set threshold with config file
        disable_cell_list=False, fractional_coordinates=True,
        format=partition.Sparse, capacity_multiplier=config["model"]["edge_multiplier"],
    )

    nbrs_init = neighbor_fn.allocate(init_sample["R"], capacity_multiplier=config["model"]["edge_multiplier"])

    print(f"Neighbors: {nbrs_init}")
    print(f"Max edges: {nbrs_init.idx.shape[1]}")
    # Positive species not required -> check again
    energy_fn_template, init_params = train_utils.define_model(
        config, dataset, nbrs_init, nbrs_init.idx.shape[1], per_particle=False, positive_species=False,
        displacement_fn=displacement_fn
    )

    energy_params = onp.load(
        f"best_params_{config['model']['type'].lower()}.pkl",
        allow_pickle=True,
    )

    energy_params = tree_util.tree_map(
        jnp.asarray, energy_params
    )
    energy_fn = energy_fn_template(energy_params)

    # TODO: Initialize the simulator from above

    # count_edge_fn = init_count_edges_fn(displacement_fn, config["model"]["r_cutoff"])


    init_fn, step_fn = simulate.nvt_nose_hoover(
        energy_or_force_fn=energy_fn,
        shift_fn=shift_fn,
        dt=config["simulator"]["dt"],
        kT=config["simulator"]["T"] * quantity.kb,
    ) # TODO: Initialize
    
    simulator_fn = init_simulator(step_fn, neighbor_fn, config["simulator"]["steps_to_printout"], 
                                  config["simulator"]["printout_steps"], state_kwargs={"species": init_sample["species"]}) # TODO: Set timings


    init_state = (
        init_fn(split, init_sample["R"], neighbor=nbrs_init, species=init_sample["species"], mass=init_sample["mass"]), nbrs_init
    )

    t_start = time.time()
    # (_, final_nbrs), traj, sim_max_edges = simulator_fn(init_state)
    (_, final_nbrs), _ = simulator_fn(init_state)
    t_end = time.time() - t_start
    print(f"Simulation time: {t_end / 60} min")
    print(f"Final neighbor list: {final_nbrs.error}")
    print(f"Final neighbor list: {final_nbrs}")
    # TODO: Check final neighbor list error code

    # TODO: Check whether simulations max edges remained below actual max edges
    # print(f"Simulation max edges: {sim_max_edges}, max edges: {max_edges}")
    # assert sim_max_edges <= max_edges, "Simulation exceeded max edges"

if __name__ == "__main__":
    main()



