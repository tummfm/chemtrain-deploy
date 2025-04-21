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
from jax_md_mod import custom_space

from collections import OrderedDict

from chemtrain.data import graphs

import train_utils

from ase.io.lammpsdata import read_lammps_data

def get_default_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("device", type=str, default="-1")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    print(f"Run on device {args.device}")
    return OrderedDict(
        model=OrderedDict(
            seed=args.seed,
            r_cutoff=0.50,
            edge_multiplier=1.5,
            # type="MACE",
            # model_kwargs=OrderedDict(
            #     hidden_irreps="32x0e + 32x1o",
            #     # hidden_irreps="128x0e + 128x1o",
            #     embed_dim=64,
            #     readout_mlp_irreps="32x0e",
            #     max_ell=2,
            #     num_interactions=2,
            #     correlation=3,
            # ),

            # type="Allegro",
            # model_kwargs=OrderedDict(
            #     # hidden_irreps="32x0e + 16x1e + 16x1o + 8x2e + 8x2o",
            #     hidden_irreps="32x0e + 32x1e + 16x1o + 8x2e + 8x2o",
            #     # hidden_irreps="64x0e + 32x1e + 32x1o + 16x2e + 16x2o + 8x3e + 8x3o",
            #     num_layers=1,
            #     mlp_n_hidden=128,
            #     embed_n_hidden=(16, 32, 64),
            # ),
            type="PaiNN",
            model_kwargs=OrderedDict(
                hidden_size=192,
                n_layers=4,
            ),
        ),
        gammas=OrderedDict(
            U=1e-6,
            F=1e-2,
        ),
        simulator=OrderedDict(
            dt=0.001,
            T=300,
            dr_threshold=0.1,
            steps_to_printout=1, # original 1000
            printout_steps=100,
        ),
    )


def init_count_edges_fn(displacement_fn, r_cutoff):
    metric = jax.vmap(space.metric(displacement_fn))

    def count(position, neighbor):
        senders, receivers = neighbor.idx
        dists = metric(position[senders], position[receivers])
        return jnp.sum(dists < r_cutoff)

    return count

def init_simulator(step_fn, count_edges_fn, neighbor_fn: partition.NeighborListFns, steps_to_printout, printout_steps, state_kwargs):

    def run_step(state, _):
        sim_state, nbrs = state
        nbrs = neighbor_fn.update(sim_state.position, neighbors=nbrs)
        sim_state = step_fn(sim_state, neighbor=nbrs, **state_kwargs)

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
    key = random.PRNGKey(0)
    key, split = random.split(key)
    config = get_default_config()

    atoms = read_lammps_data("chignolin_solvated_6.lmpdat",
                             style='atomic')

    positions = atoms.get_positions() / 10  # convert to nm
    R = positions
    box = atoms.cell
    species = atoms.get_atomic_numbers()
    box_lengths = jnp.array([box[0, 0], box[1, 1], box[2, 2]]) / 10
    # R = jnp.array(positions) / box_lengths[None, :]
    species = jnp.array(species) - 1

    masses = jnp.zeros_like(species, dtype=jnp.float32)
    masses = jnp.where(species == 0, 1.008, masses)
    masses = jnp.where(species == 1, 1.0, masses)
    masses = jnp.where(species == 2, 6.941, masses)
    masses = jnp.where(species == 3, 1.0, masses)
    masses = jnp.where(species == 4, 10.81, masses)
    masses = jnp.where(species == 5, 12.011, masses)
    masses = jnp.where(species == 6, 14.0067, masses)
    masses = jnp.where(species == 7, 15.999, masses)

    init_sample = {
        "R": R,
        "box": box_lengths,
        "species": species,
        "mass": masses,
    }

    dataset = {key: jnp.expand_dims(value, axis=0) for key, value in init_sample.items()}
    # for key in dataset.keys():
    #     dataset[key].pop("box")

    displacement_fn, shift_fn = custom_space.nonperiodic_general(fractional_coordinates=False)
 
    _, (max_neighbors, max_edges, avg_num_neighbors) = graphs.allocate_neighborlist(
        dataset, displacement_fn, 0.0, config["model"]["r_cutoff"], mask_key=None,
        format=partition.Sparse, capacity_multiplier=config["model"]["edge_multiplier"],
    )

    neighbor_fn = partition.neighbor_list(
        displacement_fn, init_sample["box"], config["model"]["r_cutoff"], dr_threshold=config["simulator"]["dr_threshold"], # TODO: Set threshold with config file
        disable_cell_list=False, fractional_coordinates=False,
        format=partition.Sparse,
    )

    max_edges = int(max_edges * config["model"]["edge_multiplier"])
    nbrs_init = neighbor_fn.allocate(init_sample["R"], capacity_multiplier=config["model"]["edge_multiplier"])

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


    init_fn, step_fn = simulate.nvt_nose_hoover(
        energy_or_force_fn=energy_fn,
        shift_fn=shift_fn,
        dt=config["simulator"]["dt"],
        kT=config["simulator"]["T"] * quantity.kb,
    ) # TODO: Initialize
    
    simulator_fn = init_simulator(step_fn, count_edge_fn, neighbor_fn, config["simulator"]["steps_to_printout"], 
                                  config["simulator"]["printout_steps"], state_kwargs={"species": init_sample["species"]}) # TODO: Set timings


    init_state = (
        init_fn(split, init_sample["R"], neighbor=nbrs_init, species=init_sample["species"], mass=init_sample["mass"]), nbrs_init
    )

    t_start = time.time()
    print(f"Starting simulation...")
    (_, final_nbrs), traj, sim_max_edges = simulator_fn(init_state)
    t_end = time.time() - t_start
    print(f"Simulation time: {t_end / 60} min")

    print(f"Final neighbor list: {final_nbrs.error}")

    # TODO: Check whether simulations max edges remained below actual max edges
    print(f"Simulation max edges: {sim_max_edges}, max edges: {max_edges}")
    assert sim_max_edges <= max_edges, "Simulation exceeded max edges"

if __name__ == "__main__":
    main()



