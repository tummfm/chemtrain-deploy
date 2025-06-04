import os
import sys

import argparse
import time

if len(sys.argv) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

import jax
from chemtrain import quantity

from jax_md import partition, simulate
from jax import random
from jax import numpy as jnp, tree_util
import numpy as onp
from jax_md_mod import custom_space

from collections import OrderedDict


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
            edge_multiplier=1.1,
            type="MACE",
            model_kwargs=OrderedDict(
                hidden_irreps="32x0e + 32x1o",
                # hidden_irreps="128x0e + 128x1o",
                embed_dim=64,
                readout_mlp_irreps="32x0e",
                max_ell=2,
                num_interactions=2,
                correlation=3,
            ),

            # type="Allegro",
            # model_kwargs=OrderedDict(
            #     hidden_irreps="64x1o + 16x2e",
            #     mlp_n_hidden=256,
            #     mlp_n_layers=2,
            #     embed_n_hidden=(128, 128, 256),
            #     embed_dim=128,
            #     max_ell=2,
            #     num_layers=3,
            # ),
            # type="PaiNN",
            # model_kwargs=OrderedDict(
            #     hidden_size=128,
            #     n_layers=4,
            # ),
        ),
        gammas=OrderedDict(
            U=1e-6,
            F=1e-2,
        ),
        simulator=OrderedDict(
            dt=500,
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

        return (sim_state, nbrs), overflowed

    def run_printout(state, _):
        state, overflow_flags = jax.lax.scan(run_step, state, jnp.arange(steps_to_printout))
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
    R = positions
    box = atoms.cell
    species = atoms.get_atomic_numbers()
    box_lengths = jnp.array([box[0, 0], box[1, 1], box[2, 2]]) / 10
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


    displacement_fn, shift_fn = custom_space.nonperiodic_general(fractional_coordinates=False)

    neighbor_fn = partition.neighbor_list(
        displacement_fn, init_sample["box"], config["model"]["r_cutoff"], dr_threshold=config["simulator"]["dr_threshold"], # TODO: Set threshold with config file
        disable_cell_list=False, fractional_coordinates=False,
        format=partition.Sparse,capacity_multiplier=config["model"]["edge_multiplier"],
    )

    nbrs_init = neighbor_fn.allocate(init_sample["R"], capacity_multiplier=config["model"]["edge_multiplier"])
    energy_fn_template, init_params = train_utils.define_model(
        config, dataset, nbrs_init, nbrs_init.idx.shape[1], per_particle=False, # per_particle=False for trainig
        positive_species=False,
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

    init_fn, step_fn = simulate.nvt_nose_hoover(
        energy_or_force_fn=energy_fn,
        shift_fn=shift_fn,
        dt=config["simulator"]["dt"],
        kT=config["simulator"]["T"] * quantity.kb,
    )
    
    simulator_fn = init_simulator(step_fn, neighbor_fn, config["simulator"]["steps_to_printout"], 
                                  config["simulator"]["printout_steps"], state_kwargs={"species": init_sample["species"]}) # TODO: Set timings


    init_state = (
        init_fn(split, init_sample["R"], neighbor=nbrs_init, species=init_sample["species"], mass=init_sample["mass"]), nbrs_init
    )

    t_start = time.time()
    (_, final_nbrs), _ = simulator_fn(init_state)
    t_end = time.time() - t_start


if __name__ == "__main__":
    main()



