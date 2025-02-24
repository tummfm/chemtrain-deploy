import os
import functools
import sys

import argparse

from pathlib import Path

#import mlflow
import tomli_w
from jax.experimental.custom_partitioning import custom_partitioning

if len(sys.argv) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]


os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

import jax
# jax.config.update("jax_debug_nans", True)

import numpy as onp

import jax
from jax import tree_util, lax, random
from chemtrain.deploy import exporter, graphs as export_graphs

import jax.numpy as jnp

from jax.sharding import PartitionSpec as P

from jax_md_mod import io, custom_quantity, custom_space, custom_energy, custom_partition
from jax_md import simulate, partition, space, util, energy, quantity as snapshot_quantity
from jax.experimental import mesh_utils

from jax_md_mod.model import layers, neural_networks, prior

import mdtraj

import optax

from collections import OrderedDict


import matplotlib.pyplot as plt

import haiku as hk
import chex
import copy
import contextlib

from chemtrain.data import preprocessing, graphs
from chemtrain.ensemble import sampling
from chemtrain import quantity, trainers, util as chem_util
from chemtrain.trainers import ForceMatching, extensions
from chemtrain.quantity import property_prediction

import e3nn_jax

from chemutils.datasets import spice
from chemutils.datasets import utils as data_utils

import train_utils


def get_default_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("device", type=str, default="-1")
    parser.add_argument("tag", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    print(f"Run on device {args.device}")
    return OrderedDict(
        tag=args.tag,
        seed=args.seed,
        model=OrderedDict(
            r_cutoff=0.5,
            edge_multiplier=1.15,
            type="MACE",
            model_kwargs=OrderedDict(
                hidden_irreps="32x0e + 32x1o",
                # hidden_irreps="128x0e + 128x1o",
                embed_dim=64,
                max_ell=2,
                num_interactions=2,
                correlation=3,
            ),

            # type="Allegro",
            # model_kwargs=OrderedDict(
            #     # hidden_irreps="32x0e + 16x1e + 16x1o + 8x2e + 8x2o + 4x3e + 4x3o",
            #     hidden_irreps="32x0e + 16x1e + 16x1o + 8x2e + 8x2o + 4x3e + 4x3o",
            #     embed_dim=128,
            #     max_ell=3,
            #     num_layers=2,
            # ),
        ),
        optimizer=OrderedDict(
            init_lr=1e-2,
            lr_decay=1e-2,
            epochs=args.epochs,
            batch=args.batch,
            cache=25,
            weight_decay=1e-4,
            type="ADAM",
            optimizer_kwargs=OrderedDict(
                b1=0.9,
                b2=0.99,
                eps=1e-8
            )
        ),
        dataset=OrderedDict(
            subsets=[
                "SPICE PubChem Set",  # Regex matching the subset names
                "Amino Acid Ligand",
                "SPICE Dipeptides",
                "DES370K"
            ],
            # subsets=[".*"],
            max_samples=10000 # Use all samples if commented out
        ),
        gammas=OrderedDict(
            U=1e-6,
            F=1e-2,
        ),
    )

def main():

    config = get_default_config()
    out_dir = train_utils.create_out_dir(config, log_mlflow=False)

    dataset, info = spice.download_spice(
        "/home/paul/Datasets",
        subsets=config["dataset"].get("subsets"),
        max_samples=config["dataset"].get("max_samples"),
        fractional=False
    )
    dataset = spice.process_dataset(dataset)

    # Update the loaded subset information
    config["dataset"]["subsets"] = list(info["subsets"].values())
    print(f"Dataset information: {info}")

    # for split in dataset.keys():
    #     # Test the sign of the forces
    #     dataset[split]["F"] *= -1.0

    # We dont need a box since we are using nonperiodic space
    for key in dataset.keys():
        dataset[key].pop("box")

    displacement_fn, _ = custom_space.nonperiodic_general(fractional_coordinates=False)
    nbrs_init, (max_neighbors, max_edges, avg_num_neighbors) = graphs.allocate_neighborlist(
        dataset["training"], displacement_fn, 0.0, 0.5, mask_key="mask",
        format=partition.Sparse,
    )

    print(f"Neighbors: {nbrs_init}")
    print(f"Max neighbors: {max_neighbors}, max edges: {max_edges}")

    # Since we deal with atomic numbers, we only provide positive species.
    energy_fn_template, init_params = train_utils.define_model(
        config, dataset, nbrs_init, max_edges, per_particle=False,
        avg_num_neighbors=avg_num_neighbors, positive_species=True,
        displacement_fn=displacement_fn
    )

    optimizer = train_utils.init_optimizer(config, dataset)

    trainer_fm = trainers.ForceMatching(
        init_params, optimizer, energy_fn_template, nbrs_init,
        batch_per_device=config["optimizer"]["batch"] // len(jax.devices()),
        batch_cache=config["optimizer"]["cache"],
        gammas=config["gammas"],
        weights_keys={
            "U": "U_weight",
            "F": "F_weight",
        },
        log_file = out_dir / "training.log"
    )

    # TODO: Check Gate and Activation of MACE model

    # extensions.log_batch_progress(trainer_fm, frequency=100)

    trainer_fm.set_dataset(
        dataset['training'], stage='training')
    trainer_fm.set_dataset(
        dataset['validation'], stage='validation', include_all=True)
    trainer_fm.set_dataset(
        dataset['testing'], stage='testing', include_all=True)

    # Train and save the results to a new folder
    trainer_fm.train(config["optimizer"]["epochs"])

    train_utils.save_training_results(config, out_dir, trainer_fm)

    predictions = trainer_fm.predict(
        dataset["validation"], trainer_fm.best_params, batch_size=config["optimizer"]["batch"],
    )

    train_utils.save_predictions(out_dir, f"preds_validation", predictions)
    train_utils.plot_predictions(
        predictions, dataset["validation"], info["subsets"], out_dir,
        f"preds_validation")
    train_utils.plot_convergence(trainer_fm, out_dir)

    # Directly export the model for later use in LAMMPS

    export_template, _ = train_utils.define_model(
        config, dataset, nbrs_init, max_edges, per_particle=True,
        avg_num_neighbors=avg_num_neighbors, positive_species=False,
        displacement_fn=displacement_fn
    )

    num_mpl_lookup = {
        "NequIP": 4,
        "MACE": 2,
        "Allegro": 0
    }

    class Model(exporter.Exporter):

        graph_type = export_graphs.SimpleSparseNeighborList

        num_mpl: int = num_mpl_lookup[config["model"]["type"]]

        r_cutoff = 5.0 # Cutoff in angstrom

        def __init__(self, export_template, params, *args, **kwargs):
            self.model = export_template(params)

            super().__init__(*args, **kwargs)

        def energy_fn(self, position, species, graph):
            neighbor = graph.to_neighborlist()
            # We trained the model with units nm and kJ/mol, so we need some scaling
            position /= 10.0
            energies = self.model(position, neighbor, species=species)
            energies /= 4.184

            return energies

    trained_model = Model(export_template, trainer_fm.best_params)
    trained_model.export()

    trained_model.save(out_dir / "model.ptb")


if __name__ == "__main__":
    main()

