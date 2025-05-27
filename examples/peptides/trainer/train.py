import os
import functools
import sys

import argparse

from pathlib import Path

import tomli_w
from jax.experimental.custom_partitioning import custom_partitioning
from sklearn import linear_model

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


import train_utils


def get_default_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("device", type=str, default="-1")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()

    print(f"Run on device {args.device}")
    return OrderedDict(
        model=OrderedDict(
            r_cutoff=0.5,
            edge_multiplier=1.15,
            type="Allegro",
            model_kwargs=OrderedDict(
                hidden_irreps="64x1o",
                mlp_n_hidden=512,
                mlp_n_layers=2,
                embed_n_hidden=(128, 256, 512),
                embed_dim=128,
                max_ell=1,
                num_layers=3,
            ),
        ),
        optimizer=OrderedDict(
            init_lr=1e-3,
            lr_decay=1e-2,  #
            epochs=args.epochs,
            batch=16, #decreesing batch sizes had significant effect
            cache=100,
            power="exponential",
            weight_decay=1e-3,
            type="ADAM",
            optimizer_kwargs=OrderedDict(
                b1=0.95,
                b2=0.999,
                eps=1e-8,
                # normalize=True,
            )
        ),
        dataset=OrderedDict(
            subsets=[
                "SPICE PubChem Set",  # Regex matching the subset names
                "Amino Acid Ligand",
                "SPICE Dipeptides",
                "DES370",
                "SPICE Solvated",
                "SPICE Water"
            ],
            # total_charge='total_charge', # Use all samples if commented out
            # max_samples=10000 # Use all samples if commented out
        ),
        gammas=OrderedDict(
            U=1e-4,
            F=5e-4,
        ),
    )

def main():

    config = get_default_config()
    out_dir = train_utils.create_out_dir(config)

    dataset, info = spice.download_spice(
        "/home/ga27pej/Datasets",
        subsets=config["dataset"].get("subsets"),
        max_samples=config["dataset"].get("max_samples"),
        fractional=False
    )
    dataset = spice.process_dataset(dataset)

    # Update the loaded subset information
    config["dataset"]["subsets"] = list(info["subsets"].values())
    print(f"Dataset information: {info}")

    for key in dataset.keys():
        dataset[key].pop("box")

    displacement_fn, _ = space.free()
    nbrs_format = partition.Sparse
    # Infer the number of neighbors within the model cutoff
    nbrs_init, (max_neighbors, max_edges, avg_num_neighbors) = graphs.allocate_neighborlist(
        dataset["training"], displacement_fn, 0.0,
        config["model"]["r_cutoff"], mask_key="mask", format=nbrs_format,
    )

    # Since we deal with atomic numbers, we only provide positive species.
    energy_fn_template, init_params = train_utils.define_model(
        config, dataset, nbrs_init, max_edges, per_particle=False,
        avg_num_neighbors=avg_num_neighbors, positive_species=True,
        displacement_fn=displacement_fn
    )

    optimizer = train_utils.init_optimizer(config, dataset)
    trainer_fm = trainers.ForceMatching(
        init_params, optimizer, energy_fn_template,
        nbrs_init,
        batch_per_device=config["optimizer"]["batch"] // len(jax.devices()),
        batch_cache=config["optimizer"]["cache"],
        gammas=config["gammas"],
        weights_keys={
            "F": "F_weight",
        },
        log_file=out_dir / "training.log",
        checkpoint_path=out_dir / "checkpoints"
    )

    # extensions.log_batch_progress(trainer_fm, frequency=100)

    trainer_fm.set_dataset(
        dataset['training'], stage='training')
    trainer_fm.set_dataset(
        dataset['validation'], stage='validation', include_all=True)
    trainer_fm.set_dataset(
        dataset['testing'], stage='testing', include_all=True)

    # Train and save the results to a new folder
    trainer_fm.train(config["optimizer"]["epochs"], checkpoint_freq=10)

    train_utils.save_training_results(config, out_dir, trainer_fm)

    predictions = trainer_fm.predict(
        dataset["validation"], trainer_fm.best_params,
        batch_size=config["optimizer"]["batch"],
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

    class Model(exporter.Exporter):

        graph_type = export_graphs.SimpleSparseNeighborList

        r_cutoff = config["model"]["r_cutoff"] * 10  # Cutoff in angstrom
        if config["model"]["type"] == "Allegro":
            nbr_order = [1, 2]
        if config["model"]["type"] == "MACE":
            nbr_order = [2, 4]
        if config["model"]["type"] == "PaiNN":
            nbr_order = [4, 8]

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

