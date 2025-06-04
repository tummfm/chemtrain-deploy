import os
import functools
import sys

import argparse

from pathlib import Path
from jax.experimental.custom_partitioning import custom_partitioning

if len(sys.argv) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

import jax
from chemtrain.deploy import exporter, graphs as export_graphs


from jax_md import partition, space
from collections import OrderedDict
from chemtrain.data import graphs
from chemtrain import trainers
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
            batch=16,
            cache=100,
            power="exponential",
            weight_decay=1e-3,
            type="ADAM",
            optimizer_kwargs=OrderedDict(
                b1=0.90,
                b2=0.999,
                eps=1e-8,
            )
        ),
        dataset=OrderedDict(
            subsets=[
                "SPICE PubChem Set",
                "Amino Acid Ligand",
                "SPICE Dipeptides",
                "DES370",
                "SPICE Solvated",
                "SPICE Water"
            ],
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
        "",
        subsets=config["dataset"].get("subsets"),
        max_samples=config["dataset"].get("max_samples"),
        fractional=False
    )
    dataset = spice.process_dataset(dataset)

    config["dataset"]["subsets"] = list(info["subsets"].values())
    print(f"Dataset information: {info}")

    for key in dataset.keys():
        dataset[key].pop("box")

    displacement_fn, _ = space.free()
    nbrs_format = partition.Sparse

    nbrs_init, (max_neighbors, max_edges, avg_num_neighbors) = graphs.allocate_neighborlist(
        dataset["training"], displacement_fn, 0.0,
        config["model"]["r_cutoff"], mask_key="mask", format=nbrs_format,
    )

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

    trainer_fm.set_dataset(
        dataset['training'], stage='training')
    trainer_fm.set_dataset(
        dataset['validation'], stage='validation', include_all=True)
    trainer_fm.set_dataset(
        dataset['testing'], stage='testing', include_all=True)

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
            position /= 10.0
            energies = self.model(position, neighbor, species=species)
            energies /= 4.184

            return energies

    trained_model = Model(export_template, trainer_fm.best_params)
    trained_model.export()

    trained_model.save(out_dir / "model.ptb")


if __name__ == "__main__":
    main()

