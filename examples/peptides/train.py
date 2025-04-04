import os
import sys
import argparse
if len(sys.argv) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

import jax
# jax.config.update("jax_debug_nans", True)
import jax
from chemtrain.deploy import exporter, graphs as export_graphs
from jax_md_mod import custom_space
from jax_md import partition
from collections import OrderedDict
from chemtrain.data import graphs
from chemtrain import trainers
from chemutils.datasets import spice
import train_utils


def get_default_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("device", type=str, default="-1")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    print(f"Run on device {args.device}")
    return OrderedDict(
        seed=args.seed,
        model=OrderedDict(
            r_cutoff=0.5,
            edge_multiplier=1.15,
            # type="MACE",
            # model_kwargs=OrderedDict(
            #     hidden_irreps="32x0e + 32x1o",
            #     # hidden_irreps="128x0e + 128x1o",
            #     embed_dim=64,
            #     max_ell=2,
            #     num_interactions=2,
            #     correlation=3,
            # ),

            type="Allegro",
            model_kwargs=OrderedDict(
                # hidden_irreps="32x0e + 16x1e + 16x1o + 8x2e + 8x2o + 4x3e + 4x3o",
                hidden_irreps="32x0e + 16x1e + 16x1o + 8x2e + 8x2o + 4x3e + 4x3o",
                # hidden_irreps="64x0e + 32x1e + 32x1o + 16x2e + 16x2o + 8x3e + 8x3o",
                embed_dim=64, # 128
                num_layers=2,
                mlp_n_hidden=64,
                # embed_n_hidden=(8, 16, 32),
            ),
            # type="PaiNN",
            # model_kwargs=OrderedDict(
            #     hidden_size=192,
            #     n_layers=4,
            # ),
        ),
        optimizer=OrderedDict(
            # init_lr=8.00E-03, # mace and allegro
            init_lr=1.00E-04, # painn
            lr_decay=1.00E-5,
            epochs=args.epochs,
            batch=args.batch,
            cache=50,
            power=0.33,
            # weight_decay=1e-4,
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
                "DES370K",
                "SPICE Solvated",
                "SPICE Water"
            ],
            # subsets=[".*"],
            # max_samples=10000 # Use all samples if commented out
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
        dataset["training"], displacement_fn, 0.0, config["model"]["r_cutoff"], mask_key="mask",
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


    class Model(exporter.Exporter):

        graph_type = export_graphs.SimpleSparseNeighborList

        r_cutoff = config["model"]["r_cutoff"] * 10 # Cutoff in angstrom
        if config["model"]["type"] == "Allegro":
            nbr_order = [1, 2]
        if config["model"]["type"] == "MACE":
            nbr_order = [3, 6]
        if config["model"]["type"] == "PaiNN":
            nbr_order = [5, 10]
    
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

