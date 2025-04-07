import os
import sys

import argparse

if len(sys.argv) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

# os.environ["CUDA_VISIBLE_DEVICES"] = "1" # set your cuda device here instead od passing as argument

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

import jax
# jax.config.update("jax_debug_nans", True)

import jax
from chemtrain.deploy import exporter, graphs as export_graphs

from jax_md import partition, space
from jax_md_mod.model import neural_networks

from collections import OrderedDict

from chemtrain.data import graphs
from chemtrain import trainers

from chemutils.datasets import water

import train_utils



def get_default_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("device", type=str, default="-1")
    parser.add_argument("tag", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=64) # original 64
    args = parser.parse_args()

    print(f"Run on device {args.device}")
    return OrderedDict(
        tag=args.tag,
        model=OrderedDict(
            r_cutoff=0.50, # dimenet train 0.25
            edge_multiplier=1.15,
            # type="Allegro",
            # model_kwargs=OrderedDict(
            #     hidden_irreps="32x0e + 16x1e + 16x1o + 8x2e + 8x2o",
            #     embed_dim=128, # original 32
            #     max_ell=3, # original 2
            #     num_layers=2, # original 1
            #     mlp_n_hidden=64,
            #     embed_n_hidden=(8, 16, 32),
            # ),
            # type="MACE",
            # model_kwargs=OrderedDict(
            #     hidden_irreps="32x0e + 32x1o",
            #     embed_dim=64,
            #     max_ell=2,
            #     num_interactions=2,
            #     correlation=3,
            # ),
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
            type="PaiNN",
            model_kwargs=OrderedDict(
                # hidden_size=128,
                hidden_size=192,
                n_layers=4,
            ),
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

def main():

    config = get_default_config()
    out_dir = train_utils.create_out_dir(config)

    dataset = water.get_dataset("/home/weilong/workspace/chemsim-lammps/datasets")
    dataset = water.get_random_subset(dataset, 1.0, seed=0)
    displacement_fn, _ = space.periodic_general(box=dataset['training']['box'][0], fractional_coordinates=True)
    if config["model"]["type"] == "DimeNetPP":
        nbrs_format = partition.Dense
        nbrs_init, (max_neighbors, max_edges, avg_num_neighbors, max_triplets) = graphs.allocate_neighborlist(
            dataset["training"], displacement_fn, None, config["model"]["r_cutoff"], box_key="box", mask_key=None,
            format=nbrs_format, count_triplets=True
        )
    else:
        nbrs_init, (max_neighbors, max_edges, avg_num_neighbors) = graphs.allocate_neighborlist(
            dataset["training"], displacement_fn, None, config["model"]["r_cutoff"], box_key="box", mask_key=None,
            format=partition.Sparse
        )

    print(f"Neighbors: {nbrs_init}")
    print(f"Max neighbors: {max_neighbors}, max edges: {max_edges}")

    # Positive species not required -> check again
    if config["model"]["type"] == "DimeNetPP":
        energy_fn_template, init_params = train_utils.define_model(
            config, dataset, nbrs_init, max_edges, per_particle=False, # per_particle=False for trainig
            avg_num_neighbors=avg_num_neighbors, positive_species=False,
            displacement_fn=displacement_fn, max_triplets=max_triplets
        ) 
    else:
        energy_fn_template, init_params = train_utils.define_model(
            config, dataset, nbrs_init, max_edges, per_particle=False, # per_particle=False for trainig
            avg_num_neighbors=avg_num_neighbors, positive_species=False,
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
    train_utils.plot_predictions(predictions, dataset["validation"], out_dir, f"preds_validation")
    train_utils.plot_convergence(trainer_fm, out_dir)

    # Directly export the model for later use in LAMMPS

    displacement_fn, _ = space.free()
    if config["model"]["type"] == "DimeNetPP":
        nbrs_format = partition.Dense
        nbrs_init, (max_neighbors, max_edges, avg_num_neighbors, max_triplets) = graphs.allocate_neighborlist(
            dataset["training"], displacement_fn, 0.0, config["model"]["r_cutoff"], mask_key=None,
            format=nbrs_format, fractional_coordinates=False, capacity_multiplier=2.0, count_triplets=True
        )
    else:
        nbrs_init, (max_neighbors, max_edges, avg_num_neighbors) = graphs.allocate_neighborlist(
            dataset["training"], displacement_fn, 0.0, config["model"]["r_cutoff"], mask_key=None,
            format=partition.Sparse, fractional_coordinates=False, capacity_multiplier=2.0
        )
    # export should be different for dimentpp
    if config["model"]["type"] == "DimeNetPP":
        export_template, _ = train_utils.define_model(
            config, dataset, nbrs_init, max_edges, per_particle=True, # per_particle=True for exporting
            avg_num_neighbors=avg_num_neighbors, positive_species=False,
            displacement_fn=displacement_fn, max_triplets=max_triplets
        )
    else:
        export_template, _ = train_utils.define_model(
            config, dataset, nbrs_init, max_edges, per_particle=True, # per_particle=True for exporting
            avg_num_neighbors=avg_num_neighbors, positive_species=False,
            displacement_fn=displacement_fn
        )


    class Model(exporter.Exporter):

        r_cutoff = config["model"]["r_cutoff"] * 10# Cutoff in Angstrom
        graph_type = export_graphs.SimpleSparseNeighborList # this should also be changed
        # nbr_order = [1+mpl,2*(1+mpl)] # this should be the changed
        if config["model"]["type"] == "MACE":
            nbr_order = [3, 6] # MACE
        if config["model"]["type"] == "Allegro":
            nbr_order = [1, 2]
        if config["model"]["type"] == "PaiNN": # change according to the mp layers
            nbr_order = [5, 10]
        mask = False
        unit_style = "metal"
        # num_mpl: int = num_mpl_lookup[config["model"]["type"]]
        displacement = displacement_fn

        def __init__(self, export_template, params, *args, **kwargs):
            self.model = export_template(params)

            super().__init__(*args, **kwargs)

        def energy_fn(self, position, species, graph):

            neighbor = graph.to_neighborlist()
            # We trained the model with units nm and kJ/mol, so we need some scaling
            # input in A -> model uses nm
            position /= 10.0 # A/nm
            energies = self.model(position, neighbor, species=species)
            # model uses kJ/mol -> export in eV
            energies /= 96.485 # (kJ/mol)/eV
            # for export to kcal/mol
            # energies /= 4.184 # (kJ/mol)/(kcal/mol)

            return energies
            
    class Dimenet_Model(exporter.Exporter):

            r_cutoff = config["model"]["r_cutoff"] * 10# Cutoff in Angstrom
            graph_type = export_graphs.SimpleDenseNeighborList
            nbr_order = [3, 6] # different nbr_order should be used for different MP layers
            mask = False
            unit_style = "metal"
            # num_mpl: int = num_mpl_lookup[config["model"]["type"]]
            displacement = displacement_fn

            def __init__(self, params, *args, **kwargs):
                self.params = params

                super().__init__(*args, **kwargs)

            def energy_fn(self, position, species, graph):

                neighbor = graph.to_neighborlist()
                _, apply_fn = neural_networks.dimenetpp_neighborlist(
                    displacement_fn, self.r_cutoff / 10, n_species=2,
                    max_edges=graph.max_edges.size, max_triplets=graph.max_triplets.size,
                    n_global=0, n_local=1, **config["model"]["model_kwargs"]
                )

                # We trained the model with units nm and kJ/mol, so we need some scaling
                # input in A -> model uses nm
                position /= 10.0 # A/nm
                energies = apply_fn(self.params, position, neighbor, species=species)
                energies = energies.squeeze(axis=-1)
                # model uses kJ/mol -> export in eV
                energies /= 96.485 # (kJ/mol)/eV
                # for export to kcal/mol
                # energies /= 4.184 # (kJ/mol)/(kcal/mol)

                return energies
    if config["model"]["type"] == "DimeNetPP":
        trained_model = Dimenet_Model(trainer_fm.best_params)
    else:    
        trained_model = Model(export_template, trainer_fm.best_params)

    trained_model.export()
    trained_model.save(out_dir / "model_default.ptb")


if __name__ == "__main__":
    main()
