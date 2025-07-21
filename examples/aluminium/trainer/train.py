import os
import sys

import argparse

if len(sys.argv) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

import jax

import jax
from chemtrain.deploy import exporter, graphs as export_graphs

from jax_md import partition, space

from collections import OrderedDict

from chemtrain.data import graphs
from chemtrain import trainers

from chemutils.datasets import aluminum

import train_utils



def get_default_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("device", type=str, default="-1")
    parser.add_argument("tag", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    dataset = "ANI"
    
    scale_energy = 96.485 # eV -> kJ/mol
    scale_pos = 0.1 # A -> nm
    fractional = True
    flip_forces = False
    per_atom = False
    shift_U = 0.0

    match dataset:
        case "ANI":
            scale_energy *= 27.2114079527 # Ha -> eV
            flip_forces = True

    print(f"Run on device {args.device}")

    return OrderedDict(
        tag=args.tag,
        dataset=dataset,
        model=OrderedDict(
            r_cutoff=0.5,
            edge_multiplier=1.15,
            # type="Allegro",
            # model_kwargs=OrderedDict(
            #     hidden_irreps="32x0e + 16x1e + 16x1o + 8x2e + 8x2o",
            #     max_ell=3,
            #     num_layers=2,
            #     mlp_n_hidden=64,
            #     embed_n_hidden=(8, 16, 32),
            # ),
            # type="MACE",
            # model_kwargs=OrderedDict(
            #     hidden_irreps="32x0e + 32x1o",
            #     embed_dim=64,
            #     readout_mlp_irreps="16x0e",
            #     max_ell=2,
            #     num_interactions=2,
            #     correlation=3,
            # ),
            type="PaiNN",
            model_kwargs=OrderedDict(
                # hidden_size=256, 
                hidden_size=196,
                n_layers=4,
            ),
        ),
        optimizer=OrderedDict(
            init_lr=1e-3,
            lr_decay=1e-05,
            epochs=args.epochs,
            batch=args.batch,
            cache=25,
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
        scaling=OrderedDict(
            scale_energy=scale_energy,
            scale_pos=scale_pos
        ),
        processing=OrderedDict(
            fractional=fractional,
            flip_forces=flip_forces,
            per_atom=per_atom,
            shift_U=shift_U
        ),
    )


def main():

    config = get_default_config()
    out_dir = train_utils.create_out_dir(config)

    dataset = aluminum.get_dataset("", config)

    displacement_fn, _ = space.periodic_general(box=dataset['training']['box'][0], fractional_coordinates=True)

    nbrs_init, (max_neighbors, max_edges, avg_num_neighbors) = graphs.allocate_neighborlist(
        dataset["training"], displacement_fn, None, config["model"]["r_cutoff"], box_key="box", mask_key="mask",
        format=partition.Sparse,
    )

    energy_fn_template, init_params = train_utils.define_model(
        config, dataset, nbrs_init, max_edges, per_particle=False,
        avg_num_neighbors=avg_num_neighbors,
        displacement_fn=displacement_fn
    )

    optimizer = train_utils.init_optimizer(config, dataset)

    trainer_fm = trainers.ForceMatching(
        init_params, optimizer, energy_fn_template, nbrs_init,
        batch_per_device=config["optimizer"]["batch"] // len(jax.devices()),
        batch_cache=config["optimizer"]["cache"],
        gammas=config["gammas"],
        weights_keys={
            "F": "F_weight",
        },
        log_file = out_dir / "training.log"
    )

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
    train_utils.plot_predictions(predictions, dataset["validation"], out_dir, f"preds_validation", config["processing"])
    train_utils.plot_convergence(trainer_fm, out_dir)


    displacement_fn, _ = space.free()
    nbrs_init, (max_neighbors, max_edges, avg_num_neighbors) = graphs.allocate_neighborlist(
        dataset["training"], displacement_fn, 0.0, config["model"]["r_cutoff"], mask_key="mask",
        format=partition.Sparse, fractional_coordinates=False, capacity_multiplier=2.0
    )
    export_template, _ = train_utils.define_model(
        config, dataset, nbrs_init, max_edges, per_particle=True, # per_particle=True for exporting
        avg_num_neighbors=avg_num_neighbors, positive_species=False,
        displacement_fn=displacement_fn
    )

    class Model(exporter.Exporter):

        r_cutoff = config["model"]["r_cutoff"] * 10 # Cutoff in Angstrom

        graph_type = export_graphs.SimpleSparseNeighborList
        if config["model"]["type"] == "Allegro":
            nbr_order = [1, 2]
        if config["model"]["type"] == "MACE":
            nbr_order = [2, 4]
        if config["model"]["type"] == "PaiNN":
            nbr_order = [4, 8]
        mask = False
        unit_style = "metal"

        displacement = displacement_fn

        def __init__(self, export_template, params, *args, **kwargs):
            self.model = export_template(params)

            super().__init__(*args, **kwargs)

        def energy_fn(self, position, species, graph):

            neighbor = graph.to_neighborlist()
            position /= 10.0 # A/nm
            energies = self.model(position, neighbor, species=species)
            energies /= 96.485

            return energies

    trained_model = Model(export_template, trainer_fm.best_params)
    trained_model.export()

    trained_model.save(out_dir / "model_default.ptb")


if __name__ == "__main__":
    main()
