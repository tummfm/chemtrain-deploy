import os
import sys

import argparse

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
# jax.config.update("jax_debug_nans", True)

import jax
from chemtrain.deploy import exporter, graphs as export_graphs

from jax_md import partition, space

from collections import OrderedDict

from chemtrain.data import graphs
from chemtrain import trainers

from chemutils.datasets import aluminum

import train_utils

_setup_script = """

units           metal
atom_style      atomic
boundary        p p p

region          box block 0 {bx} 0 {by} 0 {bz} 
create_box      1 box

mass 1 1.0

neighbor 2.5 bin
neigh_modify every 10 delay 0 check yes

"""

_evaluate_script = """
pair_style      meam 
pair_coeff      * * ../models/library.meam Al ../models/Al.meam Al

compute 1 all pe
compute 2 all property/atom fx fy fz

run 1
"""


import lammps


def get_default_config():
    parser = argparse.ArgumentParser()
    # parser.add_argument("tag", type=str, default=None)
    args = parser.parse_args()

    dataset = "ANI"

    scale_energy = 1.0 # 96.485  # eV -> kJ/mol
    scale_pos = 1.0 # 0.1  # A -> nm
    fractional = True
    flip_forces = False
    per_atom = True
    shift_U = 0.0

    match dataset:
        case "ANI":
            scale_energy *= 27.2114079527  # Ha -> eV
            flip_forces = True  # why flip forces?

    return OrderedDict(
        # tag=args.tag,
        dataset=dataset,
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

import matplotlib.pyplot as plt

import numpy as onp
import jax.numpy as jnp

def main():

    config = get_default_config()

    # out_dir = train_utils.create_out_dir(config)
    #
    dataset = aluminum.get_dataset("/home/ga27pej/Datasets", config)

    is_diag = jax.vmap(
        lambda b: jnp.all(jnp.diag(jnp.diag(b)) == b),
    )

    assert onp.sum(~is_diag(dataset["testing"]["box"][0])) == 0, "Boxes are not orthogonal"

    pots = []
    forces = []
    ref_forces = []
    for idx in range(dataset["testing"]["R"].shape[0]):
        sample = jax.tree.map(
            lambda s: s[idx], dataset["testing"]
        )

        bx, by, bz = onp.diag(sample["box"])

        lmp = lammps.lammps()
        lmp.commands_string(_setup_script.format(
            bx=bx, by=by, bz=bz
        ))

        n_atoms = onp.sum(sample["mask"])
        pos = onp.einsum("ij,nj->ni", sample["box"], sample["R"][:n_atoms, :])

        print(f"Creating {n_atoms} atoms")
        lmp.create_atoms(
            n_atoms, None, onp.ones(n_atoms, dtype=onp.int32).tolist(),
            pos.ravel().tolist(),
        )

        lmp.commands_string(_evaluate_script)

        # LAMMPS reorders the particles
        ids = lmp.numpy.extract_atom("id", lammps.LMP_TYPE_SCALAR)
        order = onp.argsort(ids)

        pots.append(lmp.extract_compute("1", lammps.LMP_STYLE_GLOBAL, lammps.LMP_TYPE_SCALAR))
        forces += lmp.numpy.extract_compute("2", lammps.LMP_STYLE_ATOM, lammps.LMP_TYPE_ARRAY)[order,:].ravel().tolist()
        ref_forces += sample["F"][:n_atoms, :].ravel().tolist()

        lmp.close()

    # Compute the per-particle errors and remove the mean energy shift
    pots = onp.asarray(pots) / onp.sum(dataset["testing"]["mask"], axis=-1)
    pred_pots = pots - onp.mean(pots)
    ref_pots = dataset["testing"]["U"] - onp.mean(dataset["testing"]["U"])

    ppa_mae = onp.mean(onp.abs(ref_pots - pred_pots))
    force_mae = onp.mean(onp.abs(onp.asarray(forces) - onp.asarray(ref_forces)))

    print(f"MAE Energy Error [eV/atom]: {ppa_mae}")
    print(f"MAE Forces Error [eV/A]: {force_mae}")

    plt.figure()
    plt.plot(ref_pots, pred_pots, ".")
    plt.title(f"Potential MAE: {ppa_mae * 1000:.1f} meV/atom")
    plt.xlabel("Reference Energy [eV/atom]")
    plt.ylabel("Predicted Energy [eV/atom]")
    plt.savefig("predictions_U.png")

    plt.figure()
    plt.plot(ref_forces, forces, ".")
    plt.title(f"Force MAE: {force_mae * 1000:.1f} meV/A")
    plt.xlabel("Reference Forces [eV/A]")
    plt.ylabel("Predicted Forces [eV/A]")
    plt.savefig("predictions_F.png")


if __name__ == "__main__":
    main()
