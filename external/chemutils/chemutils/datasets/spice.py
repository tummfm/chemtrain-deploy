# MIT License
#
# Copyright (c) 2025 Multiscale Modeling of Fluid Materials
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Downloads and prepares the SPICE dataset."""

from urllib import request
from pathlib import Path

import h5py

import re
import jax
import numpy as onp

from chemtrain.data import preprocessing
from chemutils.datasets import utils

spice_versions: dict[str, str] = {
    'v2.0.1': "https://zenodo.org/records/10975225/files/SPICE-2.0.1.hdf5?download=1"
}

def download_spice(root="./_data",
                   scale_R=0.0529177,
                   scale_U=2625.5,
                   scale_e=11.7871,
                   fractional=True,
                   max_samples=None,
                   version='v2.0.1',
                   subsets=None,
                   **kwargs):
    """Download complete SPICE dataset.

    Args:
        root: Download directory of SPICE dataset.
        scale_R: Scaling factor for atomic positions (default=5.2917721e-2 #Bohr -> nm).
        scale_U: Scaling factor for potential energies (default=2625.5 #Hartee -> kJ/mol).
        fractional: Whether to scale positions by simulation box (default=true).
        url: Url for dataset download (default refers to SPICE v2.0.1).
        subsets: List of regex to select subsets of the dataset

    Returns:
        The complete SPICE dataset.
    """

    with jax.default_device(jax.devices("cpu")[0]):
        data_dir = download_source(spice_versions[version], root=root)

        dataset = load_and_padd_samples(data_dir)
        dataset, subsets = select_subsets(data_dir, dataset, subsets)
        dataset = scale_dataset(dataset, scale_R, scale_U, scale_e, fractional)
        dataset = split_by_subset(dataset, max_samples, **kwargs)


    info = {
        "subsets": subsets,
        "version": spice_versions,
        "scaling": {
            "R": scale_R,
            "U": scale_U,
            "fractional": fractional
        },
    }

    return dataset, info


def download_source(url: str, root: str="./_data"):
    """Downloads and unpacks the SPICE dataset. 'https://zenodo.org/records/8222043' gives an overview on the available
    versions.

    Urls of available versions of the SPICE dataset:
    v1.1.2: 'https://zenodo.org/records/7338495/files/SPICE-1.1.2.hdf5?download=1'
    v1.1.3: 'https://zenodo.org/records/7606550/files/SPICE-1.1.3.hdf5?download=1'
    v1.1.4: 'https://zenodo.org/records/8222043/files/SPICE-1.1.4.hdf5?download=1'
    v2.0.0: 'https://zenodo.org/records/10835749/files/SPICE-2.0.0.hdf5?download=1'
    v2.0.1: 'https://zenodo.org/records/10975225/files/SPICE-2.0.1.hdf5?download=1'

    Args:
        url: Url for dataset download.
        root: Download directory of SPICE dataset.

    Returns:
        Path of the download directory.
    """
    data_dir = Path(root)
    data_dir.mkdir(exist_ok=True, parents=True)

    if not (data_dir / "SPICE").exists():
        (data_dir / "SPICE").mkdir()
        print(f"Download SPICE dataset from {url}")
        request.urlretrieve(url, data_dir / "SPICE/SPICE.hdf5", utils.show_progress)

    return data_dir / "SPICE"


def load_and_padd_samples(data_dir):
    """Loads and padds the atom data."""

    # Do not process more than once
    if (data_dir / "SPICE.npz").exists():
        return dict(onp.load(data_dir / "SPICE.npz"))

    with h5py.File(data_dir / "SPICE.hdf5", "r") as file:
        subsets = []
        for mol in file.keys():
            current_subset = file[mol]["subset"][0]
            if not current_subset in subsets:
                print(f"Discovered new subset: {current_subset}")
                subsets.append(current_subset)

        max_atoms = max([file[mol]["atomic_numbers"].size for mol in file.keys()])
        n_samples = sum([file[mol]["conformations"].shape[0] for mol in file.keys()])

        mols = list(file.keys())
        mols.sort()

        print(f"Found {len(file.keys())} molecules with a maximum of {max_atoms} atoms and {n_samples} samples.")

        # Reserve memory for the complete padded dataset
        dataset = {
            "id": onp.zeros((n_samples,), dtype=int),
            "R": onp.zeros((n_samples, max_atoms, 3)),
            "F": onp.zeros((n_samples, max_atoms, 3)),
            "U": onp.zeros((n_samples,)),
            "charge": onp.zeros((n_samples, max_atoms), dtype=float),
            "dipole": onp.zeros((n_samples, 3), dtype=float),
            "subset": onp.zeros((n_samples,), dtype=int),
            "species": onp.zeros((n_samples, max_atoms), dtype=int),
            "mask": onp.zeros((n_samples, max_atoms), dtype=bool),
        }

        idx = 0
        for id, mol in enumerate(mols):
            confs = file[mol]
            conf_shape = confs["conformations"].shape
            if onp.size(conf_shape)==3:
                n_samples, n_atoms, _ = conf_shape
            else:
                # skip damaged molecule (no conformations, forces, energies, ...)
                continue

            dataset["id"][idx:idx + n_samples] = onp.broadcast_to(id, (n_samples,))
            dataset["subset"][idx:idx + n_samples] = onp.broadcast_to(subsets.index(confs["subset"][0]), (n_samples,))
            dataset["mask"][idx:idx + n_samples] = onp.broadcast_to(onp.arange(max_atoms) < n_atoms, (n_samples, max_atoms))
            dataset["species"][idx:idx + n_samples, :n_atoms] = onp.broadcast_to(onp.asarray(confs["atomic_numbers"], dtype=int), (n_samples, n_atoms))

            dataset["R"][idx:idx + n_samples, :n_atoms, :] = onp.asarray(confs["conformations"])
            dataset["F"][idx:idx + n_samples, :n_atoms, :] = -1.0 * onp.asarray(confs["dft_total_gradient"])
            dataset["U"][idx:idx + n_samples] = onp.asarray(confs["formation_energy"])

            if "mbis_charges" in confs.keys():
                dataset["charge"][idx:idx + n_samples, :n_atoms] = onp.asarray(confs["mbis_charges"], dtype=float).squeeze()
            else:
                dataset["charge"][idx:idx + n_samples, :n_atoms] = onp.nan

            if "scf_dipole" in confs.keys():
                dataset["dipole"][idx:idx + n_samples] = onp.asarray(confs["scf_dipole"], dtype=float).squeeze()
            else:
                print(f"[SPICE] No dipole data for {mol}.")
                dataset["dipole"][idx:idx + n_samples] = onp.nan

            idx += n_samples

    # save dataset
    onp.savez(data_dir / "SPICE.npz", **dataset)
    with open(data_dir / "subsets.dat", "w") as file:
        file.writelines([f"{str(idx).rjust(3, '0')} {subset.decode('ascii')}\n"
                         for idx, subset in enumerate(subsets)])

    return dataset


def split_by_subset(dataset, max_samples=None, **kwargs):
    """Splits the loaded subsets individually."""

    # Find out which subsets were loaded
    total_samples = dataset["subset"].size
    subsets = onp.unique(dataset["subset"])
    keys = dataset.keys()
    split_dataset = {
        "training": [], "validation": [], "testing": []
    }


    for subset in subsets:
        sub_dataset = {key: arr[dataset["subset"] == subset] for key, arr in dataset.items()}
        sub_train, sub_val, sub_test = preprocessing.train_val_test_split(sub_dataset, **kwargs, shuffle=True, shuffle_seed=11)

        # Select a maximum number of samples relative to the total number of samples
        if max_samples is not None:
            max_subset = int(sub_dataset["subset"].size / total_samples * max_samples)
            sub_train = {key: arr[:max_subset] for key, arr in sub_train.items()}
            sub_val = {key: arr[:max_subset] for key, arr in sub_val.items()}
            sub_test = {key: arr[:max_subset] for key, arr in sub_test.items()}

        split_dataset["training"].append(sub_train)
        split_dataset["validation"].append(sub_val)
        split_dataset["testing"].append(sub_test)


    # Concatenate the subsets
    final_dataset = {}
    for split, split_data in split_dataset.items():
        final_dataset[split] = {
            key: onp.concatenate([sub[key] for sub in split_data], axis=0)
            for key in keys
        }

    return final_dataset


def process_dataset(dataset):
    """Creates weights for masked loss."""

    for split in dataset.keys():
        # Weight the potential by the number of particles. The per-particle
        # potential should have equal weight, so the error for larger systems
        # should contribute more. On the other hand, since we compute the
        # MSE for the forces, we have to correct for systems with masked
        # out particles.

        n_particles = onp.sum(dataset[split]['mask'], axis=1)
        max_particles = dataset[split]['mask'].shape[1]

        dataset[split]['U_weight'] = onp.ones_like(n_particles)
        dataset[split]['F_weight'] = max_particles / n_particles

        # Remove nan_charges
        is_nan = onp.any(onp.isnan(dataset[split]['charge']), axis=-1)
        dataset[split]['charge'][is_nan, :] = 0.0

        dataset[split]['total_charge'] = onp.sum(dataset[split]['charge'] * dataset[split]['mask'], axis=-1)
        print(f"Total charges: {dataset[split]['total_charge']}")
        print(f"Total charges are in range {dataset[split]['total_charge'].min()} to {dataset[split]['total_charge'].max()}")

        dataset[split]['charge_weight'] = max_particles / n_particles
        dataset[split]['charge_weight'] *= ~is_nan / onp.mean(~is_nan, keepdims=True)

        # Remove nan dipoles
        is_nan = onp.any(onp.isnan(dataset[split]['dipole']), axis=-1)
        dataset[split]['dipole'][is_nan, :] = 0.0
        print(f"Dipoles: {dataset[split]['dipole']}")

        dataset[split]['dipole_weight'] = max_particles / n_particles
        dataset[split]['dipole_weight'] *= ~is_nan / onp.mean(~is_nan, keepdims=True)

    return dataset


def select_subsets(data_dir, dataset, subsets):
    """Selects the subsets from the dataset."""
    if subsets is None:
        return dataset

    with open(data_dir / "subsets.dat", "r") as file:
        selection: dict[int, str] = {
            int(line.partition(" ")[0]): line.partition(" ")[2]
            for line in file.readlines()
            if any([re.search(s, line) for s in subsets])
        }

    boolean_mask = onp.isin(dataset["subset"], list(selection.keys()))
    dataset = {
        key: arr[boolean_mask] for key, arr in dataset.items()
    }

    return dataset, {idx: sel.strip("\n") for idx, sel in selection.items()}


def scale_dataset(dataset, scale_R, scale_U, scale_e, fractional=True):
    """Scales the dataset from Hartee to kJ/mol and Bohr to nm."""

    box = 10 * (dataset["R"].max() - dataset["R"].min())

    if fractional:
        dataset['R'] = dataset['R'] / box
    else:
        dataset['R'] = dataset['R'] * scale_R

    print(f"Scale dataset by {scale_R} for R and {scale_U} for U.")

    scale_F = scale_U / scale_R
    dataset['box'] = scale_R * onp.tile(box * onp.eye(3), (dataset['R'].shape[0], 1, 1))
    dataset['U'] *= scale_U
    dataset['F'] *= scale_F
    dataset['charge'] *= scale_e
    dataset['dipole'] *= scale_e * scale_R

    return dataset
