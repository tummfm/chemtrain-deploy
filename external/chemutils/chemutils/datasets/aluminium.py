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

"""Tools to download and process the ANI-Al dataset."""

import tarfile
import zipfile

from urllib import request
from pathlib import Path

import numpy as onp
import os

from . import utils as data_utils

if __name__ == "__main__":
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax

from jax import numpy as jnp, tree_util

from jax_md_mod import custom_space
from jax_md_mod.model import sparse_graph
from jax_md import quantity as snapshot_quantity, space, partition

from chemtrain.data import preprocessing


def load_subset(data_dir: str, train_ratio: float = 0.7, val_ratio: float = 0.1,
                shuffle: bool = True) -> tuple:
    """
    Loads a subset of the data from disk.

    Args:
        data_dir: Path to the directory containing the data.
        train_ratio: Fraction of data to use as training set.
        val_ratio: Fraction of data to use as validation set.
        shuffle: True if data should be shuffeled when splitting into subsets

    Returns:
        splits: Tuple containing training, validation, and testing splits of the subset.
    """
    # Load all quantities
    box = onp.load(data_dir / 'box.npy', allow_pickle=True)
    coord = onp.load(data_dir / 'coord.npy', allow_pickle=True)
    energy = onp.load(data_dir / 'energy.npy', allow_pickle=True)
    force = onp.load(data_dir / 'force.npy', allow_pickle=True)
    mask = onp.load(data_dir / "mask.npy", allow_pickle=True)

    # Get number of samples from number of box entires
    n_samples = box.shape[0]

    # Reshape the data to a standard format
    # Transpose box tensor to conform to JAX-MD format
    dataset = dict(
        box=onp.reshape(box, (n_samples, 3, 3)).swapaxes(1, 2),
        # TODO: Check if swapaxes necessary
        R=onp.reshape(coord, (n_samples, -1, 3)),
        U=onp.reshape(energy, (n_samples,)),
        F=onp.reshape(force, (n_samples, -1, 3)),
        mask=onp.reshape(mask, (n_samples, -1))
    )

    # Shuffel to get randomly distributed data and avoid systematic differences in subsets
    splits = preprocessing.train_val_test_split(
        dataset, train_ratio=train_ratio, val_ratio=val_ratio, shuffle=shuffle)

    return splits


def get_train_val_test_set(dir_files: list[str]) -> dict:
    """
    Loads multiple datasets and combines them to a large one.

    Args:
        dir_files: List of paths to data-directories.

    Returns:
        dataset: Dictionary containing training, validation, and testing splits of the dataset.
    """
    # Initialize arrays to store the data
    dataset = dict(
        training=dict(box=[], R=[], U=[], F=[], mask=[]),
        validation=dict(box=[], R=[], U=[], F=[], mask=[]),
        testing=dict(box=[], R=[], U=[], F=[], mask=[])
    )

    # Load the data from all provided files
    for i in range(len(dir_files)):
        train_split, val_split, test_split = load_subset(dir_files[i])

        for k in dataset['training'].keys():
            dataset['training'][k].append(train_split[k])
            dataset['validation'][k].append(val_split[k])
            dataset['testing'][k].append(test_split[k])

    # Concatenate to single arrays
    for split in dataset.keys():
        for quantity in dataset[split].keys():
            dataset[split][quantity] = onp.concatenate(dataset[split][quantity],
                                                       axis=0)

    return dataset


def scale_dataset(
    dataset: dict,
    scale_U: float = 1.0,
    scale_R: float = 1.0,
    fractional: bool = True,
    shift_U: float = 0.0,
    flip_forces: bool = False,
    per_atom: bool = False,
) -> dict:
    """
    Scales dataset by given factors and applies transformations.

    Args:
        dataset: Dictionary containing multiple splits of the full dataset.
            Each split is again a dictionary, containing the keys
            ``["box", "R", "U", "F", "virial", "type"]``.
        scale_U: Unit conversion factor for the energy.
        scale_R: Unit conversion factor for lengths.
        fractional: True if dataset should be converted to fractional coordinates.
        shift_U: Energy (in kJ/mol) will be shifted up by this amount before training and back after.
        flip_forces: True if sign of forces should be flipped.
        per_atom: True if energies should directly be scaled per_atom.

    Returns:
        dataset: Dictionary containing all splits of the data in the correct units.

    """
    # Unit conversion factor for forces
    scale_F = scale_U / scale_R

    # Flip forces if reference uses positive energy gradient instead of negative for force computation
    if flip_forces:
        scale_F *= -1

    # For every split, i.e., training, validation and test set
    for split in dataset.keys():

        # Initialize scaling function for fractional coordinates
        box = dataset[split]['box'][0]
        box, scale_fn = custom_space.init_fractional_coordinates(box)
        vmap_scale_fn = jax.vmap(lambda R, box: scale_fn(R, box=box),
                                 in_axes=(0, 0))

        if fractional:
            # Convert positions to fractional corrdinates and scale
            dataset[split]['R'] = vmap_scale_fn(dataset[split]['R'],
                                                dataset[split]['box'])
        else:
            # Just scale coordinates by factor
            dataset[split]['R'] = dataset[split]['R'] * scale_R

        # Scale all other quantities by corresponding factor
        # NOTE: Scaling of R not necessary for fractional coordinates, as it is relative to the scaled box!
        dataset[split]['box'] *= scale_R
        dataset[split]['U'] *= scale_U
        dataset[split]['F'] *= scale_F

        if per_atom:
            # Convert total energy to per-atom energy
            num_atoms = onp.sum(dataset[split]["mask"], axis=1)
            dataset[split]['U'] /= num_atoms

        # Shift energy by prescribed value
        dataset[split]['U'] += shift_U

    return dataset


def get_ANI(path: str, config: dict) -> dict:
    """
    Fetches the ANI-Al dataset from the given directory and splits it into training, validation and test set.
    Downloads and preprocesses the dataset if it does not exist in the directory.

    Source:
        Smith et al. (2021): 'Automated discovery of a robust interatomic potential for aluminum'
        DOI: https://doi.org/10.1038/s41467-021-21376-0

    Args:
        path: Path to the directory storing the ANI-Al dataset.
        config: Dictionary containing training configurations.

    Returns:
        dataset: Dictionary containing all splits of the ANI-Al dataset.
    """
    # Create directory if it does not exist
    data_dir = Path(path)
    data_dir.mkdir(exist_ok=True)

    # Download and unzip raw data if not present
    if not (data_dir / "ANI_Al_raw").exists():
        print("Downloading ANI Aluminum dataset...")

        # Load data from the link provided in the paper
        url = "https://github.com/atomistic-ml/ani-al/raw/refs/heads/master/data/Al-data.tgz"
        request.urlretrieve(url, data_dir / "ANI_Al_raw.tgz")

        # Unzip data
        tgz_path = data_dir / "ANI_Al_raw.tgz"
        with tarfile.open(tgz_path, "r:gz") as tar_f:
            tar_f.extractall(data_dir / "ANI_Al_raw")

    # Convert raw data to desired format if not present
    if not (data_dir / "ANI_Al").exists():
        print("Converting ANI Aluminum dataset...")

        # Currently no need to expand dataset
        expand = False

        # Convert data to usable format
        data_utils.convert_ANI(data_dir, expand)

    # Load dataset and split into training, validation and test set
    dataset = get_train_val_test_set([data_dir / "ANI_Al"])

    # Scale dataset acording to factors and processing flags in training config
    dataset = scale_dataset(
        dataset,
        scale_R=config["scaling"]["scale_pos"],
        scale_U=config["scaling"]["scale_energy"],
        fractional=config["processing"]["fractional"],
        shift_U=config["processing"]["shift_U"],
        flip_forces=config["processing"]["flip_forces"],
        per_atom=config["processing"]["per_atom"],
    )

    return dataset


def get_dataset(path: str, config: dict) -> dict:
    """
    Fetches and splits the specified dataset into training, validation and test set.
    Downloads and preprocesses the dataset if it does not exist in the directory.

    Args:
        path: Path to the directory storing all datasets.
        config: Dictionary containing training configurations.

    Returns:
        dataset: Dictionary containing all splits of the dataset.
    """
    # Check if specified dataset is implemented
    name = config["dataset"]
    implemented_datasets = ["ANI"]
    if name not in implemented_datasets:
        raise (NotImplementedError(
            f"Dataset {name} not implemented! Must be either of {implemented_datasets}!"))

    # Create path for data if it does not exist
    data_dir = Path(path) / "_data"
    data_dir.mkdir(exist_ok=True)
    data_dir = data_dir / name

    match name:
        case "ANI":
            return get_ANI(data_dir, config)
