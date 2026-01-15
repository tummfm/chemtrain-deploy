"""Downloads and prepares the SPICE dataset."""

from pathlib import Path

import re
import jax
import numpy as onp

from chemtrain.data import preprocessing


def load_pepsol(data_dir="./_data",
                scale_R=1,
                scale_U=1,
                fractional=True,
                max_samples=None,
                subsets=None,
                **kwargs):
    """Load and process Pepsol dataset into workspace.

    Units of Pepsol dataset:
        [R] = nm,
        [U] = kJ/mol,
        [F] = kJ/mol/nm

    Args:
        data_dir: Directory of Pepsol dataset.
        scale_R: Scaling factor for atomic positions (default=1 #nm -> nm).
        fractional: Whether to scale positions by simulation box (default=true).
        subsets: List of regex to select subsets of the dataset.

    Returns:
        The complete Pepsol dataset.
    """
    with jax.default_device(jax.devices("cpu")[0]):
        dataset = load_from_file(data_dir)
        dataset, subsets = select_subsets(data_dir, dataset, subsets)
        dataset = scale_dataset(dataset, scale_R, scale_U, fractional)
        dataset = split_by_subset(dataset, max_samples, **kwargs)

    info = {
        "subsets": subsets,
        "scaling": {
            "R": scale_R,
            "U": scale_U,
            "fractional": fractional
        },
    }

    return dataset, info


def load_from_file(data_dir, filename="PepSol.npz"):
    """Load Pepsol dataset from file into workspace

        Args:
            data_dir: Directory of Pepsol dataset.
            filename: Filename of Pepsol dataset.

        Returns:
            The complete Pepsol dataset.
        """
    # Do not process more than once
    assert (Path(data_dir + "/" + filename)).exists(), f'File {data_dir + "/" + filename} does not exist.'
    return dict(onp.load(data_dir + "/" + filename))


def select_subsets(data_dir, dataset, subsets):
    """Selects the subsets from the dataset."""
    if subsets is None:
        return dataset, subsets

    with open(Path(data_dir) / "subsets.dat", "r") as file:
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

        weights = n_particles / onp.mean(n_particles, keepdims=True)
        dataset[split]['F_weight'] = max_particles / n_particles

    return dataset


def scale_dataset(dataset, scale_R, scale_U, fractional=True):
    """Scales the dataset to kJ/mol and to nm."""

    print(f"Original positions: {dataset['R']}")
    print(f"Original positions: {dataset['R'].min()} to {dataset['R'].max()}")


    if fractional:
        box = dataset['box'][0, 0, 0]
        dataset['R'] = dataset['R'] / box
    else:
        dataset['R'] = dataset['R'] * scale_R

    print(f"Scale dataset by {scale_R} for R and {scale_U} for U.")

    scale_F = scale_U / scale_R
    dataset['box'] = scale_R * dataset['box']
    dataset['F'] *= scale_F

    return dataset

