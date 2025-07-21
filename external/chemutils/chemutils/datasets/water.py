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

"""Tools to download and process the water dataset."""

from pathlib import Path
import numpy as onp
import jax
import os
from chemtrain.data import preprocessing
from jax_md_mod import custom_space


def load_subset(data_dir, train_ratio=0.7, val_ratio=0.1):
    box = onp.load(data_dir / "box.npy", allow_pickle=True)
    coord = onp.load(data_dir / "coord.npy", allow_pickle=True)
    energy = onp.load(data_dir / "energy.npy", allow_pickle=True)
    force = onp.load(data_dir / "force.npy", allow_pickle=True)
    atom_type = onp.load(data_dir / "type.npy", allow_pickle=True)
    n_samples = box.shape[0]
    print("num_samples", n_samples)
    dataset = dict(
        box=onp.reshape(box, (n_samples, 3, 3)),
        # TODO: Check if swapaxes necessary
        R=onp.reshape(coord, (n_samples, -1, 3)),
        U=onp.reshape(energy, (n_samples,)),
        F=onp.reshape(force, (n_samples, -1, 3)),
        species=onp.reshape(atom_type, (n_samples, -1))
    )
    splits = preprocessing.train_val_test_split(
        dataset, train_ratio=train_ratio, val_ratio=val_ratio
    )
    return splits


def load_dataset(data_list):
    dataset = get_train_val_test_set(data_list)
    scale_energy = 96.4853722
    scale_pos = 0.1
    per_atom = False
    shift_U = True
    flip_forces = False
    fractional = True
    dataset = scale_dataset(dataset, scale_R=scale_pos, scale_U=scale_energy,
                            shift_U=shift_U, flip_forces=flip_forces,
                            fractional=fractional, per_atom=per_atom)
    return dataset


def get_train_val_test_set(dir_files):
    dataset = dict(
        training=dict(box=[], R=[], U=[], F=[], species=[]),
        validation=dict(box=[], R=[], U=[], F=[], species=[]),
        testing=dict(box=[], R=[], U=[], F=[], species=[])
    )
    print(dir_files)
    for i in range(len(dir_files)):
        train_split, val_split, test_split = load_subset(dir_files[i])
        for k in dataset['training'].keys():
            dataset['training'][k].append(train_split[k])
            dataset['validation'][k].append(val_split[k])
            dataset['testing'][k].append(test_split[k])
    for split in dataset.keys():
        for quantity in dataset[split].keys():
            dataset[split][quantity] = onp.concatenate(dataset[split][quantity],
                                                       axis=0)

    return dataset


def scale_dataset(dataset, scale_U=1.0, scale_R=1.0, shift_U=False,
                  flip_forces=False, fractional=True, per_atom=False):
    if shift_U:
        all_U = onp.concatenate(
            [dataset[split]["U"] for split in dataset.keys()])
        if per_atom:
            return NotImplementedError
        else:
            shift_energy = onp.mean(all_U)
    else:
        shift_energy = 0.0
    shift_energy *= scale_U
    scale_F = scale_U / scale_R
    if flip_forces:
        scale_F *= -1
    for split in dataset.keys():
        _, scale_fn = custom_space.init_fractional_coordinates(
            dataset[split]['box'][0])
        vmap_scale_fn = jax.vmap(lambda R, box: scale_fn(R, box=box),
                                 in_axes=(0, 0))
        if fractional:
            dataset[split]['R'] = vmap_scale_fn(dataset[split]['R'],
                                                dataset[split]['box'])
        else:
            dataset[split]['R'] = dataset[split]['R'] * scale_R

        dataset[split]['box'] *= scale_R
        dataset[split]['U'] *= scale_U
        dataset[split]['F'] *= scale_F

        if per_atom:
            return NotImplementedError

        dataset[split]['U'] -= shift_energy

    return dataset


def combine_raw_data(source_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    box = []
    coord = []
    energy = []
    force = []
    species = []

    for set_folder in sorted(os.listdir(source_dir)):
        if set_folder.startswith("set"):
            set_path = os.path.join(source_dir, set_folder)
            box_data = onp.load(os.path.join(set_path, "box.npy"),
                                allow_pickle=True)
            coord_data = onp.load(os.path.join(set_path, "coord.npy"),
                                  allow_pickle=True)
            energy_data = onp.load(os.path.join(set_path, "energy.npy"),
                                   allow_pickle=True)
            force_data = onp.load(os.path.join(set_path, "force.npy"),
                                  allow_pickle=True)
            species_data = onp.load(os.path.join(set_path, "type.npy"),
                                    allow_pickle=True)
            box.append(box_data)
            coord.append(coord_data)
            energy.append(energy_data)
            force.append(force_data)
            species.append(species_data)

    box = onp.concatenate(box, axis=0)
    coord = onp.concatenate(coord, axis=0)
    energy = onp.concatenate(energy, axis=0)
    force = onp.concatenate(force, axis=0)
    species = onp.concatenate(species, axis=0)

    onp.save(os.path.join(output_dir, "box.npy"), box)
    onp.save(os.path.join(output_dir, "coord.npy"), coord)
    onp.save(os.path.join(output_dir, "energy.npy"), energy)
    onp.save(os.path.join(output_dir, "force.npy"), force)
    onp.save(os.path.join(output_dir, "type.npy"), species)


def get_dataset(root="./datasets"):
    data_dir = Path(root) / "water"
    data_dir.mkdir(exist_ok=True)
    if not (data_dir / "h2o-data_raw").exists():
        import urllib.request
        url = "https://store.aissquare.com/datasets/455ff154-db9c-11ee-9b22-506b4b2349d8/PBE0-TS-H2O.zip"
        urllib.request.urlretrieve(url, data_dir / "PBE0-TS-H2O.zip")
        zip_path = data_dir / "PBE0-TS-H2O.zip"
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir / "h2o-data_raw")

    combine_raw_data(data_dir / "h2o-data_raw/lw_pimd",
                     data_dir / "h2o-data_lw_pimd")
    dataset = load_dataset(
        [data_dir / "h2o-data_lw_pimd"])  # load liquid water dataset
    return dataset


def get_random_subset(dataset, proportion=0.5, seed=42):
    """Randomly selects a subset of the dataset based on the given proportion."""
    onp.random.seed(seed)  # Ensure reproducibility
    subset = {}

    for split in dataset.keys():  # "training", "validation", "testing"
        subset[split] = {}
        total_samples = dataset[split]['R'].shape[0]
        num_samples = int(
            total_samples * proportion)  # Compute proportional number of samples

        # Generate random indices
        indices = onp.random.choice(total_samples, num_samples, replace=False)

        for key in dataset[split].keys():  # "box", "R", "U", "F"
            subset[split][key] = dataset[split][key][
                indices]  # Select random samples

    return subset
