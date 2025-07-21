import functools
import h5py
import sqlite3
import zlib
import re
import numpy as onp
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Union

import jax
from jax import numpy as jnp, tree_util

from jax_md_mod.model import sparse_graph
from jax_md import partition


def show_progress(block_num, block_size, total_size):
    print(f"Progess: {round(block_num * block_size / total_size * 100, 2)} %",
          end="\r")


def estimate_edge_and_triplet_count(dataset, displacement_fn, r_cutoff=0.5,
                                    capacity_multiplier=1.25):
    """Iterates through all data of all splits and determines the minimum dimensions of the graph.

    Args:
        dataset: Dictionary containing multiple splits of the full dataset.
        displacement_fn: Function to compute displacement between particles.
        r_cutoff: Cutoff radius within particles are considered neighbors
        capacity_multiplier: Factor to estimate max. number of neighbors based
            on the computes number of neighbors.

    Returns:
        Returns arrays containing the maximum number of neighbors, edges, and
        triplets of the graph and a sufficiently large neighbor list for the
        given dataset.

    """

    @jax.jit
    def compute_and_check_overflow(position, neighbor, box):
        neighbor = neighbor.update(position, box=box)
        dynamic_displacement = functools.partial(displacement_fn, box=box)

        # Compute a graph and return the estimated number of weights and triplets
        graph, _ = sparse_graph.sparse_graph_from_neighborlist(
            dynamic_displacement, position, neighbor, r_cutoff
        )

        max_triplets = jnp.int32(jnp.ceil(graph.n_triplets))
        max_edges = jnp.int32(jnp.ceil(graph.n_edges))

        return max_edges, max_triplets, neighbor.did_buffer_overflow

    n_samples = 0
    for split in dataset.keys():
        n_samples += dataset[split]['box'].shape[0]

    all_max_neighbors, all_max_edges, all_max_triplets = onp.zeros(
        (3, n_samples), dtype=int)

    neighbor = None
    overflow = False

    neighbor_fn = partition.neighbor_list(
        displacement_fn, 1.0, r_cutoff, capacity_multiplier=capacity_multiplier,
        disable_cell_list=False, dr_threshold=0.0, fractional_coordinates=True
    )

    def compute_and_check(position, neighbor, box, overflow):
        if neighbor is None or overflow:
            print(
                f"Re-compute the neighborlist for samples {idx} in split {split}")
            neighbor = neighbor_fn.allocate(position, box=box)

            return compute_and_check(position, neighbor, box, False)

        return *compute_and_check_overflow(position, neighbor, box), neighbor

    n_iter = 0
    for split in dataset.keys():
        for idx in range(dataset[split]['box'].shape[0]):
            box, position = dataset[split]['box'], dataset[split]['R']

            # assert onp.all(position <= 1.0), f"Fractional coordinates are wrong."

            max_edges, max_triplets, overflow, neighbor = compute_and_check(
                jnp.asarray(position[idx]), neighbor, jnp.asarray(box[idx]),
                overflow
            )

            all_max_edges[n_iter] = int(max_edges)
            all_max_triplets[n_iter] = int(max_triplets)
            all_max_neighbors[n_iter] = int(neighbor.idx.shape[1])

    return all_max_neighbors, all_max_edges, all_max_triplets, neighbor


def make_supercell(dataset, a=1, b=1, c=1, fractional=True):
    """Transforms the boxes of the dataset into supercells.

    Args:
        dataset: Dictionary containing multiple splits of the full dataset.
        a, b, c: Repetitions of the cells in the directions of the base
            vectors.
        fractional: Whether positions are given in fractional coordinates.

    Returns:
        Returns the dataset tiled into supercells.

    """

    assert fractional, "Not implemented in real space."

    # Move all particles into the original boxes
    for split in dataset.keys():
        dataset[split]["R"] = onp.mod(dataset[split]["R"], 1.0)

    # For all samples and all particles, we add all combinations of the multiples
    # of a, b, and c.
    @functools.partial(jax.vmap, in_axes=(0, None, None, None))  # All samples
    @functools.partial(jax.vmap, in_axes=(0, None, None, None))  # All particles
    @functools.partial(jax.vmap,
                       in_axes=(None, None, None, 0))  # All c repetitions
    @functools.partial(jax.vmap,
                       in_axes=(None, None, 0, None))  # All b repetitions
    @functools.partial(jax.vmap,
                       in_axes=(None, 0, None, None))  # All a repetitions
    def tile_positions_and_forces(subset, a_off, b_off, c_off):
        subset["R"] += jnp.asarray([a_off, b_off, c_off])
        # We do not need to changes forces, returned as is
        return subset

    for split in dataset.keys():
        a_off = jnp.arange(a)
        b_off = jnp.arange(b)
        c_off = jnp.arange(c)

        # We also add the forces to replicate them
        tiled_subset = tile_positions_and_forces(
            {"R": dataset[split]["R"], "F": dataset[split]["F"]},
            a_off, b_off, c_off
        )

        # We now combine the replicas of the box into a single large one.
        # Additionally, we have to scale back into fractional coordinates
        n_samples = tiled_subset["R"].shape[0]
        dataset[split]["R"] = tiled_subset["R"].reshape((n_samples, -1, 3))
        dataset[split]["R"] *= jnp.asarray([1 / a, 1 / b, 1 / c])
        dataset[split]["F"] = tiled_subset["F"].reshape((n_samples, -1, 3))

        # Virial normalized by volume is intensive, internal energy and box
        # are extensive

        dataset[split]["U"] *= a * b * c
        dataset[split]["box"] *= onp.asarray([a, b, c])[:, None]

    # Ensure that dataset entries are still numpy arrays
    return tree_util.tree_map(onp.asarray, dataset)


def find_max_num_atoms_per_file(file: str) -> int:
    """
    Finds maximum number of atoms of all samples in a file.

    Args:
        file: Path to the file to check.

    Returns:
        max_num_atoms: Maximum number of atoms of all samples in given file.
    """
    # Initialize maximum to zero
    max_num_atoms = 0
    # Check file
    with h5py.File(file, "r") as f:
        for _, group in f.items():
            for key, data in group.items():
                if key == "species":
                    # Update maximum number of atoms with length of species array
                    max_num_atoms = max(max_num_atoms, len(data))
    return max_num_atoms


def find_max_num_atoms(h5_files: list[str]) -> int:
    """
    Finds maximum number of atoms of all samples in a list of files.

    Args:
        h5_files: List of paths to the files to check.

    Returns:
        max_num_atoms: Maximum number of atoms of all samples in all of the given files.
    """
    # Initialize maximum to zero
    max_num_atoms = 0
    for file in h5_files:
        # Update maximum number of atoms for each file
        max_num_atoms = max(max_num_atoms, find_max_num_atoms_per_file(file))
    return max_num_atoms


def convert_ANI_padded(data_dir: str, expand: bool) -> None:
    """
    Converts dataset to a list of .npy files containing all entries for a quantity.

    Args:
        data_dir: Path to directory where all files should be stored.
        expand: True if all atoms in a sample should be duplicated in all spatial directions.
    """

    def pad_array(array: onp.array, target_length: int) -> onp.array:
        """
        Pads a 2D array with zeros to a target length along the 2nd dimension.

        Args:
            array: Unpadded array.
            target_length: Desired length of the final array.

        Returns:
            array: Array padded to target length with zeros.
        """
        # Compute necessary padding
        padding_needed = target_length - array.shape[-1]

        if padding_needed > 0:
            # Pad array
            padding = ((0, 0), (0, padding_needed))
            array = onp.pad(array, padding, mode="constant", constant_values=0)

        return array

    print("Extracting and converting data...")

    # List of all h5 files in raw dataset
    h5_files = [f"{data_dir}/ANI_Al_raw/data-{i}.h5" for i in range(1, 44)]

    # Find maximum number of atoms in all samples
    max_num_atoms = find_max_num_atoms(h5_files)

    merged_datasets = {}
    # Iterate over files
    for file in h5_files:
        # Open file
        with h5py.File(file, "r") as f:
            # Iterate over groups in file
            for _, group in f.items():
                # Iterate over keys in group
                for key, data in group.items():
                    data = onp.asarray(data)
                    if key == "force":
                        # Get number of samples and atoms form data shape
                        num_samples, num_atoms, _ = data.shape
                    if data.ndim == 3:
                        # Reshape 2D arrays to 1D. 3rd dimension is the list of entries
                        data = data.reshape(data.shape[0], -1)
                        if key != "cell":
                            # Leave cells at current length, pad all other 2D arrays to maximum number of atoms
                            data = pad_array(data, 3 * max_num_atoms)
                    if key not in merged_datasets:
                        # Add key to dataset in first iteration
                        merged_datasets[key] = data
                    else:
                        # Concatenate all samples of each quantity to the corresponding array in the dataset
                        merged_datasets[key] = onp.concatenate(
                            (merged_datasets[key], data), axis=0)
                # Mask padded atoms
                mask_per_sample = onp.arange(max_num_atoms) < num_atoms
                mask = onp.tile(mask_per_sample, num_samples)
                # Add mask to dataset
                if "mask" not in merged_datasets:
                    merged_datasets["mask"] = mask
                else:
                    merged_datasets["mask"] = onp.concatenate(
                        (merged_datasets["mask"], mask), axis=0)

    # Reshape mask similar to other quantities
    merged_datasets["mask"] = onp.reshape(merged_datasets["mask"],
                                          (-1, max_num_atoms))

    print("Saving converted data...")

    if expand:
        # Save converted data to intermediate folder if it should be extended after
        data_dir = data_dir / "ANI_Al_single"
    else:
        # Directly store dataset at final folder
        data_dir = data_dir / "ANI_Al"

    # Create folder if it does not exist already
    data_dir.mkdir(exist_ok=True)

    for key, data in merged_datasets.items():

        # Rename quantities to match other datasets and key functions
        if key == "cell":
            key = "box"
        if key == "coordinates":
            key = "coord"

        # Save all quantites as .npy file
        onp.save(f'{data_dir}/{key}.npy', data)


def expand_dataset(data_dir: str) -> None:
    """
    Expand each sample of the dataset by duplicating all atoms in each spatial direction.
    Resulting samples contain 8x as many atoms.

    Args:
        data_dir: Path to directory where all files should be stored.
    """
    # Load original data
    data_dir_single = data_dir / "ANI_Al_single"
    box = onp.load(data_dir_single / "box.npy")
    coord = onp.load(data_dir_single / "coord.npy")
    force = onp.load(data_dir_single / "force.npy")
    energy = onp.load(data_dir_single / "energy.npy")
    mask = onp.load(data_dir_single / "mask.npy").astype(int)

    # Get maximum number of atoms
    max_atoms = int(coord.shape[-1] / 3)
    # Get number of samples
    num_samples = coord.shape[0]
    # Initialize array in which atoms need to be shifted during duplication
    shift_directions = onp.array(
        onp.meshgrid([0, 1], [0, 1], [0, 1])).T.reshape(-1, 3)

    def expand_array(array: onp.array, box: onp.array,
                     shift: bool = True) -> onp.array:
        """
        Expand array to 8x length to duplicat samples in all spatial directions.

        Args:
            array: Array that should be expanded. Either coordinates R or forces F
            box: Simulation box of the sample.
            shift: True if the samples should be shifted along box dimension. True for R; False for F.

        Returns:
            array: Expanded array of 8x the original length.
        """
        # Remove padding
        array = onp.trim_zeros(array, "b")
        num_atoms = int(len(array) / 3)

        # Get 8 copies of original trimmed array
        array = onp.tile(array, 8)

        # Pad up to 8x original untrimmed array length
        padding = 24 * (max_atoms - num_atoms)
        array = onp.pad(array, (0, padding), constant_values=0)

        # Add a shift to coordinates
        if shift:
            shift_arr = onp.array([])
            # Iterate over all directions in which the array should be shifted
            for shift_dir in shift_directions:
                # Shift in the directions of the simulation box
                shift_dir = onp.reshape(box, (3, 3)) @ shift_dir
                # Repeat for each atom
                shift_dir = onp.tile(shift_dir, num_atoms)
                shift_arr = onp.concatenate((shift_arr, shift_dir))

            # Pad to match original array shape
            shift_arr = onp.pad(shift_arr, (0, padding), constant_values=0)
            array += shift_arr

        return array

    def expand_mask(array: onp.array) -> onp.array:
        """
        Expand maks to 8x the original mask.

        Args:
            array: Original mask array.

        Returns:
            array: Padded mask with 8x the original length.
        """
        # Remove padding
        array = onp.trim_zeros(array, "b")
        num_atoms = int(len(array))

        # Get 8 copies of original trimmed array
        array = onp.tile(array, 8)

        # Pad up to 8x original untrimmed array length
        padding = 8 * (max_atoms - num_atoms)
        array = onp.pad(array, (0, padding), constant_values=0)

        return array

    # Expand arrays to match duplicates
    # Shift coordinates along box dimension, but leave forces unaltered
    coord = onp.array(
        [expand_array(coord[sample], box[sample], shift=True) for sample in
         range(num_samples)])
    force = onp.array(
        [expand_array(force[sample], box[sample], shift=False) for sample in
         range(num_samples)])
    mask = onp.array(
        [expand_mask(mask[sample]) for sample in range(num_samples)])

    # Create final folder if it does not exist already
    data_dir = data_dir / "ANI_Al"
    data_dir.mkdir(exist_ok=True)

    # Save arrays in final folder
    onp.save(data_dir / "coord.npy", coord)
    onp.save(data_dir / "force.npy", force)
    onp.save(data_dir / "energy.npy", energy)
    onp.save(data_dir / "mask.npy", mask)

    # Double box dimensions to agree with new coordinates
    box *= 2
    onp.save(data_dir / "box.npy", box)


def convert_ANI(data_dir: str, expand: bool = False) -> None:
    """
    Convert raw ANI data to format needed for training and stores in .npy files.

    Args:
        data_dir: Path to directory where data should be stored.
        expand: True if samples should be expanded by duplicating all atoms in each spatial direction.
    """
    # Convert dataset to right format. Expand if necessary
    if expand:
        # First convert normal dataset
        if not (data_dir / "ANI_Al_single").exists():
            convert_ANI_padded(data_dir, expand)
        # Expand normal dataset
        if not (data_dir / "ANI_Al").exists():
            print("Expanding samples...")
            expand_dataset(
                data_dir)  # double in all dimensions -> 8x atoms per sample
    else:
        # Directly convert normal dataset
        if not (data_dir / "ANI_Al").exists():
            convert_ANI_padded(data_dir, expand)
