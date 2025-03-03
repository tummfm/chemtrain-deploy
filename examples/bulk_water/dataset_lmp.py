import numpy as np
from chemutils.datasets.water import get_train_val_test_set
from pathlib import Path
from jax_md_mod import custom_space
import jax
dataset_path = Path("/home/weilong/workspace/chemsim-lammps/datasets/water/h2o-data_lw_pimd")
dataset = get_train_val_test_set([dataset_path])  # Pass Path object directly
fractional = False
if fractional:
    for split in dataset.keys():
        _, scale_fn = custom_space.init_fractional_coordinates(
            dataset[split]['box'][0])
        vmap_scale_fn = jax.vmap(lambda R,box: scale_fn(R, box=box),
                                    in_axes=(0, 0))
        dataset[split]['R'] = vmap_scale_fn(dataset[split]['R'], dataset[split]['box'])

indicies = 12492 # some random indice

positions = dataset['training']['R']
position = positions[indicies]

boxs = dataset['training']['box']
box_sample = boxs[indicies]

atom_type = dataset['training']['species']
atom_type_sample = atom_type[indicies]


num_atoms = position.shape[0]


# Define box boundaries (modify as needed)
xlo, xhi = 0.0, box_sample[0][0]
ylo, yhi = 0.0, box_sample[1][1]
zlo, zhi = 0.0, box_sample[2][2]

# Write to LAMMPS data file
with open("water.lmp", "w") as f:
    f.write("LAMMPS data file\n\n")
    f.write(f"{num_atoms} atoms\n")
    f.write("2 atom types\n")
    f.write(f"{xlo} {xhi} xlo xhi\n")
    f.write(f"{ylo} {yhi} ylo yhi\n")
    f.write(f"{zlo} {zhi} zlo zhi\n")
    f.write("0.0 0.0 0.0 xy xz yz\n\n")
    f.write("Atoms\n\n")
    
    for i, (x, y, z) in enumerate(position, start=1):

        species = atom_type_sample[i-1]
        f.write(f"{i} {species+1} {x:.3f} {y:.3f} {z:.3f}\n")

print("LAMMPS data file 'water.lmp' generated successfully.")
