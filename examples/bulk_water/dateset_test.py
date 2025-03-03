import numpy as onp
import hashlib
from tqdm import tqdm

# Load dataset
box = onp.load("/home/weilong/workspace/chemsim-lammps/datasets/water/h2o-data_lw_pimd/coord.npy")
energy = onp.load("/home/weilong/workspace/chemsim-lammps/datasets/water/h2o-data_lw_pimd/energy.npy")
force = onp.load("/home/weilong/workspace/chemsim-lammps/datasets/water/h2o-data_lw_pimd/force.npy")
print("box_new_size", box.shape)
print("energy_shape", energy.shape)
# Dictionary to store hashes of configurations
hash_dict = {}

# Loop through the dataset
redundant_pairs = []

for i in tqdm(range(box.shape[0])):
    # Hash the configuration
    hash_val = hashlib.sha256(box[i].tobytes()).hexdigest()
    
    # Check for redundancy
    if hash_val in hash_dict:
        redundant_pairs.append((hash_dict[hash_val], i))
    else:
        hash_dict[hash_val] = i

# Print redundant pairs
for pair in redundant_pairs:
    print(pair)
print("done")
print(box[2526][:10])
print(box[44002][:10])
print(energy[2526])
print(energy[44002])
print(force[2526][:10])
print(force[44002][:10])

print(force[9169][:10])
print(force[61979][:10])
print(force[21626][:10])
print(force[74446][:10])
print(force[34132][:10])
print(force[87009][:10])