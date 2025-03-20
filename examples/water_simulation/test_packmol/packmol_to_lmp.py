import sys
def xyz_to_lammps(xyz_file, lammps_file):
    with open(xyz_file, 'r') as file:
        lines = file.readlines()
    num_atoms = int(lines[0].strip())
    atoms = []
    atom_types = {
    "O": 1,
    "H": 2,
    }
    box_size = 30
    for line in lines[2:num_atoms+2]:
        parts = line.split()
        element, x, y, z = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
        atoms.append((atom_types[element], x, y, z))
    with open(lammps_file, 'w') as f:
        f.write("#LAMMPS data file generated from XYZ file\n\n")
        f.write(f"{num_atoms} atoms\n")
        f.write(f"2 atom types\n")
        f.write(f"0 {box_size} xlo xhi\n")
        f.write(f"0 {box_size} ylo yhi\n")
        f.write(f"0 {box_size} zlo zhi\n")
        f.write("\nAtoms\n\n")
        for i, (atom_type, x, y, z) in enumerate(atoms, start=1):
            f.write(f"{i} {atom_type} {x} {y} {z}\n")
if __name__ == "__main__":
    xyz_to_lammps("/home/weilong/workspace/chemsim-lammps/examples/water_simulation/test_packmol/output.xyz",
                "/home/weilong/workspace/chemsim-lammps/examples/water_simulation/test_packmol/water_packmol.lmp")




