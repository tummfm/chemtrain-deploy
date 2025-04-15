def extract_lammps_data(input_file, output_file):
    with open(input_file, "r") as f:
        lines = f.readlines()
    
    atoms_section = False
    velocities_section = False
    bonds_section = False
    atom_count = 0
    atom_data = []
    velocity_data = []
    box_bounds = []
    
    with open(output_file, "w") as out:
        # Write LAMMPS header
        out.write("LAMMPS data file \n\n")
        
        for line in lines:
            # Extract box boundaries
            if any(bound in line for bound in ["xlo xhi", "ylo yhi", "zlo zhi"]):
                box_bounds.append(line.strip())
                continue
            
            # Detect sections
            if "Atoms" in line:
                atoms_section = True
                velocities_section = False
                bonds_section = False
                continue
            elif "Velocities" in line:
                velocities_section = True
                atoms_section = False
                bonds_section = False
                continue
            elif "Bonds" in line:
                bonds_section = True
                atoms_section = False
                velocities_section = False
                continue
            elif line.strip() == "":
                continue
            
            # Extract atom data
            if atoms_section and line[0].isdigit():
                parts = line.split()
                atom_id = parts[0]
                atom_type = parts[2]
                x, y, z = parts[4], parts[5], parts[6]
                atom_data.append(f"{atom_id} {atom_type} {x} {y} {z}")
                atom_count += 1
            
            # Extract velocity data
            if velocities_section and line[0].isdigit():
                parts = line.split()
                atom_id = parts[0]
                vx, vy, vz = parts[1], parts[2], parts[3]
                velocity_data.append(f"{atom_id} {vx} {vy} {vz}")
        
        # Write atom count and types
        out.write(f"{atom_count} atoms\n")
        out.write("2 atom types\n\n")
        
        # Write box boundaries
        for bound in box_bounds:
            out.write(bound + "\n")
        out.write("\n")
        
        # Write atoms section
        out.write("Atoms\n\n")
        for atom in atom_data:
            out.write(atom + "\n")
        out.write("\n")
        
        # Write velocities section
        out.write("Velocities\n\n")
        for velocity in velocity_data:
            out.write(velocity + "\n")
        out.write("\n")
    
    print(f"Filtered LAMMPS data saved to {output_file}")
    
if __name__ == "__main__":
    input_filename = "/home/weilong/workspace/chemsim-lammps/examples/water_vapor/reference/final_config.data"  # Replace with your input file
    output_filename = "final_config_filtered.data"
    extract_lammps_data(input_filename, output_filename)