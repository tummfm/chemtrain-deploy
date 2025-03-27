# Convert PDB file to gro initial structure and select AMBER94 FF
printf "2 \n" | gmx pdb2gmx -f C5.pdb -water tip3p -ignh \
              -o simulation/initial.gro \
              -p simulation/topol.top \
              -i simulation/posre.itp

# Edit the box size
gmx editconf -f simulation/initial.gro -bt cubic -box 2.7 2.7 2.7 \
    -o simulation/cubic.gro

# Solvate the protein
gmx solvate -cp simulation/cubic.gro -cs spc216.gro \
    -p simulation/topol.top \
    -o simulation/water.gro
