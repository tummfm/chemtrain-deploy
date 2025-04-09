## Test Case: Running LAMMPS with Chemtrain-dev Plugin

The script `in_mace.run` is used to run a LAMMPS simulation with the ChemTrain-dev plugin. It can be run directly under current directory path.

### Key Notes:

- **`comm_modify` Setting**:  
  Since the MACE message passing layer we used is 2, the communication cutoff should be calculated as:  
  `cutoff × (MP + 1) + neighbor skin = 5 * (2+1) + 2.5 = 17.5`.
  Also, we always use `newton on`.

- **`read_data`**:  
  This command loads the initial configuration for the simulation. The structure is chignolin solvated in water. It was originally prepared in GROMACS and then converted to the LAMMPS `.lmpdat` format.

- **`pair_coeff`**:  
  This line specifies the machine learning potential model, which should be provided in `.ptb` format.

### How to Run:

```bash
lmp -in in_mace.run
