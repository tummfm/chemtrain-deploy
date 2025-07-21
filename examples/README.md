## Test Case Setup and Instructions

There are **three test cases** provided:

1. **Aluminium**
2. **Water Vapor**
3. **Peptides** (solvated in water)

### Folder Structure

Each test case has its own main folder:

- `aluminium/`
- `water/`
- `peptides/`

Inside each of these folders:

- `trainer/`: contains training-related files.
- `simulation/`: used for running scale-up simulations.

> **Note:** For the peptides case, **Chignolin** is used specifically for the scale-up simulation.

### Simulation Folder Details

Within each `simulation/` folder:

- Pretrained model weights are saved in the `models/` directory.
- Please create an `output/` folder (if not already present) to store log files. This folder should be at the **same level as** the `run.sh` script.

### Running Simulations

To run a simulation:

1. Set your CUDA device (e.g., for a single GPU):
   ```bash
   export CUDA_VISIBLE_DEVICES=0
   ```

2. Run the simulation using MPI:
   ```bash
   mpirun -np <num_processes> ./run.sh <mode> <model>
   ```
   - `<num_processes>` should equal the number of GPUs you plan to use (matching `CUDA_VISIBLE_DEVICES`).
   - `<mode>`: either `strong` or `weak` scaling.
   - `<model>`: one of `allegro`, `painn`, or `mace`.

   **Example:**
   ```bash
   mpirun -np 2 ./run.sh strong allegro
   ```
