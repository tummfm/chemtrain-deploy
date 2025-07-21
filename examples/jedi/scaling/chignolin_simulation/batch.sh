#!/bin/bash -x
#SBATCH --job-name=ctd-pep
#SBATCH --nodes=8

# Production: Always use all GPUs per node
#SBATCH --ntasks-per-node=4

# Budget account where contingent is taken from
#SBATCH --account=chemtrain-deploy

# Share CPUs equally among four tasks
#SBATCH --cpus-per-task=72

# Run for at most 10 minutes
#SBATCH --time=00:10:00

#SBATCH --partition=all

# *** start of job script ***

# Note: The current working directory at this point is
# the directory where sbatch was executed.

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun bash run.sh $1 $2
