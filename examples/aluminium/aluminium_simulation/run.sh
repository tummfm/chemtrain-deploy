#! /bin/bash

# Infer the number of GPUs from the rank size
D=$OMPI_COMM_WORLD_SIZE

# For weak scaling, adjust the number of reps accordingly
if [ "$1" == "strong" ]; then
  script="strong_scaling.lmp"
elif [ "$1" == "weak" ]; then
  script="weak_scaling.lmp"
else
  echo "Invalid mode. Please set mode to either 'strong' or 'weak'."
  exit 1
fi

if [ "$2" == "mace" ]; then
  model="mace"
  reps=10 # Maximum number of repetitions in each dimension
  commdist=10.0 # Communication distance required by mace model
else
  echo "Invalid model. Please set model to either 'mace' or 'comming soon'."
  exit 1
fi

lmp -in $script \
    -v name $model \
    -v procs $D \
    -v model ../models/${model}.ptb \
    -v Nrep $reps \
    -v commdist $commdist \
    -v logfile "output/strong_scaling_${model}_${D}_gpus.log"
