#! /bin/bash

# Infer the number of GPUs from the rank size
D=$OMPI_COMM_WORLD_SIZE

# For weak scaling_million_atoms, adjust the number of reps accordingly
if [ "$1" == "strong" ]; then
  script="strong_scaling.lmp"
  logg="strong"
elif [ "$1" == "weak" ]; then
  script="weak_scaling.lmp"
  logg="weak"
else
  echo "Invalid mode. Please set mode to either 'strong' or 'weak'."
  exit 1
fi

command="lmp"
if [ "$2" == "meam" ]; then
  command="/p/project1/chemtrain-deploy/source/lammps_kokkos/build-kokkos/lmp -k on g $OMPI_COMM_WORLD_LOCAL_SIZE -sf kk -pk kokkos newton on neigh half"
  model="meam"
  reps=64
  commdist=5.0
  if [ "$1" == "strong" ]; then
    script="strong_scaling_meam.lmp"
  else
    script="weak_scaling_meam.lmp"
  fi
elif [ "$2" == "mace" ]; then
  model="mace"
  reps=64
  commdist=10.0
elif [ "$2" == "allegro" ]; then
  model="allegro"
  reps=64
  commdist=5.0
elif [ "$2" == "painn" ]; then
  model="painn"
  reps=64
  commdist=20.0
else
  echo "Invalid model. Please set model to either 'mace/allergo/painn' or 'comming soon'."
  exit 1
fi

$command -in $script \
    -v name $model \
    -v procs $D \
    -v model ./models/${model}.ptb \
    -v Nrep $reps \
    -v commdist $commdist \
    -v logfile "output/${logg}_scaling_${model}_${D}_gpus.log"
