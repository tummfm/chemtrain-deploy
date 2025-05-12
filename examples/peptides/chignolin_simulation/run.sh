#! /bin/bash

# Infer the number of GPUs from the rank size
D=$OMPI_COMM_WORLD_SIZE

# For weak scaling, adjust the number of reps accordingly
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

if [ "$2" == "mace" ]; then
  model="mace"
  reps=3
  commdist=10.0
  min=0
elif [ "$2" == "allegro" ]; then
  model="allegro"
  reps=4
  commdist=5.
  min=1
elif [ "$2" == "painn" ]; then
  model="painn"
  reps=2
  commdist=20.0
  min=0
else
  echo "Invalid model. Please set model to either 'mace/allergo/painn' or 'comming soon'."
  exit 1
fi

lmp -in $script \
    -v name $model \
    -v procs $D \
    -v minimize $min \
    -v model ./models/${model}.ptb \
    -v Nrep $reps \
    -v commdist $commdist \
    -v logfile "output/${logg}_scaling_${model}_${D}_gpus.log"