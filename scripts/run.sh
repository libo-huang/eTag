#!/bin/bash
# bash -i run.sh <dataset> <gpu-0> <num_task>
# dataset: imagenet
cd ..
eval $(conda shell.bash hook)
conda activate HedTog
conda info | egrep "conda version|active environment" 

if [ "$1" != "" ]; then
    echo "Running on dataset: $1"
else
    echo "No dataset has been assigned."
fi

if [ "$2" != "" ]; then
    echo "Running on gpu: $2"
else
    echo "No gpu has been assigned."
fi

if [ "$3" != "" ]; then
    echo "Running with # tasks: $3"
else
    echo "No # task has been assigned."
fi

for SEED in 0 1 2 3 4 
do 
  if [ "$1" = "imagenet" ]; then
    python eTag_train.py -data imagenet_sub -log_dir ./checkpoints/imagenet -num_task $3 -nb_cl_fg 50 -gpu $2 -epochs 70 -epochs_gan 100 -tau 3 -lr_decay_step 30 -seed $SEED;
    python eTag_eval.py -data imagenet_sub -log_dir ./checkpoints/imagenet -num_task $3 -nb_cl_fg 50 -gpu $2 -epochs 70 -seed $SEED;
  elif [ "$1" = "cifar" ]; then
    python eTag_train.py -data cifar100 -log_dir ./checkpoints/cifar -num_task $3 -nb_cl_fg 50 -gpu $2 -epochs 100 -epochs_gan 100 -tau 3 -lr_decay_step 30 -seed $SEED;
    python eTag_eval.py -data cifar100 -log_dir ./checkpoints/cifar -num_task $3 -nb_cl_fg 50 -gpu $2 -epochs 100 -seed $SEED;
  else
    echo "No dataset has been assigned."
done