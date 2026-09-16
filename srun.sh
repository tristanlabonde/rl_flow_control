#!/bin/bash
#SBATCH -p wells01
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -J cans
#SBATCH -o jobs/stdout.%J
#SBATCH -e jobs/stderr.%J
#SBATCH --gres=gpu:1
export OMP_NUM_THREADS=1

CANS_DIR=/home/kametaniy/CaNS/CaNS_bmp
export LD_LIBRARY_PATH=${CANS_DIR}/dependencies/cuDecomp/build/lib:$LD_LIBRARY_PATH
##echo HOST=$(hostname)
##echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
##nvidia-smi -L
##nvidia-smi

rm -rf data/*
mpirun -np 1  ./cans
