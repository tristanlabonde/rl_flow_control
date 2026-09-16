#!/bin/bash
#SBATCH -p wells01
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -J flow_control
#SBATCH -o saved_std/stdout_control.%J
#SBATCH -e saved_std/stderr_control.%J
#SBATCH --gres=gpu:1
export OMP_NUM_THREADS=1

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate conda_env

CANS_DIR=/home/kametaniy/CaNS/CaNS_bmp
export LD_LIBRARY_PATH=${CANS_DIR}/dependencies/cuDecomp/build/lib:$LD_LIBRARY_PATH
##echo HOST=$(hostname)
##echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
##nvidia-smi -L
##nvidia-smi

NB_ROUND=$1

if [ -z "$NB_ROUND" ]; then
	python -u main.py --verbose input_mean_env.txt
else
	python -u main.py --verbose input_mean_env.txt $NB_ROUND
fi
