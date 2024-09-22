#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=a100
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=60:00:00

eval "$(conda shell.bash hook)"
conda init bash
conda activate cl

export HF_HOME=/scratch/yw26g23/cache/

python captioner.py


