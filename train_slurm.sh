#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=scavenger_8h100
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00

eval "$(conda shell.bash hook)"
conda init bash
conda activate cl

export HF_HOME=/scratch/yw26g23/cache/
export WANDB_MODE="offline"
export WANDB_API_KEY="a4d3a740e939973b02ac59fbd8ed0d6a151df34b"
export NO_ALBUMENTATIONS_UPDATE=1

srun python train.py --cfg configs/cc12m.yaml


