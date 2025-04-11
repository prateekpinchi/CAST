#!/bin/bash
#SBATCH --job-name=cast_finetune_pannuke
#SBATCH --account=stellayu
#SBATCH --output=slurm_out/train_output.log
#SBATCH --error=slurm_out/train_error.log
#SBATCH --partition=stellayu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pinchi@umich.edu
#SBATCH --time=24:00:00

# Activate ct_env
conda init
source ~/.bashrc
conda activate cast

# Train the model
bash scripts/segmenter/train_pannuke_cast.sh