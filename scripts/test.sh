#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=8:00:00                   # Adjust as needed
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --account=vcj@v100


module load pytorch-gpu/py3/1.2.0
conda activate olivier

nvidia-smi

python antmaze_test_with_z.py