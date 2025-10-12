#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=15:00:00                   # Adjust as needed
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --account=vcj@h100
#SBATCH -C h100

module purge


module load arch/h100
module load pytorch-gpu/py3/2.4.0 

module unload cudnn
module load cudnn/9.8.0.87-cuda

conda activate demo

nvidia-smi

ls /usr/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

python fb_agent.py