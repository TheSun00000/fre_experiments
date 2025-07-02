#!/bin/bash
#SBATCH --output=.slurm_logs/%x_%j.out
#SBATCH --error=.slurm_logs/%x_%j.err
#SBATCH --time=10:00:00                   # Adjust as needed
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


# python offline_fre.py --iql_training_steps 1000000
# python offline_fre+traj_score.py --iql_training_steps 300000
python offline_fre+expectile_G.py --iql_training_steps 500000 --folder_name "r=0.5 500000" --optimal_states_ratio 0.5
