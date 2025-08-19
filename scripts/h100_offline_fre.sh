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


python main_v1.py --env_name cheetah --method fre --policy_extraction_method ddpg\
            --reward_generator_training_steps 1 --rg_dropout 0.5 \
            --encoder_training_steps 0 \
            --iql_training_steps 1000000 \
            --num_evals 5 \
            --encoder_checkpoint models/2025-08-15_19-28-25_cheetah-fre-awr/fre_network.pth \


# python main_v1_var_test.py --env_name walker --method fre --policy_extraction_method ddpg\
#             --reward_generator_training_steps 20000 --rg_dropout 0.5 \
#             --topk_rewards 500 --encoder_training_steps 0 \
#             --iql_training_steps 1000000 \
#             --num_evals 5 \
#             --encoder_checkpoint models/2025-08-15_19-32-35_walker-fre-awr/fre_network.pth

# --encoder_checkpoint models/2025-08-15_19-32-35_walker-fre-awr/fre_network.pth


# python main_v1_eval_test.py --env_name walker --method fre --policy_extraction_method ddpg --benchmark_set 3 \
#             --num_unsupervised_reward_functions 0\
#             --iql_training_steps 1000000 \
#             --num_evals 5 \
#             --encoder_checkpoint models/2025-08-15_19-32-35_walker-fre-awr/fre_network.pth \




# python iql_dmc.py

# python main-ogbench.py --env_name antmaze-large-navigate-v0 --method fre \
#             --reward_generator_training_steps 20000 --rg_dropout 0.5 \
#             --encoder_training_steps 7000 \
#             --iql_training_steps 1000000 \
#             --num_evals 5



# python main-supervised.py --env_name walker --method fre \
#             --reward_generator_training_steps 1 --rg_dropout 0.5 \
#             --encoder_training_steps 1 \
#             --iql_training_steps 1000000 \
#             --num_evals 10



# python main.py --env_name cheetah --method rg \
#         --reward_generator_training_steps 0 --rg_dropout 0.5 \
#         --encoder_training_steps 0 \
#         --iql_training_steps 0 \
#         --num_evals 5 \
#         --rg_checkpoint      models/2025-07-31_15-17-24_cheetah-rg/rg_model.pth \
#         --encoder_checkpoint models/2025-07-31_15-17-24_cheetah-rg/fre_network.pth \
#         --iql_checkpoint     models/2025-07-31_15-17-24_cheetah-rg/iql_agent.pth \


# python main-beast.py --env_name antmaze --method fre \
#                --reward_generator_training_steps 1 --rg_dropout 0.5 \
#                 --iql_training_steps 100000 \
#                --num_evals 5


# python main.py --env_name walker --method fre \
#                --reward_generator_training_steps 20000 --rg_dropout 0.5 \
#                --encoder_training_steps 50000 \
#                --iql_training_steps 300000 \
#                --num_evals 5






# python offline_fre-dmc+topk_trajectories.py --reward_generator_training_steps 20000 --encoder_training_steps 20000 --iql_training_steps 100000 --num_evals 5 --method fre



# python offline_fre-dmc.py --reward_generator_training_steps 15000 --encoder_training_steps 100000 --iql_training_steps 300000 --num_evals 5 --method rg --file_suffix without_aux
# python offline_fre-dmc.py --reward_generator_training_steps 15000 --encoder_training_steps 100000 --iql_training_steps 300000 --num_evals 5 --method fre --file_suffix only_linear


# python offline_fre-dmc.py --encoder_training_steps 0 --iql_training_steps 300000 --num_evals 5





# python offline_fre+topk_trajectories.py --iql_training_steps 100000

# python offline_fre.py --iql_training_steps 1000000
# python offline_fre+reward_generator.py --iql_training_steps 100000
# python offline_fre+traj_score.py --iql_training_steps 300000
# python offline_fre+expectile_G.py --iql_training_steps 500000 --folder_name "r=0.5 500000" --optimal_states_ratio 0.5
# python offline_fre+expectile_G.py --iql_training_steps 10 --folder_name "tmp" --optimal_states_ratio 0.5
