# [1]:


import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import numpy as np
import random

from tqdm import tqdm
from IPython.display import clear_output

import omegaconf
import dataclasses

import os
from datetime import datetime

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
device


# [2]:


import matplotlib.patches as patches


def add_largest_maze_walls(ax):    

    maze_optim = [
        (1, 1, 1, 2),    
        (0, 4, 2, 1),
        (3, 1, 1, 4),
        (5, 0, 1, 1),
        (4, 2, 3, 1),
        (1, 6, 3, 1),
        (4, 4, 2, 1),
        (5, 6, 2, 1),
        (1, 8, 1, 1),
        (3, 7, 1, 2),
        (5, 8, 1, 2)
    ]

    block_size = 0.025 * 80

    height, width = 7, 10
    torso_x, torso_y = (width - 1)*block_size, (height - 1)*block_size

    rects = []
    for i in range(len(maze_optim)):
        (y, x, w, h) = maze_optim[i]
            
        x = x * block_size * 2 - torso_x + (h - 1) * block_size - h * block_size + 18
        y = y * block_size * 2 - torso_y + (w - 1) * block_size - w * block_size + 12
        h, w = h * block_size * 2, w * block_size * 2
        w = w * 1.
        y = y * 1.
        rect = patches.Rectangle((x, y), h, w, linewidth=2, edgecolor='gray', facecolor='gray')

        ax.add_patch(rect)
        
  


# [3]:


ENV_NAME = 'cheetah' # cheetah | walker | antmaze


# [4]:


BENCHMARK_SET = [0, 1, 2, 3]

NUM_TRAJECTORIES = 10000
TRAJECTORY_LEN = 1000

training_steps = 1_000_000

now = datetime.now()
date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")

exp_name = f'SF-TD3-{ENV_NAME}'
    
LOGS_FOLDER = f'./logs/{date_time_str}_{exp_name}'
MODEL_SAVE_FOLDER = f'./models/{date_time_str}_{exp_name}'


os.makedirs(LOGS_FOLDER)
os.makedirs(MODEL_SAVE_FOLDER)

print('LOGS_FOLDER:', LOGS_FOLDER)
print('MODEL_SAVE_FOLDER:', MODEL_SAVE_FOLDER)


# [5]:


import gym
import d4rl # Import required to register environments, you may need to also import the submodule

# Create the environment
from dm_control import suite

if ENV_NAME == 'cheetah':
    STATE_DIM = 17
    ACTION_DIM = 6
    # AUX_DIM = 1
    env = suite.load(
        domain_name='cheetah',
        task_name='run',
        environment_kwargs=dict(flat_observation=True)
    )
    dataset = np.load('datasets/cheetah_rnd.npy', allow_pickle=True).item()
    aux = np.load('datasets/aux_cheetah.npy')
    angmomentum_aux = np.load('datasets/angmomentum_aux_cheetah.npy')
    aux = np.concatenate((
        aux.reshape(10000, 1000, 1),
        angmomentum_aux.reshape(10000, 1000, 1)
    ), axis=-1)
    # dataset['observations'] = np.concatenate((dataset['observations'], aux.reshape(10000, 1000, 1)), axis=-1)
    
elif ENV_NAME == 'walker':
    STATE_DIM = 24
    ACTION_DIM = 6
    # AUX_DIM = 3
    env = suite.load(
        domain_name='walker',
        task_name='run',
        environment_kwargs=dict(flat_observation=True)
    )
    dataset = np.load('datasets/walker_rnd.npy', allow_pickle=True).item()
    aux = np.load('datasets/aux_walker.npy')
    angmomentum_aux = np.load('datasets/angmomentum_aux_walker.npy')
    aux = np.concatenate((
        aux.reshape(10000, 1000, 3),
        angmomentum_aux.reshape(10000, 1000, 1)
    ), axis=-1)
    # dataset['observations'] = np.concatenate((dataset['observations'], aux.reshape(10000, 1000, 3)), axis=-1)
    
elif ENV_NAME == 'antmaze':
    import gym
    import d4rl
    STATE_DIM = 29
    ACTION_DIM = 8
    NUM_TRAJECTORIES = 999
    TRAJECTORY_LEN = 1001
    env = gym.make('antmaze-large-diverse-v2')
    dataset = env.get_dataset()


# [6]:


dataset_trajectories = torch.tensor(dataset['observations']).float()
dataset_trajectories = dataset_trajectories
# dataset_trajectories = torch.concatenate((dataset_trajectories[..., [0, 1]], dataset_trajectories[..., [15, 16]]), dim=-1)

dataset_actions = torch.tensor(dataset['actions']).float()
dataset_terminals = torch.tensor(dataset['terminals']).float()
dataset_timeouts = torch.zeros(NUM_TRAJECTORIES, TRAJECTORY_LEN).bool()
dataset_timeouts[:, -1] = True

dataset_goals = torch.tensor(dataset['infos/goal']).float()

if ENV_NAME == 'antmaze':
    N = NUM_TRAJECTORIES * TRAJECTORY_LEN
    dataset_trajectories = dataset_trajectories[:N]
    dataset_actions = dataset_actions[:N]
    dataset_terminals = dataset_terminals[:N]
    dataset_timeouts = dataset_timeouts[:N]


dataset_trajectories = dataset_trajectories.reshape(-1, TRAJECTORY_LEN, STATE_DIM)
dataset_actions = dataset_actions.reshape(-1, TRAJECTORY_LEN, ACTION_DIM)
dataset_terminals = dataset_terminals
dataset_timeouts = dataset_timeouts.reshape(-1, TRAJECTORY_LEN)


num_trajectories, len_trajectory, obs_dim = dataset_trajectories.shape


# [7]:


dataset_mean = dataset_trajectories.mean([0, 1])
dataset_std = dataset_trajectories.std([0, 1])


def normalize_dataset_coords(dataset_):
    return dataset_

def denormalize_dataset_coords(dataset_):
    return dataset_

dataset_trajectories = normalize_dataset_coords(dataset_trajectories)
dataset_trajectories_cuda = dataset_trajectories.to(device)


# [8]:


class VelocityRewardFunctionCheetah:
    def __init__(self):
        pass    
    
    def _sigmoids(self, x, value_at_1, sigmoid):
        if sigmoid == 'linear':
            scale = 1-value_at_1
            scaled_x = x*scale
            return np.where(abs(scaled_x) < 1, 1 - scaled_x, 0.0)
        else:
            raise NotImplementedError
    
    def tolerance(self, x, lower, upper, margin=0.0, sigmoid='linear', value_at_margin=0):
        in_bounds = np.logical_and(lower <= x, x <= upper)
        d = np.where(x < lower, lower - x, x - upper) / margin
        value = np.where(in_bounds, 1.0, self._sigmoids(d, value_at_margin, sigmoid))
        return value
    
    def compute_reward(self, states, params):
        
        if params != 'flip':
            if isinstance(params, int):
                params = torch.full((*states.shape[:-1], 1), fill_value=params)
            
            assert len(states.shape) == len(params.shape), states.shape # (batch_size, obs_dim) OR (batch_size, num_pairs, obs_dim)

            horizontal_velocity = states[..., [0]]
            sign_of_param = np.sign(params)
            horizontal_velocity = horizontal_velocity * sign_of_param
            rew = self.tolerance(horizontal_velocity,
                                lower=np.abs(params),
                                upper=float('inf'),
                                margin=np.abs(params),
                                value_at_margin=0,
                                sigmoid='linear')
            
        else: # Flip
            _SPIN_SPEED = 5
            angmomentum = states[..., [1]]
            rew = self.tolerance(angmomentum,
                                lower=_SPIN_SPEED,
                                upper=float('inf'),
                                margin=_SPIN_SPEED,
                                value_at_margin=0,
                                sigmoid='linear')

        return torch.tensor(rew[..., 0], dtype=torch.float32)
    
    
    

class VelocityRewardFunctionWalker:
    def __init__(self):
        pass    

    def _sigmoids(self, x, value_at_1, sigmoid):
        if sigmoid == 'gaussian':
            scale = np.sqrt(-2 * np.log(value_at_1))
            return np.exp(-0.5 * (x*scale)**2)

        elif sigmoid == 'linear':
            scale = 1-value_at_1
            scaled_x = x*scale
            return np.where(abs(scaled_x) < 1, 1 - scaled_x, 0.0)
    
    def tolerance(self, x, lower, upper, margin=0.0, sigmoid='gaussian', value_at_margin=0.1):
        in_bounds = np.logical_and(lower <= x, x <= upper)
        d = np.where(x < lower, lower - x, x - upper) / margin
        value = np.where(in_bounds, 1.0, self._sigmoids(d, value_at_margin, sigmoid))
        return torch.tensor(value)
    
    def compute_reward(self, states, params):
        
        if params != 'flip':
            if isinstance(params, int) or isinstance(params, float):
                params = torch.full((*states.shape[:-1], 1), fill_value=params)
            
            assert len(states.shape) == len(params.shape), states.shape # (batch_size, obs_dim) OR (batch_size, num_pairs, obs_dim)

            _STAND_HEIGHT = 1.2
            horizontal_velocity = states[..., [0]]
            torso_upright = states[..., [1]]
            torso_height = states[..., [2]]
            standing = self.tolerance(torso_height, lower=_STAND_HEIGHT, upper=float('inf'), margin=_STAND_HEIGHT/2)
            upright = (1 + torso_upright) / 2
            stand_reward = (3*standing + upright) / 4
            move_reward = self.tolerance(horizontal_velocity,
                                            lower=params,
                                            upper=float('inf'),
                                            margin=params/2,
                                            value_at_margin=0.5,
                                            sigmoid='linear')
            # move_reward[params == 0] = stand_reward[params == 0]
            
        
        else: # flip
            _STAND_HEIGHT = 1.2
            _SPIN_SPEED = 5
            horizontal_velocity = states[..., [0]]
            torso_upright = states[..., [1]]
            torso_height = states[..., [2]]
            angmomentum = states[..., [3]]
            standing = self.tolerance(torso_height, lower=_STAND_HEIGHT, upper=float('inf'), margin=_STAND_HEIGHT/2)
            upright = (1 + torso_upright) / 2
            stand_reward = (3*standing + upright) / 4
            move_reward = self.tolerance(angmomentum,
                                            lower=_SPIN_SPEED,
                                            upper=float('inf'),
                                            margin=_SPIN_SPEED,
                                            value_at_margin=0.5,
                                            sigmoid='linear')
            # move_reward[params == 0] = stand_reward[params == 0]
        
        rew = stand_reward * (5*move_reward + 1) / 6    
                
        return torch.tensor(rew[..., 0], dtype=torch.float32)
    
    



    
if ENV_NAME == 'cheetah':
    velocity_reward_function = VelocityRewardFunctionCheetah()
    benchmarks = [
        (velocity_reward_function.compute_reward, 'vel10Back', -10),
        (velocity_reward_function.compute_reward, 'vel2Back', -2),
        (velocity_reward_function.compute_reward, 'vel2', 2),
        (velocity_reward_function.compute_reward, 'vel10', 10),
        (velocity_reward_function.compute_reward, 'flip', 'flip'),
    ]
if ENV_NAME == 'walker':
    velocity_reward_function = VelocityRewardFunctionWalker()
    benchmarks = [
        (velocity_reward_function.compute_reward, 'vel0.1', 0.1),
        (velocity_reward_function.compute_reward, 'vel1', 1),
        (velocity_reward_function.compute_reward, 'vel4', 4),
        (velocity_reward_function.compute_reward, 'vel8', 8),
        (velocity_reward_function.compute_reward, 'flip', 'flip'),
    ]
if ENV_NAME == 'antmaze':
    from utils.antmaze_benchmark import VelocityRewardFunction, SimplexRewardFunction, TestRewPath, TestRewLoop, TestRewMatrixEdges, goal_reaching_reward
    velocity_reward_function = VelocityRewardFunction()
    simplex_reward_function = SimplexRewardFunction(num_simplex=10)
    benchmarks = [
        (goal_reaching_reward, 'goal_bottom', np.array([28, 0])),
        (goal_reaching_reward, 'goal_left', np.array([0, 15])),
        (goal_reaching_reward, 'goal_top', np.array([35, 24])),
        (goal_reaching_reward, 'goal_center', np.array([12, 24])), 
        (goal_reaching_reward, 'goal_right', np.array([33, 16])),
        (velocity_reward_function.compute_reward, 'vel_left', [-1, 0]),
        (velocity_reward_function.compute_reward, 'vel_up', [0, 1]),
        (velocity_reward_function.compute_reward, 'vel_down', [0, -1]),
        (velocity_reward_function.compute_reward, 'vel_right', [1, 0]),
        (simplex_reward_function.compute_reward, 'simplex_1', 1),
        (simplex_reward_function.compute_reward, 'simplex_2', 2),
        (simplex_reward_function.compute_reward, 'simplex_3', 3),
        (simplex_reward_function.compute_reward, 'simplex_4', 4),
        (simplex_reward_function.compute_reward, 'simplex_5', 5),
        (TestRewPath().compute_reward, 'path_center', None),
        (TestRewLoop().compute_reward, 'path_loop', None),
        (TestRewMatrixEdges().compute_reward, 'path_edges', None)
    ]


# [556]:


@dataclasses.dataclass
class Dataset:
    trajectories: torch.Tensor
    actions: torch.Tensor
    terminals: torch.Tensor
    timeouts: torch.Tensor

dataset = Dataset(
        trajectories=dataset_trajectories,
        actions=dataset_actions,
        terminals=dataset_terminals,
        timeouts=dataset_timeouts
    )


def get_iql_training_data(dataset:Dataset, batch_size):

    num_trajectories, len_trajectory, obs_dim = dataset.trajectories.shape

    trajectory_idx = torch.randint(0, num_trajectories, (batch_size,))
    state_idx = torch.randint(0, len_trajectory, (batch_size,)) % (len_trajectory - 1)

    states = dataset.trajectories[trajectory_idx, state_idx].reshape(batch_size, obs_dim)
    next_states = dataset.trajectories[trajectory_idx, state_idx+1].reshape(batch_size, obs_dim)
    actions = dataset.actions[trajectory_idx, state_idx].reshape(batch_size, -1)
    timeout = dataset.timeouts[trajectory_idx, state_idx+1].reshape(batch_size, -1).float()
    
    aux_ = aux[trajectory_idx, state_idx].reshape(batch_size, -1)
    aux_ = torch.tensor(aux_)
    
    # return {
    #     'states': states.to(device),
    #     'actions': actions.to(device),
    #     'next_states': next_states.to(device),
    # }
    
    return (
        states.to(device), actions.to(device), next_states.to(device), timeout.to(device), aux_
    )


# [630]:


def timestep2obs(timestep):
    obs = np.concatenate([v if len(v.shape) != 0 else v.reshape(-1) for k, v in timestep.observation.items()])
    return obs


def run_test_dmc(env, agent, benchmarks, benchmark_id, num_evals):


    benchmark_reward_function, benchmark_test_label, benchmark_param = benchmarks[benchmark_id]

    produced_trajectories = []
    produced_trajectories_physics = []
    
    for _ in range(num_evals):
        

        # _, encode_obs, _ = sample_reward_function_fre(batch_size=1, num_random_samples=num_eval_anchors)
        nb_eval_samples = 10000
        eval_obs, _, _, _, eval_aux = get_iql_training_data(dataset, batch_size=nb_eval_samples)
        benchmark_reward_function, _, benchmark_param = benchmarks[benchmark_id]
        eval_reward = benchmark_reward_function(eval_aux, benchmark_param).to(device)

        # meta = agent.infer_meta_from_obs_and_rewards(eval_obs, eval_reward)
        
        z = sf_agent.g(eval_obs).T @ eval_reward
        z = F.normalize(z, dim=-1)
        
        eval_obs, eval_reward
            
            
        timestep = env.reset()        
        state = timestep2obs(timestep)
    
        produced_trajectory = []   
        produced_trajectory_physics = [] 


        for step in tqdm(range(1000)):
            
            physics = env.physics.get_state()
            
            produced_trajectory_physics.append(physics)
            
            
            if state.shape[-1] == 24: # walker
                horizontal_velocity = env.physics.horizontal_velocity()
                torso_upright = env.physics.torso_upright()
                torso_height = env.physics.torso_height()
                angmomentum = env.physics.named.data.subtree_angmom['torso'][1]
                aux = np.array([horizontal_velocity, torso_upright, torso_height, angmomentum])

            elif state.shape[-1] == 17: # cheetah:
                horizontal_velocity = env.physics.speed()
                angmomentum = env.physics.named.data.subtree_angmom['torso'][1]
                aux = np.array([horizontal_velocity, angmomentum])
            
            observation_aux = np.concatenate([state, aux])
            
            produced_trajectory.append(observation_aux)
            
            
            with torch.no_grad():
                tensor_state = torch.tensor(state).reshape(-1).to(device).float()
                action = agent.actor_forward(tensor_state, z).mean.cpu().numpy()

            timestep = env.step(action)
            
            next_state = timestep2obs(timestep)
            # state = normalize_dataset_coords(next_state)
            state = next_state
            
            
        produced_trajectory = np.stack(produced_trajectory)
        produced_trajectory_physics = np.stack(produced_trajectory_physics)
                
        produced_trajectories.append(produced_trajectory)
        produced_trajectories_physics.append(produced_trajectory_physics)
    
    produced_trajectories = np.stack(produced_trajectories)
    produced_trajectories_physics = np.stack(produced_trajectories_physics)
    

    return produced_trajectories, produced_trajectories_physics, z






def run_benchmark(env, agent, benchmarks, num_evals):
    fig, axs = plt.subplots(len(benchmarks), 3, figsize=(15, len(benchmarks)*4))

    
    all_produced_trajectories = []
    
    for benchmark_id in range(len(benchmarks)):
        
        benchmark_reward_function, benchmark_test_label, benchmark_param = benchmarks[benchmark_id]
        print(benchmark_test_label)
        
        if ENV_NAME in ['cheetah', 'walker']:
            
            produced_trajectory, produced_trajectory_physics, z = run_test_dmc(
                env, agent, benchmarks, benchmark_id=benchmark_id, num_evals=num_evals
            )
            
            if ENV_NAME == 'cheetah':
                produced_trajectory_aux = produced_trajectory[..., -2:]
            elif ENV_NAME == 'walker':
                produced_trajectory_aux = produced_trajectory[..., -4:]
                

        
        eval_obs, _, _, _, eval_aux = get_iql_training_data(dataset, batch_size=10_000)
        
        with torch.no_grad():
            # Compute latent variable z:

            expectation_obs = eval_obs
            exp_B = sf_agent.g(expectation_obs)
            exp_B = F.normalize(exp_B, dim=0)
            C = (exp_B.T @ exp_B)
            inv_C = torch.inverse(C)
            B = sf_agent.g(eval_obs)
            eval_rewards = ((B @ inv_C) @ z.unsqueeze(-1)).flatten().cpu()
                
        real_eval_rewards = benchmark_reward_function(eval_aux, benchmark_param).cpu()
        eval_obs = eval_obs.cpu()

        if ENV_NAME == 'cheetah':
            axs[benchmark_id, 0].scatter(eval_obs[..., 8], eval_aux[..., 0], c=real_eval_rewards)
            axs[benchmark_id, 1].scatter(eval_obs[..., 8], eval_aux[..., 0], c=eval_rewards)
            axs[benchmark_id, 2].scatter(
                produced_trajectory_physics[..., 0],
                torch.arange(1000).unsqueeze(1).repeat(1, num_evals).T,
                c='red', s=1
            )
            axs[benchmark_id, 2].set_xlim([-25, 25])
        elif ENV_NAME == 'walker':
            axs[benchmark_id, 0].scatter(eval_obs[..., 16], eval_obs[..., 24], c=real_eval_rewards)
            axs[benchmark_id, 1].scatter(eval_obs[..., 16], eval_obs[..., 24], c=eval_rewards)
            axs[benchmark_id, 2].scatter(
                produced_trajectory_physics[..., 1],
                torch.arange(1000).unsqueeze(1).repeat(1, num_evals).T,
                c='red', s=1
            )

            
        axs[benchmark_id, 0].set_title(f'{benchmark_test_label}')
        axs[benchmark_id, 1].set_title(f'Reconstructed Reward Function')
        axs[benchmark_id, 2].set_title(f'Agent Trajectory')
        
        all_produced_trajectories.append(produced_trajectory)
    
        
    # np.savez(f"{args.MODEL_SAVE_FOLDER}/all_produced_trajectories", all_produced_trajectories)
    if (training_steps < 10) or steps % (training_steps // 10) == 0 or (steps == training_steps):
        plt.savefig(f"{LOGS_FOLDER}/benchmark-steps:{steps}.png")
    plt.close()
    
    
    all_produced_trajectories = np.stack(all_produced_trajectories)
    state_dim = all_produced_trajectories.shape[-1]
    
    benchmark_rewards = []
    
    for benchmark_id in range(len(benchmarks)):

        benchmark_reward_function, benchmark_test_label, benchmark_param = benchmarks[benchmark_id]
        
        if ENV_NAME == 'cheetah':
            trajectory_states_aux = torch.tensor(all_produced_trajectories[benchmark_id, ..., -2:]).reshape(1, -1, 2)
        elif ENV_NAME == 'walker':
            trajectory_states_aux = torch.tensor(all_produced_trajectories[benchmark_id, ..., -4:]).reshape(1, -1, 4) 
            
        trajectory_states_rewards = benchmark_reward_function(trajectory_states_aux, benchmark_param).float()
        trajectory_states_rewards = trajectory_states_rewards.reshape(
            all_produced_trajectories.shape[1],
            all_produced_trajectories.shape[2],
        )
        trajectory_rewards = trajectory_states_rewards.sum(dim=-1)
        
        if ENV_NAME == 'antmaze' and 'goal' in benchmark_test_label: 
            trajectory_rewards = torch.where(trajectory_rewards != -all_produced_trajectories.shape[2], 1., 0.)
            
        print(benchmark_test_label, ':')
        print('\tRewards:', trajectory_rewards.tolist())
        print('\tmean:', trajectory_rewards.mean().item())
        print('\tstd:', trajectory_rewards.std().item())
        
        benchmark_rewards.append(trajectory_rewards.mean().item())

    benchmark_rewards = np.array(benchmark_rewards)
    
    return benchmark_rewards




# # Autoencoders:

# [631]:


import typing as tp
import math


def _nl(name: str, dim: int) -> tp.List[nn.Module]:
    """Returns a non-linearity given name and dimension"""
    if name == "irelu":
        return [nn.ReLU(inplace=True)]
    if name == "relu":
        return [nn.ReLU()]
    if name == "ntanh":
        return [nn.LayerNorm(dim), nn.Tanh()]
    if name == "layernorm":
        return [nn.LayerNorm(dim)]
    if name == "tanh":
        return [nn.Tanh()]
    if name == "L2":
        return [_L2(dim)]
    raise ValueError(f"Unknown non-linearity {name}")


def mlp(*layers: tp.Sequence[tp.Union[int, str]]) -> nn.Sequential:
    """Provides a sequence of linear layers and non-linearities
    providing a sequence of dimension for the neurons, or name of
    the non-linearities
    Eg: mlp(10, 12, "relu", 15) returns:
    Sequential(Linear(10, 12), ReLU(), Linear(12, 15))
    """
    assert len(layers) >= 2
    sequence: tp.List[nn.Module] = []
    assert isinstance(layers[0], int), "First input must provide the dimension"
    prev_dim: int = layers[0]
    for layer in layers[1:]:
        if isinstance(layer, str):
            sequence.extend(_nl(layer, prev_dim))
        else:
            assert isinstance(layer, int)
            sequence.append(nn.Linear(prev_dim, layer))
            prev_dim = layer
    return nn.Sequential(*sequence)



class SuccessorFeatures(nn.Module):
    def __init__(self, f_input_dim, f_output_dim, g_input_dim, g_output_dim, ):
        super(SuccessorFeatures, self).__init__()
        
        hidden_dim = 1024
        
        self.f = mlp(f_input_dim, hidden_dim, "ntanh", 
                        hidden_dim, "irelu",
                        hidden_dim, "irelu",
                        f_output_dim,
                    )
        
        self.g = mlp(g_input_dim, hidden_dim, "ntanh", hidden_dim, "relu", g_output_dim)
        
        self.target_f = mlp(f_input_dim, hidden_dim, "ntanh", 
                        hidden_dim, "irelu",
                        hidden_dim, "irelu",
                        f_output_dim,
                    )
        self.target_f.load_state_dict(self.f.state_dict())
        
        self.target_g = mlp(g_input_dim, hidden_dim, "ntanh", hidden_dim, "relu", g_output_dim)
        self.target_g.load_state_dict(self.g.state_dict())
        
        
        self.optimizer = torch.optim.Adam([
                {'params': self.f.parameters()},  # type: ignore
                {'params': self.g.parameters()}
            ],
            lr=1e-4
        )

def soft_update_params(net, target_net, tau) -> None:
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)
    
def orth_loss_fn(z):
    cov = z.T @ z
    id = torch.eye(50, dtype=torch.float32, device=device)
    orht_loss = (cov - id).pow(2).mean() + (torch.inverse(cov) - id).pow(2).mean() 
    return orht_loss


def autoencoder_loss(ae_agent, obs):
    z = ae_agent.g(obs)
    pred_obs = ae_agent.f(z)
    
    ae_loss = (obs - pred_obs).pow(2).mean()
    orth_loss = orth_loss_fn(z)
    
    loss = ae_loss + orth_loss
    return loss


def LRA_P(sf_agent, obs, action, next_obs):
    f = sf_agent.f(torch.concat((obs, action), dim=-1)) 
    g = sf_agent.g(next_obs)
    P = torch.einsum("sd, td -> st" , f, g)
    I = torch.eye(*P.size(), device=device)
    off_diag = ~I.bool()
    sf_loss = -2 * P.diag().mean() + P[off_diag].pow(2).mean()
    
    loss = sf_loss + orth_loss_fn(g)
    
    return loss

def trans(sf_agent, obs, action, next_obs):
    g = sf_agent.g(obs)
    f = sf_agent.f(torch.concat([g, action], dim=-1))
    
    loss = (f - next_obs).pow(2).mean() + orth_loss_fn(g)
    
    return loss


def LRA_SF(sf_agent, obs, action, next_obs):
    discount = 0.99
    g = sf_agent.g(next_obs)
    f = sf_agent.f(obs)
    SR = torch.einsum('sd, td -> st',f ,g)
    with torch.no_grad():
        target_g = sf_agent.target_g(next_obs)
        target_f = sf_agent.target_f(next_obs)
        target_SR = torch.einsum("sd, td -> st", target_f, target_g)
    
    I = torch.eye(* SR.size(), device = SR.device)
    off_diag = ~ I.bool()
    loss = - 2 * SR.diag().mean() + ( SR - discount * target_SR.detach())[off_diag].pow(2).mean()
    
    loss += orth_loss_fn(g)
    
    return loss


# [632]:


Z_DIM = 50

sf_agent = SuccessorFeatures(
    f_input_dim=STATE_DIM, f_output_dim=Z_DIM, 
    g_input_dim=STATE_DIM, g_output_dim=Z_DIM,
).to(device)
sf_agent.load_state_dict(torch.load('models/sf_lra_sf.pth'))


# # Training a policy:

# [650]:


class TruncatedNormal(torch.distributions.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6) -> None:
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x) -> torch.Tensor:
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()) -> torch.Tensor:  # type: ignore
        shape = self._extended_shape(sample_shape)
        eps = torch.distributions.utils._standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


class DDPG(nn.Module):
    def __init__(self, obs_dim, z_dim, action_dim):
        super(DDPG, self).__init__()
        
        hidden_dim = 1024
        
        self.actor = mlp(obs_dim + z_dim, hidden_dim, "ntanh", 
                        hidden_dim, "irelu",
                        hidden_dim, "irelu",
                        action_dim,
                    )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        
        
        self.Q1 = mlp(obs_dim + action_dim + z_dim, hidden_dim, "ntanh", 
                        hidden_dim, "irelu",
                        hidden_dim, "irelu",
                        1,
                    )
        
        self.Q2 = mlp(obs_dim + action_dim + z_dim, hidden_dim, "ntanh", 
                        hidden_dim, "irelu",
                        hidden_dim, "irelu",
                        1,
                    )
        
        self.Q_optimizer = torch.optim.Adam([
                {'params': self.Q1.parameters()},  # type: ignore
                {'params': self.Q2.parameters()}
            ], lr=1e-4)
        
        
        self.Q1_target = mlp(
            obs_dim + action_dim + z_dim, hidden_dim, "ntanh",
            hidden_dim, "irelu",
            hidden_dim, "irelu",
            1,
        )
        self.Q2_target = mlp(
            obs_dim + action_dim + z_dim, hidden_dim, "ntanh",
            hidden_dim, "irelu",
            hidden_dim, "irelu",
            1,
        )

        self.Q1_target.load_state_dict(self.Q1.state_dict())
        self.Q2_target.load_state_dict(self.Q2.state_dict())
        
        
        
        
    def actor_forward(self, obs, z, std=0.2):
        x = torch.concat([obs, torch.zeros_like(z)], dim=-1)
        mu = self.actor(x)
        
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = TruncatedNormal(mu, std)
        
        return dist


# [651]:


agent = DDPG(obs_dim=STATE_DIM, z_dim=Z_DIM, action_dim=ACTION_DIM).to(device)


# [654]:


actor_losses = []
rewards_list = []
rewards_logs = np.zeros((len(benchmarks), 0),)



# [659]:


for steps in tqdm(range(1_000_000)):

    batch_size = 1024

    z = torch.normal(0, 1, size=(batch_size, Z_DIM), device=device)
    z = F.normalize(z, dim=-1)
    z = torch.zeros_like(z)
    obs, action, next_obs, terminals, obs_aux = get_iql_training_data(dataset, batch_size=batch_size)
    with torch.no_grad():
        # reward = (sf_agent.g(obs) * z).sum(dim=-1).unsqueeze(-1)
        benchmark_reward_function, _, benchmark_param = benchmarks[2]
        reward = benchmark_reward_function(obs_aux, benchmark_param).to(device).float().unsqueeze(-1)

    # ------------------------------------------------------------------
    # 2) Compute target Q values with next actions
    # ------------------------------------------------------------------
    with torch.no_grad():
        # Sample next action from actor
        dist = agent.actor_forward(next_obs, z)
        next_action = dist.sample(clip=0.3)

        # Q-targets: min over twin critics
        s_a_z = torch.cat([next_obs, next_action, z], dim=-1)
        q_target = torch.min(agent.Q1_target(s_a_z), agent.Q2_target(s_a_z))

        y = reward + 0.99 * (1 - terminals) * q_target
        
        
    # ------------------------------------------------------------------
    # 3) Critic update
    # ------------------------------------------------------------------
    q1_pred = agent.Q1(torch.cat([obs, action, z], dim=-1))
    q2_pred = agent.Q2(torch.cat([obs, action, z], dim=-1))

    critic_loss = F.mse_loss(q1_pred, y) + F.mse_loss(q2_pred, y)

    agent.Q_optimizer.zero_grad()
    critic_loss.backward()
    agent.Q_optimizer.step()


    soft_update_params(agent.Q1, agent.Q1_target, tau=0.001)
    soft_update_params(agent.Q2, agent.Q2_target, tau=0.001)

    # ------------------------------------------------------------------
    # 4) Actor update
    # ------------------------------------------------------------------

    z = torch.normal(0, 1, size=(batch_size, Z_DIM), device=device)
    z = F.normalize(z, dim=-1)
    z = torch.zeros_like(z)
    obs, _, _, _, _ = get_iql_training_data(dataset, batch_size=batch_size)

    dist = agent.actor_forward(obs, z)
    action = dist.sample(clip=0.3)
    s_a_z = torch.cat([obs, action, z], dim=-1)
    Q = torch.min(agent.Q1(s_a_z), agent.Q2(s_a_z)) 

    actor_loss = -Q.mean()

    agent.actor_optimizer.zero_grad()
    actor_loss.backward()
    agent.actor_optimizer.step()
    
    
    actor_losses.append(actor_loss.item())
    
    if steps % 1000 == 0:
        clear_output(True)
        benchmark_rewards = run_benchmark(env, agent, benchmarks, 1)
        
        rewards_logs = np.concatenate((rewards_logs, benchmark_rewards.reshape(-1, 1)), axis=-1)
        # print(rewards_logs)
        num_cols = 4
        num_rows = len(benchmarks)//num_cols + int(len(benchmarks) % num_cols != 0)
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols*6, num_rows*5))
        axs = axs.flatten()
        
        for i in range(len(benchmarks)):
            _, benchmark_test_label, _ = benchmarks[i]
            axs[i].plot(rewards_logs[i])
            axs[i].set_title(benchmark_test_label)
        
        plt.savefig(f"{LOGS_FOLDER}/rewards.png")
        plt.close()
        
    
    # break

