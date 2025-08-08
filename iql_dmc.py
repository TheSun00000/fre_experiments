#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import numpy as np
import random

from tqdm import tqdm
from IPython.display import clear_output

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

from dm_control import suite

import os
from datetime import datetime




# In[2]:


NUM_TRAJECTORIES = 10000
TRAJECTORY_LEN = 1000
ENV_NAME = 'walker' # walker | cheetah
POLICY_EXTRACTION_METHOD = 'ddpg' # awr |  ddpg 
VALUE_STEPS = 1000000
POLICY_STEPS = 1000000


now = datetime.now()
date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")

exp_name = f'iql-{POLICY_EXTRACTION_METHOD}-{ENV_NAME}'

print(exp_name)
    
LOGS_FOLDER = f'./logs/{date_time_str}_{exp_name}'

os.makedirs(LOGS_FOLDER)


# In[3]:


if ENV_NAME == 'walker':
    STATE_DIM = 27
    ACTION_DIM = 6
    AUX_DIM = 3
    env = suite.load(
        domain_name='walker',
        task_name='run',
        environment_kwargs=dict(flat_observation=True)
    )
    dataset = np.load('datasets/walker_rnd.npy', allow_pickle=True).item()
    aux = np.load('datasets/aux_walker.npy')
    dataset['observations'] = np.concatenate((dataset['observations'], aux.reshape(10000, 1000, 3)), axis=-1)


elif ENV_NAME == 'cheetah':
    STATE_DIM = 18
    ACTION_DIM = 6
    AUX_DIM = 1
    env = suite.load(
        domain_name='cheetah',
        task_name='run',
        environment_kwargs=dict(flat_observation=True)
    )
    dataset = np.load('datasets/cheetah_rnd.npy', allow_pickle=True).item()
    aux = np.load('datasets/aux_cheetah.npy')
    dataset['observations'] = np.concatenate((dataset['observations'], aux.reshape(10000, 1000, 1)), axis=-1)

# In[4]:


dataset_trajectories = torch.tensor(dataset['observations']).float()
dataset_trajectories = dataset_trajectories
# dataset_trajectories = torch.concatenate((dataset_trajectories[..., [0, 1]], dataset_trajectories[..., [15, 16]]), dim=-1)

dataset_actions = torch.tensor(dataset['actions']).float()
dataset_terminals = torch.tensor(dataset['terminals']).float()
dataset_timeouts = torch.zeros(NUM_TRAJECTORIES, TRAJECTORY_LEN).bool()
dataset_timeouts[:, -1] = True

dataset_goals = torch.tensor(dataset['infos/goal']).float()



dataset_trajectories = dataset_trajectories.reshape(-1, TRAJECTORY_LEN, STATE_DIM)
dataset_actions = dataset_actions.reshape(-1, TRAJECTORY_LEN, ACTION_DIM)
dataset_terminals = dataset_terminals
dataset_timeouts = dataset_timeouts.reshape(-1, TRAJECTORY_LEN)


num_trajectories, len_trajectory, obs_dim = dataset_trajectories.shape


# In[5]:


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
        
        if isinstance(params, int) or isinstance(params, float):
            params = torch.full((*states.shape[:-1], 1), fill_value=params)
        
        assert len(states.shape) == len(params.shape), states.shape # (batch_size, obs_dim) OR (batch_size, num_pairs, obs_dim)

        _STAND_HEIGHT = 1.2
        horizontal_velocity = states[..., 24:25]
        torso_upright = states[..., 25:26]
        torso_height = states[..., 26:27]
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
        rew = stand_reward * (5*move_reward + 1) / 6
        # rew = (5*move_reward + 1) / 6
        
        return torch.tensor(rew[..., 0])


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
        
        if isinstance(params, int):
            params = torch.full((*states.shape[:-1], 1), fill_value=params)
        
        assert len(states.shape) == len(params.shape), states.shape # (batch_size, obs_dim) OR (batch_size, num_pairs, obs_dim)

        horizontal_velocity = states[..., 17:18]
        sign_of_param = np.sign(params)
        horizontal_velocity = horizontal_velocity * sign_of_param
        rew = self.tolerance(horizontal_velocity,
                             lower=np.abs(params),
                             upper=float('inf'),
                             margin=np.abs(params),
                             value_at_margin=0,
                             sigmoid='linear')
        
        return torch.tensor(rew[..., 0], dtype=torch.float32)
    
  
if ENV_NAME == 'walker':
    velocity_reward_function = VelocityRewardFunctionWalker()

    def reward_function(state):
        return velocity_reward_function.compute_reward(state, 8)
    
elif ENV_NAME == 'cheetah':
    velocity_reward_function = VelocityRewardFunctionCheetah()

    def reward_function(state):
        return velocity_reward_function.compute_reward(state, 10)


# # IQL:

# In[14]:


import torch
import torch.nn as nn

class MLP(nn.Module):
    """Generic MLP with Mish activation and LayerNorm."""
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim

        # Add hidden layers with Mish activation and LayerNorm
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Mish())
            # layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim  # Update input size for next layer
        
        # Final output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ValueCritic(nn.Module):
    def __init__(self, obs_dim, hidden_dims):
        super().__init__()
        self.model = MLP(obs_dim, hidden_dims, output_dim=1)

    def forward(self, x):
        return self.model(x)


class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims):
        super().__init__()

        self.model1 = MLP(obs_dim + action_dim, hidden_dims, output_dim=1)
        self.model2 = MLP(obs_dim + action_dim, hidden_dims, output_dim=1)

    def forward(self, x, action):
        x = torch.cat((x, action), dim=-1)
        
        q1 = self.model1(x)
        q2 = self.model2(x)
        
        return q1, q2



class Actor(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dims, init_std=0.0):
        super().__init__()
        self.model = MLP(input_dim, hidden_dims, output_dim=action_dim)  # MLP only predicts mean
        
        # Learnable log standard deviation (initialized to `init_std`)
        self.log_std = nn.Parameter(torch.ones(action_dim) * init_std)

    def forward(self, x, temperature=1.0):
        mean = self.model(x)  # Predict action mean

        log_std = torch.clip(self.log_std, -20, 2)
        mean = torch.clip(mean, -5, 5)

        return torch.distributions.Normal(
            mean, 
            torch.exp(log_std)*temperature
        )


from torch.optim.lr_scheduler import CosineAnnealingLR
import copy


class IQL(nn.Module):
    def __init__(self, state_dim, action_dim, w_dim=128):
        super(IQL, self).__init__()
        self.obs_len = state_dim
                
        self.critic = Critic(w_dim + state_dim, action_dim, hidden_dims=[512, 512, 512])
        self.target_critic = copy.deepcopy(self.critic)
        for param in self.target_critic.parameters():
            param.requires_grad = False
        
        self.value = ValueCritic(w_dim + state_dim, hidden_dims=[512, 512, 512])        
        self.actor = Actor(w_dim + state_dim, action_dim, hidden_dims=[512, 512, 512])
        
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=0.003)
        self.value_optim = torch.optim.Adam(self.value.parameters(), lr=0.003)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=0.003)
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optim, POLICY_STEPS)
        
        
    def get_value(self, obs):
        w_and_obs = torch.concatenate([obs], dim=-1)
        return self.value(w_and_obs)

    def get_critic(self, obs, actions):
        w_and_obs = torch.concatenate([obs], dim=-1)
        return self.critic(w_and_obs, actions)
    
    def get_target_critic(self, obs, actions):
        w_and_obs = torch.concatenate([obs], dim=-1)
        return self.target_critic(w_and_obs, actions)

    def get_actor(self, obs, temperature=1.0):
        w_and_obs = torch.concatenate([obs], dim=-1)
        return self.actor(w_and_obs, temperature)
    
    
    
# def update_target_critic(critic, target_critic, tau):

#     critic_state_dict = critic.state_dict()
#     target_critic_state_dict = target_critic.state_dict()

#     for key in critic_state_dict:
#         target_critic_state_dict[key] = tau * critic_state_dict[key] + (1 - tau) * target_critic_state_dict[key]

#     target_critic.load_state_dict(target_critic_state_dict)
    
    
def update_target_critic(source, target, alpha):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1. - alpha).add_(source_param.data, alpha=alpha)
    
    
def expectile_loss(adv, diff, expectile=0.7):
    weight = torch.where(
        adv >= 0, 
        torch.tensor(expectile, dtype=adv.dtype), 
        torch.tensor(1 - expectile, dtype=adv.dtype)
    )
    return weight * (diff ** 2)


def smooth_and_downsample(losses, smoothing=0.9, max_points=100):
    if not losses: return []
    smoothed = [losses[0]]
    for loss in losses[1:]:
        smoothed.append(smoothing * smoothed[-1] + (1 - smoothing) * loss)
    downsample = max(1, len(smoothed) // max_points)  # Ensure at least 1
    return smoothed[::downsample]


# In[21]:


def get_iql_training_data(batch_size):

    trajectory_idx = torch.randint(0, num_trajectories, (batch_size,))
    state_idx = torch.randint(0, len_trajectory, (batch_size,)) % (len_trajectory - 1)

    states = dataset_trajectories[trajectory_idx, state_idx].reshape(batch_size, obs_dim)
    next_states = dataset_trajectories[trajectory_idx, state_idx+1].reshape(batch_size, obs_dim)
    actions = dataset_actions[trajectory_idx, state_idx].reshape(batch_size, ACTION_DIM)
    masks = ~dataset_timeouts[trajectory_idx, state_idx+1].reshape(batch_size, 1)
    
    # rewards = reward_function(states).unsqueeze(-1)
    rewards = reward_function(next_states).unsqueeze(-1)
    
    if ENV_NAME == 'walker':
        states = states[..., :24]
        next_states = next_states[..., :24]
    elif ENV_NAME == 'cheetah':
        states = states[..., :17]
        next_states = next_states[..., :17]
    
    return {
        'states': states.to(device),
        'actions': actions.to(device),
        'next_states': next_states.to(device),
        'masks': masks.to(device),
        'rewards': rewards.to(device)
    }




# In[22]:




def timestep2obs(timestep):
    obs = np.concatenate([v if len(v.shape) != 0 else v.reshape(-1) for k, v in timestep.observation.items()])
    return obs

def run_test(iql_agent, num_evals):


    produced_trajectories = []
    produced_trajectories_physics = []
    
    for _ in range(num_evals):
        
            
            
        timestep = env.reset()        
        state = timestep2obs(timestep)
        # state = normalize_dataset_coords(state)
    
        produced_trajectory = []   
        produced_trajectory_physics = [] 


        # for step in tqdm(range(1000)):
        for step in range(1000):
            
            physics = env.physics.get_state()
            
            produced_trajectory_physics.append(physics)
            
            if ENV_NAME == 'walker':
                horizontal_velocity = env.physics.horizontal_velocity()
                torso_upright = env.physics.torso_upright()
                torso_height = env.physics.torso_height()
                aux = np.array([horizontal_velocity, torso_upright, torso_height])

            elif ENV_NAME == 'cheetah':
                horizontal_velocity = env.physics.speed()
                aux = np.array([horizontal_velocity])
            
            observation_aux = np.concatenate([state, aux])
            
            produced_trajectory.append(observation_aux)
            
            
            with torch.no_grad():
                tensor_state = torch.tensor(state).reshape(1, -1).to(device).float()
                                
                dist = iql_agent.get_actor(tensor_state)
                action = dist.loc.cpu()
                action = np.array(action[0]).clip(-1, 1)

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
    

    return produced_trajectories, produced_trajectories_physics




# In[44]:


iql_agent = IQL(state_dim=STATE_DIM - AUX_DIM, action_dim=ACTION_DIM, w_dim=0).to(device)
# iql_agent.load_state_dict(torch.load('shared_models/iql_agent_walker-rg.pth'))

actor_losses = []
v_losses, q_losses = [], []
mse_errors = []
stds = []
rewards = []


# In[48]:


config = {
    'expectile': 0.9,
    'temperature': 10.0,
    'discount': 0.99,
    'tau': 0.005,
    
    'bc_coefficient': 0.01
}

iql_batch_size = 1024


for timestep in tqdm(range(VALUE_STEPS+POLICY_STEPS)):
    
    batch = get_iql_training_data(
        batch_size=iql_batch_size, 
    )

        
    with torch.no_grad():
        
        target_q1, target_q2 = iql_agent.get_target_critic(batch['states'], batch['actions'])
        target_q1, target_q2 = target_q1.detach(), target_q2.detach()
        target_q = torch.minimum(target_q1, target_q2)
        next_v = iql_agent.get_value(batch['next_states']).detach()
    
    
    if timestep < VALUE_STEPS:
    
        # Value Loss: Update V towards expectile of min(q1, q2).
        
        v = iql_agent.get_value(batch['states'])
        adv = target_q - v
        v_loss = expectile_loss(adv, adv, config['expectile'])
        v_loss = v_loss.mean()

        iql_agent.value.zero_grad(set_to_none=True)
        v_loss.backward()
        iql_agent.value_optim.step()

        # Critic Loss. Update Q = r #############################
        targets = batch['rewards'] + config['discount'] * batch['masks'] * next_v

        q1, q2 = iql_agent.get_critic(batch['states'], batch['actions'])
        q_loss = ((q1 - targets).pow(2).mean() + (q2 - targets).pow(2).mean()) / 2
        q_loss = q_loss
        
        iql_agent.critic.zero_grad(set_to_none=True)
        q_loss.backward()
        iql_agent.critic_optim.step()

        # if timestep % 10 == 0:
        update_target_critic(iql_agent.critic, iql_agent.target_critic, config['tau'])

        value_loss = v_loss + q_loss
        value_info = {
            'v_loss': v_loss,
            'q_loss': q_loss,
            'v': v.mean(),
            'q': torch.minimum(q1, q2).mean(),
        }
        
        v_losses.append(v_loss.item())
        q_losses.append(q_loss.item())


    # Actor Loss ############################################

    else:    
        if POLICY_EXTRACTION_METHOD == 'awr':
            v = iql_agent.get_value(batch['states'])
            q1, q2 = iql_agent.get_critic(batch['states'], batch['actions'])
            q = torch.minimum(q1, q2)
            adv = q - v
            
            actions = batch['actions']
            exp_a = torch.exp(adv.detach() * config['temperature']).clamp(max=100)
            
            dist = iql_agent.get_actor(batch['states'])
            log_probs = dist.log_prob(actions)
            actor_loss = -(exp_a * log_probs).mean()
            
            
        elif POLICY_EXTRACTION_METHOD == 'ddpg':
            dist = iql_agent.get_actor(batch['states'])
            normalized_actions = torch.tanh(dist.loc)
            q1, q2 = iql_agent.get_critic(batch['states'], batch['actions'])
            q = (q1 + q2) / 2
            q_loss_ = -q.mean()
            log_probs = dist.log_prob(batch['actions'])
            bc_loss = -((config['bc_coefficient'] * log_probs)).mean()
            
            actor_loss = ((q_loss_ + bc_loss)).mean()
        
        iql_agent.actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        iql_agent.actor_optim.step()
        iql_agent.actor_lr_schedule.step()
        
        
        std = dist.stddev.mean()
        mse_error = ((dist.loc - batch['actions'])**2).mean()
        
        actor_info = {
            'actor_loss': actor_loss,
            'std': std,
            'adv': adv.mean(),
            'mse_error': mse_error,
        }
        
        actor_losses.append(actor_loss.item())
        mse_errors.append(mse_error.item())
        stds.append(std.item())

    ########################################################################################
    

    
    
    
    
    
    
    
    if timestep > VALUE_STEPS and timestep % 10000 == 0:
        
        produced_trajectory, produced_trajectory_physics = run_test(iql_agent, num_evals=20)
        r = reward_function(torch.tensor(produced_trajectory))
        r_mean = r.sum(dim=1).mean().item()
        r_std = r.sum(dim=1).std().item()
        rewards.append(r_mean)
        
        clear_output(True)
        fig, axs = plt.subplots(1, 6, figsize=(35, 5))
        
        axs[0].plot(smooth_and_downsample(actor_losses))
        # axs[1].set_ylim(0,max(actor_losses[-100:]))
        axs[0].set_title("Actor Loss")
        
        axs[1].plot(smooth_and_downsample(v_losses))
        # axs[1].set_ylim(0,max(v_losses[-100:]))
        axs[1].set_title("V Loss")
        
        axs[2].plot(smooth_and_downsample(q_losses))
        # axs[2].set_ylim(0,max(q_losses[-100:]))
        axs[2].set_title("Q Loss")
        
        axs[3].plot(smooth_and_downsample(mse_errors))
        # axs[3].set_ylim(0,max(mse_errors[-100:]))
        axs[3].set_title("MSE Errors")
        
        axs[4].plot(smooth_and_downsample(stds))
        axs[4].set_title("std")
        
        axs[5].plot(smooth_and_downsample(rewards))
        axs[5].set_title("rewards")
        
        # plt.show()
        
        plt.savefig(f"{LOGS_FOLDER}/iql_training.png")
        plt.close()
        
        
        
    # break


# In[26]:

