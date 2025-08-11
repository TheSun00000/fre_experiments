#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import numpy as np
import random

from tqdm import tqdm
from IPython.display import clear_output

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device


import gym
import d4rl



import os
from datetime import datetime



ENV_NAME = 'halfcheetah-medium-v2'



now = datetime.now()
date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")

exp_name = f'iql-{ENV_NAME}'

print(exp_name)
    
LOGS_FOLDER = f'./logs/{date_time_str}_{exp_name}'

os.makedirs(LOGS_FOLDER)




# In[ ]:


NUM_TRAJECTORIES = 2000
TRAJECTORY_LEN = 1000


env = gym.make(ENV_NAME)

file_path = 'datasets/halfcheetah_medium_expert-v2.hdf5'
if os.path.isfile(file_path):
    dataset = env.get_dataset(h5path=file_path)
else:
    dataset = env.get_dataset()
    

STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]


# In[ ]:


# In[ ]:


dataset_trajectories = torch.tensor(dataset['observations']).float()
dataset_rewards = torch.tensor(dataset['rewards']).float()
dataset_actions = torch.tensor(dataset['actions']).float()
dataset_terminals = torch.tensor(dataset['terminals']).float()
dataset_timeouts = torch.zeros(NUM_TRAJECTORIES, TRAJECTORY_LEN).bool()
dataset_timeouts[:, -1] = True



dataset_trajectories = dataset_trajectories.reshape(-1, TRAJECTORY_LEN, STATE_DIM)
dataset_rewards = dataset_rewards.reshape(-1, TRAJECTORY_LEN)
dataset_actions = dataset_actions.reshape(-1, TRAJECTORY_LEN, ACTION_DIM)
dataset_terminals = dataset_terminals
dataset_timeouts = dataset_timeouts.reshape(-1, TRAJECTORY_LEN)

num_trajectories, len_trajectory, obs_dim = dataset_trajectories.shape


# # IQL:

# In[ ]:


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
                
        self.critic = Critic(w_dim + state_dim, action_dim, hidden_dims=[256, 256])
        # self.target_critic = copy.deepcopy(self.critic)
        # for param in self.target_critic.parameters():
        #     param.requires_grad = False
        
        self.value = ValueCritic(w_dim + state_dim, hidden_dims=[256, 256])       
        self.target_value = copy.deepcopy(self.value)
        for param in self.target_value.parameters():
            param.requires_grad = False
             
        self.actor = Actor(w_dim + state_dim, action_dim, hidden_dims=[256, 256])
        
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=0.003)
        self.value_optim = torch.optim.Adam(self.value.parameters(), lr=0.003)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=0.003)
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optim, 100000)
        
        
    def get_value(self, obs):
        w_and_obs = torch.concatenate([obs], dim=-1)
        return self.value(w_and_obs)
    
    def get_target_value(self, obs):
        w_and_obs = torch.concatenate([obs], dim=-1)
        return self.target_value(w_and_obs)

    def get_critic(self, obs, actions):
        w_and_obs = torch.concatenate([obs], dim=-1)
        return self.critic(w_and_obs, actions)
    
    # def get_target_critic(self, obs, actions):
    #     w_and_obs = torch.concatenate([obs], dim=-1)
    #     return self.target_critic(w_and_obs, actions)

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
    
    
def expectile_loss(u, expectile=0.7):
    weight = torch.where(
        u >= 0, 
        torch.tensor(expectile, dtype=u.dtype), 
        torch.tensor(1 - expectile, dtype=u.dtype)
    )
    return weight * (u ** 2)


def smooth_and_downsample(losses, smoothing=0.9, max_points=100):
    if not losses: return []
    smoothed = [losses[0]]
    for loss in losses[1:]:
        smoothed.append(smoothing * smoothed[-1] + (1 - smoothing) * loss)
    downsample = max(1, len(smoothed) // max_points)  # Ensure at least 1
    return smoothed[::downsample]


# In[ ]:


def get_iql_training_data(batch_size):

    trajectory_idx = torch.randint(0, num_trajectories, (batch_size,))
    state_idx = torch.randint(0, len_trajectory, (batch_size,)) % (len_trajectory - 1)

    states = dataset_trajectories[trajectory_idx, state_idx].reshape(batch_size, obs_dim)
    next_states = dataset_trajectories[trajectory_idx, state_idx+1].reshape(batch_size, obs_dim)
    actions = dataset_actions[trajectory_idx, state_idx].reshape(batch_size, ACTION_DIM)
    masks = ~dataset_timeouts[trajectory_idx, state_idx+1].reshape(batch_size, 1)
    
    # rewards = reward_function(states).unsqueeze(-1)
    # rewards = reward_function(next_states).unsqueeze(-1)
    rewards = dataset_rewards[trajectory_idx, state_idx+1].reshape(batch_size, 1)
    
    ENV_NAME = 'walker'
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


def evaluate_policy(env, iql_agent, max_episode_steps=1000):
    obs = env.reset()
    total_reward = 0.
    for _ in range(max_episode_steps):
        with torch.no_grad():
            obs = torch.tensor(obs, device=device, dtype=torch.float)
            action = iql_agent.get_actor(obs).loc.cpu().numpy()
                # olicy.act(torchify(obs), deterministic=deterministic).cpu().numpy()
        next_obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
        else:
            obs = next_obs
    return total_reward


iql_agent = IQL(state_dim=STATE_DIM, action_dim=ACTION_DIM, w_dim=0).to(device)

actor_losses = []
v_losses, q_losses = [], []
mse_errors = []
stds = []
rewards = []



config = {
    'expectile': 0.9,
    'temperature': 10.0,
    'discount': 0.99,
    'tau': 0.005,
    
    'bc_coefficient': 0.01
}


iql_batch_size = 1024


for timestep in tqdm(range(1000000)):
    
    batch = get_iql_training_data(
        batch_size=iql_batch_size, 
    )

    
        
    # with torch.no_grad():
        
        # target_q1, target_q2 = iql_agent.get_target_critic(batch['states'], batch['actions'])
        # target_q1, target_q2 = target_q1.detach(), target_q2.detach()
        # target_q = torch.minimum(target_q1, target_q2)
        # next_v = iql_agent.get_value(batch['next_states']).detach()
    
    
    # Value Loss: Update V towards expectile of min(q1, q2).
    
    v = iql_agent.get_value(batch['states'])
    
    q1, q2 = iql_agent.get_critic(batch['states'], batch['actions'])
    q = torch.minimum(q1, q2).detach()
    
    adv = q - v
    v_loss = expectile_loss(adv, config['expectile'])
    v_loss = v_loss.mean()

    iql_agent.value.zero_grad(set_to_none=True)
    v_loss.backward()
    iql_agent.value_optim.step()
    

    # Critic Loss. Update Q = r #############################
    
    next_v = iql_agent.get_target_value(batch['states']).detach()
    
    targets = batch['rewards'] + config['discount'] * batch['masks'] * next_v

    q1, q2 = iql_agent.get_critic(batch['states'], batch['actions'])
    q_loss = ((q1 - targets).pow(2).mean() + (q2 - targets).pow(2).mean()) / 2
    
    iql_agent.critic.zero_grad(set_to_none=True)
    q_loss.backward()
    iql_agent.critic_optim.step()
    
    

    # if timestep % 10 == 0:
    update_target_critic(iql_agent.value, iql_agent.target_value, config['tau'])

    value_loss = v_loss + q_loss
    value_info = {
        'v_loss': v_loss,
        'q_loss': q_loss,
        'v': v.mean(),
        'q': torch.minimum(q1, q2).mean(),
    }


    # Actor Loss ############################################

    EPOLICY_EXTRACTION_METHON = 'awr' # awr |  ddpg 
    
    if EPOLICY_EXTRACTION_METHON == 'awr':
        v = iql_agent.get_value(batch['states'])
        q1, q2 = iql_agent.get_critic(batch['states'], batch['actions'])
        q = torch.minimum(q1, q2)
        adv = q - v
        
        actions = batch['actions']
        exp_a = torch.exp(adv.detach() * config['temperature']).clamp(max=100)
        
        dist = iql_agent.get_actor(batch['states'])
        log_probs = dist.log_prob(actions)
        actor_loss = -(exp_a * log_probs).mean()
        
    elif EPOLICY_EXTRACTION_METHON == 'ddpg':
        dist = iql_agent.get_actor(batch['states'])
        normalized_actions = torch.tanh(dist.loc)
        q1, q2 = iql_agent.get_critic(batch['states'], batch['actions'])
        q = (q1 + q2) / 2
        q_loss = -q.mean()
        log_probs = dist.log_prob(batch['actions'])
        bc_loss = -((config['bc_coefficient'] * log_probs)).mean()
        
        actor_loss = ((q_loss + bc_loss)).mean()
    
    
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
    
    
    ########################################################################################
    

    
    
    
    actor_losses.append(actor_loss.item())
    v_losses.append(v_loss.item())
    q_losses.append(q_loss.item())
    mse_errors.append(mse_error.item())
    stds.append(std.item())
    
    # break
    # continue

    if timestep % 1000 == 0:
        
        
        eval_returns = np.array([evaluate_policy(env, iql_agent) for _ in range(10)])
        normalized_returns = d4rl.get_normalized_score(ENV_NAME, eval_returns) * 100.0
        rewards.append(normalized_returns.mean())
        
        clear_output(True)
        fig, axs = plt.subplots(1, 6, figsize=(35, 5))
        
        axs[0].plot(smooth_and_downsample(actor_losses))
        axs[1].set_ylim(0,max(actor_losses[-100:]))
        axs[0].set_title("Actor Loss")
        
        axs[1].plot(smooth_and_downsample(v_losses))
        axs[1].set_ylim(0,max(v_losses[-100:]))
        axs[1].set_title("V Loss")
        
        axs[2].plot(smooth_and_downsample(q_losses))
        axs[2].set_ylim(0,max(q_losses[-100:]))
        axs[2].set_title("Q Loss")
        
        axs[3].plot(smooth_and_downsample(mse_errors))
        axs[3].set_ylim(0,max(mse_errors[-100:]))
        axs[3].set_title("MSE Errors")
        
        axs[4].plot(smooth_and_downsample(stds))
        axs[4].set_title("std")
        
        axs[5].plot(smooth_and_downsample(rewards))
        axs[5].set_title("rewards")
        
        plt.savefig(f"{LOGS_FOLDER}/iql_training.png")
        plt.close()
        
    # break


# In[ ]:


print(list(iql_agent.actor.model.parameters())[0][0, :5])


# In[ ]:




