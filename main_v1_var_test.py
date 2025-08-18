#!/usr/bin/env python
# coding: utf-8

# [1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import math

import numpy as np
import random

from tqdm import tqdm
from IPython.display import clear_output
from dataclasses import dataclass


device = 'cuda' if torch.cuda.is_available() else 'cpu'
device




@dataclass
class Dataset:
    trajectories: torch.Tensor
    actions: torch.Tensor
    terminals: torch.Tensor
    timeouts: torch.Tensor




class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a long enough PEs tensor
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # (1, max_len, d_model) so it can be broadcast over batch
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]  # match seq_len
        return self.dropout(x)


# [8]:


class FRENetwork(nn.Module):
    def __init__(self, state_dim, action_dim, num_heads=2, num_layers=2, d_model=128):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.num_discrete_embeddings = 32
        
        self.encoder_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                self.d_model, 
                num_heads, 
                dim_feedforward=4*self.d_model, 
                batch_first=True
            ),
            num_layers=num_layers
        )
        self.encoder_mean = nn.Linear(self.d_model, self.d_model)
        self.encoder_log_std = nn.Linear(self.d_model, self.d_model)

        self.state_embed = nn.Linear(self.state_dim, self.d_model // 2)
        self.action_embed = nn.Linear(self.action_dim, self.d_model // 2)

        self.action_predict = nn.Sequential(
            nn.Linear(self.state_dim + self.d_model, 512),
            nn.LayerNorm(512),
            nn.Mish(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.Mish(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.Mish(),
            nn.Linear(512, action_dim),
            nn.Tanh()
        )
        
        self.positional_encoding = PositionalEncoding(d_model=d_model, dropout=0.1, max_len=200)



    def get_transformer_encoding(self, states, actions, pad_mask):  
        
        batch_size, num_anchors = states.shape[0], states.shape[1]
        
        if pad_mask is None:
            pad_mask = torch.zeros((batch_size, num_anchors), dtype=torch.bool, device=states.device)
        # pad_mask.shape = [batch, anchors]
                
        state_emb = self.state_embed(states)
        action_emb = self.action_embed(actions)
        
        state_action_emb = torch.concat([state_emb, action_emb], dim=-1)
        
        x = self.positional_encoding(state_action_emb)
        w_pre = self.encoder_transformer(x, src_key_padding_mask=pad_mask) # [batch, anchors, emb_dim]
        
        
        valid_tokens = (~pad_mask).float()  # (B, T), converts True -> 0, False -> 1
        valid_tokens = valid_tokens.unsqueeze(-1)  # (B, T, 1)
        sum_embeddings = (w_pre * valid_tokens).sum(dim=1)  # Sum over sequence dimension
        w_pair_mean = sum_embeddings / valid_tokens.sum(dim=1)
        
        # w_pair_mean = w_pre.mean(axis=1)
        # print(w_pair_mean.shape)
        w_mean = self.encoder_mean(w_pair_mean)
        w_log_std = self.encoder_log_std(w_pair_mean)

        return w_mean, w_log_std # (batch_size, emb_dim)
    
    
    def get_action_pred(self, w, states): # Reward Pairs: [batch, seq, obs_dim]
        z_expand = w.unsqueeze(1) # [batch, 1, emb_dim]
        z_expand = z_expand.repeat(1, states.shape[1], 1)        
        
        w_and_obs = torch.concatenate([z_expand, states], axis=-1)
        
        reward_pred = self.action_predict(w_and_obs)
        
        return reward_pred # [batch, reward_pairs]
        


# FRE Unsupervised Reward Functions ####################################################################################v####

# [9]:
class MLPRewards:
    def __init__(self, N, obs_len):
        
        self.param_w1 = torch.normal(0, 1, size=(N, obs_len, 32)) * np.sqrt(1/32)
        self.param_b1 = torch.normal(0, 1, size=(N, 1, 32)) * np.sqrt(16)
        self.param_w2 = torch.normal(0, 1, size=(N, 32, 1)) * np.sqrt(1/16)
        
        if obs_len == 18:
            self.param_w1[:, -1:] = 0
        elif obs_len == 27:
            self.param_w1[:, -3:] = 0
            
    def __len__(self):
        return self.param_w1.shape[0]
    
    def sample(self, N):
        return torch.randint(0, len(self), (N,))

    def __call__(self, obs, param_id=None):
        """
        obs.shape = (batch_size, num_samples, obs_dim)
        param_id.shape = (batch_size, 1)
        """
        
        batch_size, num_samples, obs_dim = obs.shape
        batch_size_, _ = param_id.shape
        assert (batch_size == batch_size_)
        
        device = obs.device
        
        if param_id is None:
            param_id = torch.randint(0, len(self), size=(obs.shape[0], 1))
        
        param_id_expanded = param_id.repeat(1, obs.shape[1]).cpu()
        param1_w = self.param_w1[param_id_expanded].to(device)
        param1_b = self.param_b1[param_id_expanded].to(device)
        param2_w = self.param_w2[param_id_expanded].to(device)

        # obs[..., 2:] = 0.0

        x = torch.unsqueeze(obs, -2) # [batch, (pairs), 1, features_in]
        x = torch.matmul(x, param1_w) # [batch, (pairs), 1, features_out]
        x = x + param1_b
        x = torch.tanh(x)
        r = torch.matmul(x, param2_w) # [batch, (pairs), 1, 1]
        r = r.squeeze(-1).squeeze(-1) # [batch, (pairs)]
        r = torch.clip(r, -1, 1)

        return r, param_id
    


class LinearRewards:
    def __init__(self, N, obs_len):
        
        self.param_w1 = torch.rand(size=(N, obs_len, 1)) * 2 - 1
        self.random_mask = torch.rand(size=(N, obs_len)) < 0.9
        
        random_mask_positive = np.random.randint(2, obs_len, size=(N,))
        self.random_mask[np.arange(N), random_mask_positive] = False # Force at least one positive weight.
        
        if obs_len == 18:
            self.param_w1[:, -1:] = 0
        elif obs_len == 27:
            self.param_w1[:, -3:] = 0
        elif obs_len == 29:
            self.random_mask[..., :2] = True
    
    def __len__(self):
        return self.param_w1.shape[0]

    def sample(self, N):
        return torch.randint(0, len(self), (N,))

    def __call__(self, obs, param_id=None):
        """
        obs.shape = (batch_size, num_samples, obs_dim)
        param_id.shape = (batch_size, 1)
        """
        
        batch_size, num_samples, obs_dim = obs.shape
        batch_size_, _ = param_id.shape
        assert (batch_size == batch_size_)
        
        device = obs.device
        
        if param_id is None:
            param_id = torch.randint(0, len(self), size=(obs.shape[0], 1))
        
        param_id_expanded = param_id.repeat(1, obs.shape[1]).cpu()
        param1_w = self.param_w1[param_id_expanded].to(device)
        mask = self.random_mask[param_id_expanded].to(device)
        obs = (~mask) * obs
        
        x = torch.unsqueeze(obs, -2) # [batch, (pairs), 1, features_in]
        r = torch.matmul(x, param1_w) # [batch, (pairs), 1, features_out]
        r = r.squeeze(-1).squeeze(-1) # [batch, (pairs)]
        r = torch.clip(r, -1, 1)

        return r, param_id



class GoalRewards:
    def __init__(self):
        pass
    
    def sample_goals(self, N, dataset_trajectories):
        num_trajectories, len_trajectory, _ = dataset_trajectories.shape
        goals = dataset_trajectories[
            torch.randint(0, num_trajectories, (N,)),
            torch.randint(0, len_trajectory, (N,)),
        ]
        return goals
    
    # def get_states(self, num_states, goals):
    #     cond = (dataset_trajectories[..., :2] - goal[..., :2]).norm(dim=-1) < 0.5
    #     dataset_trajectories_cuda[cond][torch.randint(0, cond.sum(), (1,))]
    
    def __call__(self, obs, goals=None):
        """
        obs.shape = (batch_size, num_samples, obs_dim)
        param.shape = (batch_size, obs_dim)
        """
        
        if obs.shape[-1] == 18: # cheetah
            std = torch.tensor([[0.4407440506721877, 10.070289916801876, 0.5172332956856273, 0.5601041145815341, 0.518947027289748, 0.3204431592542281, 0.5501848643154092, 0.3856393812067661, 1.9882502334402663, 1.6377168569884073, 4.308505013609855, 12.144181770553105, 13.537567521831702, 16.88983033626308, 7.715009572436841, 14.345667964212357, 10.6904255152284, 100]])
        elif obs.shape[-1] == 27: # walker
            std = torch.tensor([[0.7212967364054736, 0.6775020895964047, 0.7638155887842976, 0.6395721376821286, 0.6849394775886244, 0.7078581708129903, 0.7113168519036742, 0.6753408522523937, 0.6818095329625652, 0.7133958718133511, 0.65227578338642, 0.757622576816855, 0.7311826446274479, 0.6745824928740024, 0.36822491550384456, 2.1134839667805805, 1.813353841099317, 10.594648894374815, 17.41041469033713, 17.836743227082106, 22.399097178637533, 16.1492222730888, 15.693574546557201, 18.539929326905067, 100, 100, 100]])
        elif obs.shape[-1] == 29: # antmaze
            std = torch.ones((29,))
        
        batch_size, num_samples, obs_dim = obs.shape
        batch_size_, obs_dim_ = goals.shape
        assert (batch_size == batch_size_) and (obs_dim == obs_dim_)
        
        device = obs.device
        
        goals = goals.to(device)

        # r = torch.norm(obs - goals.unsqueeze(-2), dim=-1) < 2
        dists_per_dim = obs - goals.unsqueeze(-2)
        dists_per_dim = dists_per_dim / std.to(device)
        dists = torch.norm(dists_per_dim, dim=-1) / obs.shape[-1]
        r = (dists < 0.2)
        r = r.float() * 2 - 1
        r = torch.clip(r, -1, 1)

        return r, goals






# [13]:



class UnsupervsiedReward:
    def __init__(self, args, dataset):
        
        self.args = args
        self.dataset_trajectories = dataset.trajectories
        obs_len = self.dataset_trajectories.shape[-1]
        
        self.linear_rewards = LinearRewards(N=10, obs_len=obs_len)
        self.mlp_rewards = MLPRewards(N=10, obs_len=obs_len)
        self.goal_rewards = GoalRewards()
    
        
    def sample_reward_function_fre(self, batch_size, num_random_samples):

        num_trajectories, len_trajectory, obs_len = self.dataset_trajectories.shape
        
        trajectories_idx = torch.randint(0, num_trajectories, (batch_size*num_random_samples,))
        states_idx = torch.randint(0, len_trajectory, (batch_size*num_random_samples,))
        random_states = self.dataset_trajectories[trajectories_idx, states_idx] # get the random states
        random_states = random_states.reshape(batch_size, num_random_samples, obs_len).to(device)

        reward_params = torch.zeros((batch_size, 128))
        random_states_rewards = torch.zeros((batch_size, num_random_samples))

        for b in range(batch_size):
            # reward_type = torch.randint(0, 3, (1,)) # 0: goal_reaching | 1: linear_reward | 2: mlp_reward
            # reward_type = torch.randint(1, 3, (1,))   # 1: linear_reward | 2: mlp_reward
            
            linear_rewards_ratio = len(self.linear_rewards) / (len(self.linear_rewards) + len(self.mlp_rewards))
            reward_type = 1 if random.random() < linear_rewards_ratio else 2
            
            reward_params[b, 0] = reward_type
            if reward_type == 0:
                goal = self.goal_rewards.sample_goals(1, self.dataset_trajectories)
                goal = goal.repeat(1, 1)
                r, param_id = self.goal_rewards(random_states[[b]], goals=goal)    
                reward_params[b, 1:1+obs_len] = param_id
                random_states[b, 0] = goal
                r[0, 0] = 1.
                
            elif reward_type == 1:
                param_id = self.linear_rewards.sample(1).unsqueeze(0)
                r, param_id = self.linear_rewards(random_states[[b]], param_id)
                reward_params[b, 1] = param_id
                
            elif reward_type == 2:
                param_id = self.mlp_rewards.sample(1).unsqueeze(0)
                r, param_id = self.mlp_rewards(random_states[[b]], param_id) 
                reward_params[b, 1] = param_id
                
            random_states_rewards[b] = r
            
        return reward_params, random_states, random_states_rewards


    def get_reward(self, reward_params, random_states):

        assert len(reward_params.shape) == 2
        assert len(random_states.shape) == 3
        assert reward_params.shape[0] == random_states.shape[0]
        
        _, _, obs_len = self.dataset_trajectories.shape
        all_rewards = torch.zeros((random_states.shape[0], random_states.shape[1]), device=device)

        for b in range(reward_params.shape[0]):
            param = reward_params[b]
            x = random_states[b].unsqueeze(0)
            if (param[0] == 2):
                rewards, _ = self.mlp_rewards(obs=x, param_id=param[1].reshape(1, 1).repeat(x.shape[0], 1).long())
            elif (param[0] == 1):
                rewards, _ = self.linear_rewards(obs=x, param_id=param[1].reshape(1, 1).repeat(x.shape[0], 1).long())
            elif (param[0] == 0):
                rewards, _ = self.goal_rewards(obs=x, goals=param[1:obs_len+1].unsqueeze(0).repeat(x.shape[0], 1))
            
            all_rewards[b] = rewards.float()

        return all_rewards


def get_best_reward_function_ids(args, dataset: Dataset, unsupervsied_rewards: UnsupervsiedReward):
    stds = []

    for reward_type in [1, 2]:

        num_functions = len(unsupervsied_rewards.linear_rewards) if reward_type == 1 else len(unsupervsied_rewards.mlp_rewards)
        
        for i in tqdm(range(num_functions)):
        
            with torch.no_grad():
                
                reward_params = torch.zeros((1, 128))
                reward_params[0, 0] = reward_type
                reward_params[0, 1] = i
                
                r = unsupervsied_rewards.get_reward(reward_params, random_states=dataset.trajectories[::10, ::10].reshape(1, -1, args.state_dim))
                std = r.reshape(1000, 100).mean(dim=1).std().item()
                stds.append(std)
            
            # print(std)
            
        
    stds = torch.tensor(stds)

    plt.plot()
    _ = plt.hist(stds, bins=100)
    plt.xlabel('Reward expressiveness')
    plt.title(f'Histogram of the expressiveness of {len(unsupervsied_rewards.linear_rewards)+len(unsupervsied_rewards.mlp_rewards)} reward functions on {args.env_name}')
    plt.savefig(f"{args.LOGS_FOLDER}/rewards_var_hist.png")

    sorted_ids = stds.argsort()


    top_rewards_ids = sorted_ids[-args.topk_rewards:]
    
    len_linear_rewards = len(unsupervsied_rewards.linear_rewards)
    
    ids_to_keep_linear = top_rewards_ids[(top_rewards_ids // len_linear_rewards) == 0] % len_linear_rewards
    ids_to_keep_mlp = top_rewards_ids[(top_rewards_ids // len_linear_rewards) == 1] % len_linear_rewards
    
    return ids_to_keep_linear, ids_to_keep_mlp



# FRE ####################################################################################################################

class FRENetwork(nn.Module):
    def __init__(self, obs_len, num_heads=2, num_layers=2, reward_pairs_emb_dim=128):
        super().__init__()
        
        self.obs_len = obs_len
        self.reward_pairs_emb_dim = reward_pairs_emb_dim
        self.num_discrete_embeddings = 32
        
        self.encoder_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                self.reward_pairs_emb_dim, 
                num_heads, 
                dim_feedforward=4*self.reward_pairs_emb_dim, 
                batch_first=True
            ),
            num_layers=num_layers
        )
        self.encoder_mean = nn.Linear(self.reward_pairs_emb_dim, self.reward_pairs_emb_dim)
        self.encoder_log_std = nn.Linear(self.reward_pairs_emb_dim, self.reward_pairs_emb_dim)

        self.reward_embed = nn.Embedding(self.num_discrete_embeddings, self.reward_pairs_emb_dim // 2)
        self.state_embed = nn.Linear(self.obs_len, self.reward_pairs_emb_dim // 2)

        self.reward_predict = nn.Sequential(
            nn.Linear(self.obs_len + self.reward_pairs_emb_dim, 512),
            nn.LayerNorm(512),
            nn.Mish(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.Mish(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.Mish(),
            nn.Linear(512, 1),
        )


    def get_transformer_encoding(self, reward_state_pairs):
        reward_states = reward_state_pairs[:, :, :-1]
        reward_values = reward_state_pairs[:, :, -1]
        reward_values_idx = torch.floor((reward_values / 2.0 + 0.5) * self.num_discrete_embeddings).int()
        reward_values_idx = torch.clip(reward_values_idx, 0, self.num_discrete_embeddings - 1)

        
        reward_state_emb = self.state_embed(reward_states)
        reward_state_val = self.reward_embed(reward_values_idx)
        
        reward_state_pairs = torch.concatenate([reward_state_emb, reward_state_val], axis=-1)

        w_pre = self.encoder_transformer(reward_state_pairs) # [batch, reward_pairs, emb_dim]
        
        w_pair_mean = w_pre.mean(axis=1)
        # print(w_pair_mean.shape)
        w_mean = self.encoder_mean(w_pair_mean)
        w_log_std = self.encoder_log_std(w_pair_mean)

        return w_mean, w_log_std # (batch_size, emb_dim)
    
    
    def get_reward_pred(self, w, reward_states): # Reward Pairs: [batch, reward_pairs, obs_dim + 1]
        z_expand = w.unsqueeze(1) # [batch, 1, emb_dim]
        z_expand = z_expand.repeat(1, reward_states.shape[1], 1)        
        
        w_and_obs = torch.concatenate([z_expand, reward_states], axis=-1)
        
        reward_pred = self.reward_predict(w_and_obs)
        
        return reward_pred # [batch, reward_pairs]
        


# Reward Generator ####################################################################################################################


class RewardGeneratorTransformer(nn.Module):
    def __init__(self, obs_len, num_heads=2, num_layers=2, reward_pairs_emb_dim=128):
        super().__init__()
        
        self.obs_len = obs_len
        self.reward_pairs_emb_dim = reward_pairs_emb_dim
        self.num_discrete_embeddings = 32
        
        self.encoder_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                self.reward_pairs_emb_dim, 
                num_heads, 
                dim_feedforward=4*self.reward_pairs_emb_dim, 
                batch_first=True
            ),
            num_layers=num_layers
        )
        self.encoder_mean = nn.Linear(self.reward_pairs_emb_dim, self.reward_pairs_emb_dim)
        self.encoder_log_std = nn.Linear(self.reward_pairs_emb_dim, self.reward_pairs_emb_dim)

        self.state_embed = nn.Linear(self.obs_len, self.reward_pairs_emb_dim // 2)
        self.reward_embed = nn.Embedding(self.num_discrete_embeddings, self.reward_pairs_emb_dim // 2)

        self.reward_predict = nn.Sequential(
            nn.Linear(self.obs_len + self.reward_pairs_emb_dim, 512),
            nn.LayerNorm(512),
            nn.Mish(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.Mish(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.Mish(),
            nn.Linear(512, 1),
            nn.Tanh()
        )



    def get_transformer_encoding(self, states, rewards, pad_mask):  
                
        if states.shape[-1] == 18: # cheetah
            states[..., -1:] = 0
        elif states.shape[-1] == 27: # walker
            states[..., -3:] = 0
        
        mask = (states != 0).float()
        # states = normalize_dataset_coords(states)
        states = states * mask
        
        batch_size, num_anchors = states.shape[0], states.shape[1]
        
        if pad_mask is None:
            pad_mask = torch.zeros((batch_size, num_anchors), dtype=torch.bool, device=device)
        # pad_mask.shape = [batch, anchors]
        
        # reward_values_idx = torch.floor(rewards * self.num_discrete_embeddings).int()
        reward_values_idx = torch.floor((rewards*0.5+0.5) * self.num_discrete_embeddings).int() # dont forget that rewards are in [-1, 1]
        reward_values_idx = torch.clip(reward_values_idx, 0, self.num_discrete_embeddings - 1)

        
        state_emb = self.state_embed(states)
        reward_emb = self.reward_embed(reward_values_idx.squeeze(-1))
        state_reward_emd = torch.concat((state_emb, reward_emb), dim=-1)
        
        w_pre = self.encoder_transformer(state_reward_emd, src_key_padding_mask=pad_mask) # [batch, anchors, emb_dim]
        
        
        valid_tokens = (~pad_mask).float()  # (B, T), converts True -> 0, False -> 1
        valid_tokens = valid_tokens.unsqueeze(-1)  # (B, T, 1)
        sum_embeddings = (w_pre * valid_tokens).sum(dim=1)  # Sum over sequence dimension
        w_pair_mean = sum_embeddings / valid_tokens.sum(dim=1)
        
        # w_pair_mean = w_pre.mean(axis=1)
        # print(w_pair_mean.shape)
        w_mean = self.encoder_mean(w_pair_mean)
        w_log_std = self.encoder_log_std(w_pair_mean)
        
        return w_mean, w_log_std # (batch_size, emb_dim)
    
    
    def get_reward_pred(self, w, states): # Reward Pairs: [batch, reward_pairs, obs_dim + 1]
                        
        if states.shape[-1] == 18: # cheetah
            states[..., -1:] = 0
        elif states.shape[-1] == 27: # walker
            states[..., -3:] = 0
                        
        mask = (states != 0).float()
        # states = normalize_dataset_coords(states)
        states = states * mask
        
        z_expand = w.unsqueeze(1) # [batch, 1, emb_dim]
        z_expand = z_expand.repeat(1, states.shape[1], 1)        
        
        w_and_obs = torch.concatenate([z_expand, states], axis=-1)
        
        reward_pred = self.reward_predict(w_and_obs)
        
        return reward_pred # [batch, reward_pairs]
    
    
    
class RewardGenerator:
    def __init__(self, fre_network: RewardGeneratorTransformer, dataset_trajectories, dropout):
        
        self.dataset_trajectories = dataset_trajectories
        self.dropout = dropout
        
        self.fre_network = fre_network.to(device)
        self.optimimizer = torch.optim.Adam(self.fre_network.parameters(), lr=0.001)
        
        self.len_params = 128
        self.resampling_weights = None
        
    
    def get_reward(self, obs, w):
        self.fre_network.eval()
        
        # obs.shape == (batch_size, obs_len)
        # w.shape == (batch_size, w_dim)
        assert obs.shape[0] == w.shape[0]
        assert len(obs.shape) == 2
        
        obs = obs.unsqueeze(1)
        with torch.no_grad():
            rewards_pred = self.fre_network.get_reward_pred(w, obs)
        
        return rewards_pred.reshape(-1, 1)
    
    
    def get_training_data(self, batch_size, min_num_anchors, max_num_anchors):
        assert min_num_anchors <= max_num_anchors

        dataset_trajectories = self.dataset_trajectories
        num_trajectories, len_trajectories, obs_dim = dataset_trajectories.shape
        

        # buffer = dataset_trajectories[..., :2].reshape(-1, 2)
        # buffer = dataset_trajectories
        
        anchors = torch.zeros((batch_size, max_num_anchors, obs_dim), dtype=torch.float32)

        # idx = self.get_importance_sampling_indices(batch_size*max_num_anchors,)
        # anchors = buffer[idx, :2]
        
        trajectories_idx = torch.randint(0, num_trajectories, (batch_size*max_num_anchors,))
        states_idx = torch.randint(0, len_trajectories, (batch_size*max_num_anchors,))
        anchors = dataset_trajectories[trajectories_idx, states_idx]
        anchors = anchors.reshape(batch_size, max_num_anchors, obs_dim)

            
        anchors_rewards = torch.zeros((batch_size, max_num_anchors), dtype=torch.float32)
        pad_mask = torch.ones((batch_size, max_num_anchors), dtype=torch.bool)

        # Generate random number of anchors for each batch element
        num_anchors = torch.randint(min_num_anchors, max_num_anchors + 1, (batch_size,))

        reward_indices = torch.arange(max_num_anchors).unsqueeze(0)
        reward_mask = reward_indices < num_anchors.unsqueeze(1)

        
        # candidates = torch.tensor([-1, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.])
        
        reward_types = torch.ones((batch_size,), dtype=torch.long) # 0: goal reaching | 1: MLP
        
        candidates = [
            # torch.tensor([-1, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.]),
            torch.tensor([-1, -0.5, 0.0, 0.5, 1.]),
            # torch.tensor([-1, -0.75, -0.5, -0.25, 0.0]),
            # torch.tensor([0.0, 0.25, 0.5, 0.75, 1.]),
        ]
        res = []
        for b in range(batch_size):
            x = torch.randint(0, len(candidates), (1,))
            res.append(candidates[x][torch.randint(0, candidates[x].shape[0], (max_num_anchors,))])
        anchors_rewards = torch.stack(res)
        
        # anchors_rewards[reward_mask] = candidates[torch.randint(0, candidates.shape[0], (reward_mask.sum(),))]
        
        # Goal reaching reward functions:
        random_rows = torch.tensor([n for n in range(batch_size) if random.random() < 0.3], dtype=torch.long)
        anchors_rewards[random_rows] = -1.
        anchors_rewards[random_rows, 0] = 1.
        reward_types[random_rows] = 0

            
        # rewards[reward_mask] = torch.exp(2*(rewards[reward_mask] - 1))
        anchors_rewards = anchors_rewards.unsqueeze(-1)

        pad_mask_indices = torch.arange(max_num_anchors).unsqueeze(0)
        pad_mask = pad_mask_indices < num_anchors.unsqueeze(1)
        pad_mask = ~pad_mask
                
        
        anchors = anchors.float()
        anchors_rewards = anchors_rewards.float()
        pad_mask = pad_mask
        
        return (anchors, anchors_rewards, pad_mask), {'reward_types':reward_types}
    

    def generate_boolean_mask(self, batch_size, length, p=0.5):
        
        all_vecs = []
        for b in range(batch_size):
            if torch.rand(1) < 0.5:
                vecs = (torch.rand(1, length) > 0.9).bool()
            else:
                vecs = (torch.rand(1, length) > p).bool()
            mask = ~vecs.any(dim=1)
            if mask.any():
                rows = mask.nonzero(as_tuple=False).squeeze(1)
                cols = torch.randint(0, length, (rows.size(0),))
                vecs[rows, cols] = True
            all_vecs.append(vecs)
        vecs = torch.concat(all_vecs)
            
        return vecs * 1.
    
        
    def train_step_VAE(self, batch_size, min_num_anchors, max_num_anchors):
        self.fre_network.train()
        
        (anchors, anchors_rewards, pad_mask), info = self.get_training_data(
            batch_size=batch_size, 
            min_num_anchors=min_num_anchors, 
            max_num_anchors=max_num_anchors,
        )
        anchors = anchors.to(device)
        anchors_rewards = anchors_rewards.to(device)
        pad_mask = pad_mask.to(device)
        
        state_dim = anchors.shape[-1]
        mask = self.generate_boolean_mask(batch_size, state_dim, p=0.5)
        if anchors.shape[-1] == 29: # antmaze
            mask[info['reward_types'] == 0, :2] = 1
            mask[info['reward_types'] == 0, 2:] = 0

        mask = mask.unsqueeze(1).repeat(1, max_num_anchors, 1)
        mask = mask.to(device)
        anchors = anchors * mask
                                

        w_mean, w_log_std = self.fre_network.get_transformer_encoding(anchors, anchors_rewards, pad_mask=pad_mask)
        
        
        
        w = w_mean + torch.normal(0, 1, size=w_mean.shape, device=device) * torch.exp(w_log_std)
        # w = w_mean
        rewards_pred = self.fre_network.get_reward_pred(w, anchors)
        
        
        reward_pred_loss = ((rewards_pred[~pad_mask] - anchors_rewards[~pad_mask])**2).mean()
                        
        
        kl_loss = -0.5 * (1 + 2*w_log_std - w_mean**2 - torch.exp(w_log_std)**2).mean()
        loss = reward_pred_loss + kl_loss * 0.01
        
        
        self.optimimizer.zero_grad()
        loss.backward()
        self.optimimizer.step()
        
        return {
            'loss': loss.item(),
            'reward_pred_loss': reward_pred_loss.item(),
            'kl_loss': kl_loss.item(),
            'get_training_data:info': info,
            'anchors': anchors.cpu(),
            'anchors_rewards': anchors_rewards.cpu(),
            'pad_mask': pad_mask.cpu(),
        }
        
    
    def get_z_from_random_anchors(self, batch_size: int, min_num_anchors:int, max_num_anchors:int, mask=None):
        self.fre_network.eval()
        
        (anchors, anchors_rewards, pad_mask), info = self.get_training_data(
            batch_size=batch_size, 
            min_num_anchors=min_num_anchors, 
            max_num_anchors=max_num_anchors,
        )
        anchors = anchors.to(device)
        anchors_rewards = anchors_rewards.to(device)
        pad_mask = pad_mask.to(device)
        
        state_dim = anchors.shape[-1]
        if mask is None:
            mask = self.generate_boolean_mask(batch_size, state_dim, p=0.0)
        
        assert mask.shape == (batch_size, state_dim)
        
        if anchors.shape[-1] == 29: # antmaze
            reward_types = info['reward_types']
            mask[reward_types == 0, :2] = 1
            mask[reward_types == 0, 2:] = 0
        
        mask = mask.unsqueeze(1).repeat(1, max_num_anchors, 1)
        mask = mask.to(device)
        anchors = anchors * mask
        
        eval_z, _ = self.get_z_from_anchors(anchors, anchors_rewards, pad_mask)
        
        return eval_z, {
            'anchors': anchors.cpu(), 
            'anchors_rewards': anchors_rewards.cpu(), 
            'pad_mask': pad_mask.cpu(), 
            'get_training_data:info': info
        }
    

    def get_z_from_anchors(self, anchors: torch.Tensor, anchors_rewards: torch.Tensor, pad_mask: torch.Tensor):    
        assert anchors.shape[:-1] == pad_mask.shape
        self.fre_network.eval()
        
        batch_size = anchors.shape[0]
        
        with torch.no_grad():
            w_mean, w_log_std = self.fre_network.get_transformer_encoding(anchors, anchors_rewards, pad_mask) 
        
        # eps = torch.normal(0, 1, (batch_size, self.emperical_mean.shape[0]), device=device)
        # z = w_mean + eps * torch.exp(w_log_std)
        z = w_mean
        
        return z, {'anchors': anchors.cpu()}
            
    
    def get_importance_sampling_indices(self, N):
        indices = torch.multinomial(self.resampling_weights, N, replacement=True)
        return indices


def sample_reward_function_fre_RG(reward_generator, dataset: Dataset, batch_size, num_random_samples):

    dataset_trajectories = dataset.trajectories
    num_trajectories, len_trajectory, state_dim = dataset_trajectories.shape
    trajectories_idx = torch.randint(0, num_trajectories, (batch_size*num_random_samples,))
    # states_idx = torch.randint(0, len_trajectory, (batch_size*num_random_samples,))
    states_idx = torch.randint(0, len_trajectory, (batch_size*num_random_samples,))

    random_states = dataset_trajectories[trajectories_idx, states_idx] # get the random states
    random_states = random_states.reshape(batch_size, num_random_samples, state_dim).to(device)

    reward_params = torch.zeros((batch_size, 128))
    random_states_rewards = torch.zeros((batch_size, num_random_samples))


    mask = reward_generator.generate_boolean_mask(batch_size, state_dim, p=reward_generator.dropout)
    

    with torch.no_grad():
        reward_params, info = reward_generator.get_z_from_random_anchors(
            batch_size, min_num_anchors=args.min_num_anchors, max_num_anchors=args.max_num_anchors, mask=mask
        )
        
        if state_dim == 29: # antmaze
            reward_types = info['get_training_data:info']['reward_types']
            mask[reward_types == 0, :2] = 1
            mask[reward_types == 0, 2:] = 0
        
        x = random_states * mask.unsqueeze(1).repeat(1, num_random_samples, 1).to(device)
        
        random_states_rewards = reward_generator.get_reward(
            x.reshape(-1, state_dim), 
            reward_params.unsqueeze(1).repeat(1, num_random_samples, 1).reshape(-1, 128)
        ).reshape(batch_size, num_random_samples)
    
    return reward_params, random_states, random_states_rewards, mask



def get_reward_RG(reward_generator, reward_params, mask, random_states):
    """
    reward_params: (batch_size, z_dim)
    random_states: (batch_size, num_states, obs_dim)
    """

    assert len(reward_params.shape) == 2
    assert len(random_states.shape) == 3
    assert reward_params.shape[0] == random_states.shape[0]
    
    batch_size, num_random_samples, state_dim = random_states.shape
    
    with torch.no_grad():
        
        mask = mask.unsqueeze(1).repeat(1, num_random_samples, 1).to(random_states.device)
        
        masked_random_states = random_states * mask
        
        rewards = reward_generator.get_reward(
            masked_random_states.reshape(-1, state_dim), 
            reward_params.unsqueeze(1).repeat(1, num_random_samples, 1).reshape(-1, 128)
        ).reshape(batch_size, num_random_samples)
    
    return rewards

# Benchamrks ########################################################################################################################

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
        
        return torch.tensor(rew[..., 0], dtype=torch.float32)




# IQL ############################################################################################################################



class MLP(nn.Module):
    """Generic MLP with Mish activation and LayerNorm."""
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim

        # Add hidden layers with Mish activation and LayerNorm
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
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
        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
        

    def forward(self, x, temperature=1.0):
        mean = self.model(x)  # Predict action mean
        mean = torch.nn.functional.tanh(mean)

        log_std = torch.clip(self.log_std, -5.0, 2.0)
        # mean = torch.clip(mean, -5, 5)

        return torch.distributions.MultivariateNormal(
            mean, 
            scale_tril=torch.diag(torch.exp(log_std))
        )


from torch.optim.lr_scheduler import CosineAnnealingLR
import copy

class IQL(nn.Module):
    def __init__(self, state_dim, action_dim, args, w_dim=128):
        super(IQL, self).__init__()
        self.obs_len = state_dim
                
        self.critic = Critic(w_dim + state_dim, action_dim, hidden_dims=[512, 512, 512])
        self.target_critic = copy.deepcopy(self.critic)
        for param in self.target_critic.parameters():
            param.requires_grad = False
        
        self.value = ValueCritic(w_dim + state_dim, hidden_dims=[512, 512, 512])        
        self.actor = Actor(w_dim + state_dim, action_dim, hidden_dims=[512, 512, 512])
        
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.value_optim = torch.optim.Adam(self.value.parameters(), lr=3e-4)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optim, args.iql_training_steps)
        
        
    def get_value(self, w, obs):
        w_and_obs = torch.concatenate([w, obs], dim=-1)
        return self.value(w_and_obs)

    def get_critic(self, w, obs, actions):
        w_and_obs = torch.concatenate([w, obs], dim=-1)
        return self.critic(w_and_obs, actions)
    
    def get_target_critic(self, w, obs, actions):
        w_and_obs = torch.concatenate([w, obs], dim=-1)
        return self.target_critic(w_and_obs, actions)

    def get_actor(self, w, obs, temperature=1.0):
        w_and_obs = torch.concatenate([w, obs], dim=-1)
        return self.actor(w_and_obs, temperature)
    
    
    
def update_target_critic(source, target, alpha):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1. - alpha).add_(source_param.data, alpha=alpha)
    

def expectile_loss(u, expectile=0.7):
    return torch.abs(expectile - (u < 0).float()) * u**2


def smooth_and_downsample(losses, smoothing=0.9, max_points=100):
    if not losses: return []
    smoothed = [losses[0]]
    for loss in losses[1:]:
        smoothed.append(smoothing * smoothed[-1] + (1 - smoothing) * loss)
    downsample = max(1, len(smoothed) // max_points)  # Ensure at least 1
    return smoothed[::downsample]


# [211]:


def get_iql_training_data(dataset:Dataset, batch_size, num_states):

    num_trajectories, len_trajectory, obs_dim = dataset.trajectories.shape

    trajectory_idx = torch.randint(0, num_trajectories, (batch_size*num_states,))
    state_idx = torch.randint(0, len_trajectory, (batch_size*num_states,)) % (len_trajectory - 1)

    states = dataset.trajectories[trajectory_idx, state_idx].reshape(batch_size, num_states, obs_dim)
    next_states = dataset.trajectories[trajectory_idx, state_idx+1].reshape(batch_size, num_states, obs_dim)
    actions = dataset.actions[trajectory_idx, state_idx].reshape(batch_size, num_states, args.action_dim)
    masks = ~dataset.timeouts[trajectory_idx, state_idx+1].reshape(batch_size, num_states, 1)
    
    return {
        'states': states.to(device),
        'actions': actions.to(device),
        'next_states': next_states.to(device),
        'masks': masks.to(device),
    }



        
def get_eval_rewards(dataset: Dataset, fre_network, eval_z, to_keep:list=None):
    obs_dim = dataset.trajectories.shape[-1]
    states = dataset.trajectories[0:300, :1000:10].reshape(-1, obs_dim)
    states = states.to(device)

    res = []
    for i in range(len(eval_z)):
        
        zi = eval_z[i].unsqueeze(0)
        with torch.no_grad():
            x = states.unsqueeze(0)
            if to_keep is not None:
                x = torch.zeros_like(x)
                x[..., to_keep] = states[..., to_keep]
            r = fre_network.get_reward_pred(zi, x).cpu()
            
            res.append(r)
    
    res = torch.stack(res).squeeze(-1)
    
    return x.cpu(), res.cpu()



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
        
    


def timestep2obs(timestep):
    obs = np.concatenate([v if len(v.shape) != 0 else v.reshape(-1) for k, v in timestep.observation.items()])
    return obs


def run_test_dmc(env, dataset, fre_network, iql_agent, benchmarks, benchmark_id, num_evals, num_eval_anchors):


    benchmark_reward_function, benchmark_test_label, benchmark_param = benchmarks[benchmark_id]

    produced_trajectories = []
    produced_trajectories_physics = []
    
    for _ in range(num_evals):
        

        # _, encode_obs, _ = sample_reward_function_fre(batch_size=1, num_random_samples=num_eval_anchors)
        batch = get_iql_training_data(
            dataset=dataset,
            batch_size=1,
            num_states=num_eval_anchors
        )
        encode_obs = batch['states'].to(device)


        encode_rewards = benchmark_reward_function(encode_obs.cpu(), benchmark_param).unsqueeze(-1).to(device)

        reward_state_pairs = torch.concatenate((encode_obs, encode_rewards), axis=-1)

        with torch.no_grad():
            w_mean, _ = fre_network.get_transformer_encoding(reward_state_pairs)  
            
            
            
        timestep = env.reset()        
        state = timestep2obs(timestep)
        # state = normalize_dataset_coords(state)
    
        produced_trajectory = []   
        produced_trajectory_physics = [] 


        for step in tqdm(range(1000)):
            
            physics = env.physics.get_state()
            
            produced_trajectory_physics.append(physics)
            
            
            if state.shape[-1] == 24: # walker
                horizontal_velocity = env.physics.horizontal_velocity()
                torso_upright = env.physics.torso_upright()
                torso_height = env.physics.torso_height()
                aux = np.array([horizontal_velocity, torso_upright, torso_height])

            elif state.shape[-1] == 17: # cheetah:
                horizontal_velocity = env.physics.speed()
                aux = np.array([horizontal_velocity])
            
            observation_aux = np.concatenate([state, aux])
            
            produced_trajectory.append(observation_aux)
            
            
            with torch.no_grad():
                tensor_state = torch.tensor(state).reshape(1, -1).to(device).float()
                                
                # if tensor_state.shape[-1] == 17: # cheetah
                #     tensor_state[..., -1:] = 0
                # elif tensor_state.shape[-1] == 24: # walker
                #     tensor_state[..., -3:] = 0
                
                # dist = iql_agent.get_actor(w_mean, tensor_state)
                # action = dist.loc.cpu()
                # action = np.array(action[0]).clip(-1, 1)
                
                action = iql_agent.get_actor(w_mean, tensor_state).mean.cpu().numpy()

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
    

    return produced_trajectories, produced_trajectories_physics, w_mean



def reset_to_location_antamze(env, location):
    env.sim.reset()
    qpos = env.init_qpos + env.np_random.uniform(low=-.1, high=.1, size=env.model.nq)
    qpos[:2] = np.array(location).astype(env.observation_space.dtype)
    qvel = env.init_qvel + env.np_random.randn(env.model.nv) * .1
    env.set_state(qpos, qvel)
    return env.unwrapped._get_obs()


def run_test_antamze(env, dataset, fre_network, iql_agent, benchmarks, benchmark_id, num_evals, num_eval_anchors):


    benchmark_reward_function, benchmark_test_label, benchmark_param = benchmarks[benchmark_id]

    produced_trajectories = []
    for _ in range(num_evals):

        # _, encode_obs, _ = sample_reward_function_fre(batch_size=1, num_random_samples=num_eval_anchors)
        batch = get_iql_training_data(
            dataset=dataset,
            batch_size=1,
            num_states=num_eval_anchors
        )
        encode_obs = batch['states'].to(device)

        encode_rewards = benchmark_reward_function(encode_obs.cpu(), benchmark_param).unsqueeze(-1).to(device)

        reward_state_pairs = torch.concatenate((encode_obs, encode_rewards), axis=-1)

        with torch.no_grad():
            w_mean, _ = fre_network.get_transformer_encoding(reward_state_pairs)  
            
            
            
        env.reset()
        location = (20, 15)
        start_state = reset_to_location_antamze(env, location)
        state = start_state

        tensor_state = torch.tensor(state).reshape(1, -1).to(device).float() 
    
    
        produced_trajectory = []    

        for step in tqdm(range(2000)):
            
            produced_trajectory.append(state)
            
            with torch.no_grad():
                tensor_state = torch.tensor(state).reshape(1, -1).to(device).float()
                # dist = iql_agent.get_actor(w_mean, tensor_state)
                # action = dist.loc.cpu()
                # action = np.array(action[0]).clip(-1, 1)
                
                action = iql_agent.get_actor(w_mean, tensor_state).mean.cpu().numpy().reshape(-1)
                
                

            new_state, _, _, _ = env.step(action)
            
            state = new_state
            
        produced_trajectory = np.stack(produced_trajectory)
        produced_trajectories.append(produced_trajectory)
    
    produced_trajectories = np.stack(produced_trajectories)

    return produced_trajectories, None, w_mean





def run_benchmark(args, env, dataset: Dataset, fre_network, iql_agent, benchmarks, steps, num_evals):
    fig, axs = plt.subplots(len(benchmarks), 3, figsize=(15, len(benchmarks)*4))

    
    all_produced_trajectories = []
    
    for benchmark_id in range(len(benchmarks)):

        benchmark_reward_function, benchmark_test_label, benchmark_param = benchmarks[benchmark_id]
        print(benchmark_test_label)
        
        if args.env_name in ['cheetah', 'walker']:
            produced_trajectory, produced_trajectory_physics, w_mean = run_test_dmc(
                env, dataset, fre_network, iql_agent, benchmarks, benchmark_id=benchmark_id, num_evals=num_evals, num_eval_anchors=128
            )
        elif args.env_name == 'antmaze':
            produced_trajectory, _, w_mean = run_test_antamze(
                env, dataset, fre_network, iql_agent, benchmarks, benchmark_id=benchmark_id, num_evals=num_evals, num_eval_anchors=128
            )

        eval_states, eval_rewards = get_eval_rewards(dataset, fre_network, w_mean)
        real_eval_rewards = benchmark_reward_function(eval_states, benchmark_param)

        if args.env_name == 'cheetah':
            axs[benchmark_id, 0].scatter(eval_states[..., 8], eval_states[..., 17], c=real_eval_rewards)
            axs[benchmark_id, 1].scatter(eval_states[..., 8], eval_states[..., 17], c=eval_rewards)
            axs[benchmark_id, 2].scatter(
                produced_trajectory_physics[..., 0],
                torch.arange(1000).unsqueeze(1).repeat(1, num_evals).T,
                c='red', s=1
            )
        elif args.env_name == 'walker':
            axs[benchmark_id, 0].scatter(eval_states[..., 16], eval_states[..., 24], c=real_eval_rewards)
            axs[benchmark_id, 1].scatter(eval_states[..., 16], eval_states[..., 24], c=eval_rewards)
            axs[benchmark_id, 2].scatter(
                produced_trajectory_physics[..., 1],
                torch.arange(1000).unsqueeze(1).repeat(1, num_evals).T,
                c='red', s=1
            )
        elif args.env_name == 'antmaze':
            axs[benchmark_id, 0].scatter(eval_states[..., 0], eval_states[..., 1], c=real_eval_rewards)
            axs[benchmark_id, 1].scatter(eval_states[..., 0], eval_states[..., 1], c=eval_rewards)
            axs[benchmark_id, 2].scatter(produced_trajectory[..., 0], produced_trajectory[..., 1], c='red', s=5)
            add_largest_maze_walls(axs[benchmark_id, 0])
            add_largest_maze_walls(axs[benchmark_id, 1])
            add_largest_maze_walls(axs[benchmark_id, 2])

            
        axs[benchmark_id, 0].set_title(f'{benchmark_test_label}')
        axs[benchmark_id, 1].set_title(f'Reconstructed Reward Function')
        axs[benchmark_id, 2].set_title(f'Agent Trajectory')
        
        all_produced_trajectories.append(produced_trajectory)
    
        
    np.savez(f"{args.MODEL_SAVE_FOLDER}/all_produced_trajectories", all_produced_trajectories)
    if args.iql_training_steps < 10 or steps % (args.iql_training_steps // 10) == 0 or (steps == args.iql_training_steps):
        plt.savefig(f"{args.LOGS_FOLDER}/benchmark-steps:{steps}.png")
    plt.close()
    
    
    all_produced_trajectories = np.stack(all_produced_trajectories)
    state_dim = all_produced_trajectories.shape[-1]
    
    benchmark_rewards = []
    
    for benchmark_id in range(len(benchmarks)):

        benchmark_reward_function, benchmark_test_label, benchmark_param = benchmarks[benchmark_id]
        

        trajectory_states = torch.tensor(all_produced_trajectories[benchmark_id]).reshape(1, -1, state_dim)
        trajectory_states_rewards = benchmark_reward_function(trajectory_states, benchmark_param).float()
        trajectory_states_rewards = trajectory_states_rewards.reshape(
            all_produced_trajectories.shape[1],
            all_produced_trajectories.shape[2],
        )
        trajectory_rewards = trajectory_states_rewards.sum(dim=-1)
        
        if args.env_name == 'antmaze' and 'goal' in benchmark_test_label: 
            trajectory_rewards = torch.where(trajectory_rewards != -all_produced_trajectories.shape[2], 1., 0.)
            
        print(benchmark_test_label, ':')
        print('\tRewards:', trajectory_rewards.tolist())
        print('\tmean:', trajectory_rewards.mean().item())
        print('\tstd:', trajectory_rewards.std().item())
        
        benchmark_rewards.append(trajectory_rewards.mean().item())

    benchmark_rewards = np.array(benchmark_rewards)
    
    return benchmark_rewards






def main(args):
    
    
    ################################################################################################################################################
    
    # Create the environment

    if args.env_name == 'cheetah':
        from dm_control import suite
        args.state_dim = 18
        args.action_dim = 6
        args.num_trajectories = 10000
        args.trajectory_len = 1000
        env = suite.load(
            domain_name='cheetah',
            task_name='run',
            environment_kwargs=dict(flat_observation=True)
        )
        dataset = np.load('datasets/cheetah_rnd.npy', allow_pickle=True).item()
        aux = np.load('datasets/aux_cheetah.npy')
        dataset['observations'] = np.concatenate((dataset['observations'], aux.reshape(10000, 1000, 1)), axis=-1)
        
        velocity_reward_function = VelocityRewardFunctionCheetah()
        benchmarks = [
            (velocity_reward_function.compute_reward, 'vel10Back', -10),
            (velocity_reward_function.compute_reward, 'vel2Back', -2),
            (velocity_reward_function.compute_reward, 'vel2', 2),
            (velocity_reward_function.compute_reward, 'vel10', 10),
        ]

    elif args.env_name == 'walker':
        from dm_control import suite
        args.state_dim = 27
        args.action_dim = 6
        args.num_trajectories = 10000
        args.trajectory_len = 1000
        env = suite.load(
            domain_name='walker',
            task_name='walk',
            environment_kwargs=dict(flat_observation=True)
        )
        dataset = np.load('datasets/walker_rnd.npy', allow_pickle=True).item()
        aux = np.load('datasets/aux_walker.npy')
        dataset['observations'] = np.concatenate((dataset['observations'], aux.reshape(10000, 1000, 3)), axis=-1)
        
        velocity_reward_function = VelocityRewardFunctionWalker()
        benchmarks = [
            (velocity_reward_function.compute_reward, 'vel0.1', 0.1),
            (velocity_reward_function.compute_reward, 'vel1', 1),
            (velocity_reward_function.compute_reward, 'vel4', 4),
            (velocity_reward_function.compute_reward, 'vel8', 8),
        ]

    elif args.env_name == 'antmaze':
        import gym
        import d4rl
        args.state_dim = 29
        args.action_dim = 8
        args.num_trajectories = 999
        args.trajectory_len = 1001
        env = gym.make('antmaze-large-diverse-v2')
        dataset = env.get_dataset()
        
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
        

    dataset_trajectories = torch.tensor(dataset['observations']).float()
    dataset_actions = torch.tensor(dataset['actions']).float()
    dataset_terminals = torch.tensor(dataset['terminals']).float()
    dataset_timeouts = torch.zeros(args.num_trajectories, args.trajectory_len).bool()
    dataset_timeouts[:, -1] = True
        
    if args.env_name == 'antmaze':
        N = args.num_trajectories * args.trajectory_len
        dataset_trajectories = dataset_trajectories[:N]
        dataset_actions = dataset_actions[:N]
        dataset_terminals = dataset_terminals[:N]
        dataset_timeouts = dataset_timeouts[:N]

    dataset_trajectories = dataset_trajectories.reshape(-1, args.trajectory_len, args.state_dim)
    dataset_actions = dataset_actions.reshape(-1, args.trajectory_len, args.action_dim)
    dataset_terminals = dataset_terminals
    dataset_timeouts = dataset_timeouts.reshape(-1, args.trajectory_len)

    dataset = Dataset(
        trajectories=dataset_trajectories,
        actions=dataset_actions,
        terminals=dataset_terminals,
        timeouts=dataset_timeouts
    )
    
    obs_len = dataset.trajectories.shape[-1]
    
    
    
    
    
    ################################################################################################################################################
    
    if args.method == 'rg':
        
        if args.env_name in ['cheetah', 'walker']:
            args.min_num_anchors, args.max_num_anchors = 2, 4
        elif args.env_name in ['antmaze']:
            args.min_num_anchors, args.max_num_anchors = 2, 8
        
        rg_model = RewardGeneratorTransformer(obs_len=obs_len)
        if args.rg_checkpoint:
            rg_model.load_state_dict(torch.load(args.rg_checkpoint))
            print('Reward generator loaded')

        reward_generator = RewardGenerator(
            fre_network=rg_model, 
            dataset_trajectories=dataset_trajectories,
            dropout=args.rg_dropout
        )

        vae_loss, vae_kl_loss = [], []

        print('Reward Generator training...')
        for step in tqdm(range(args.reward_generator_training_steps), desc='Reward Generator training', leave=False):
            
            vae_loss_dict = reward_generator.train_step_VAE(
                batch_size=256,
                min_num_anchors=args.min_num_anchors,
                max_num_anchors=args.max_num_anchors,
            )
            vae_loss.append(vae_loss_dict['loss'])
            vae_kl_loss.append(vae_loss_dict['kl_loss'])   
            
            # break
            
            if step % 100 == 0:
                clear_output(True)
                fig, axs = plt.subplots(1, 2, figsize=(10, 4))
                axs[0].plot(vae_loss)
                axs[0].set_ylim([0, 0.5])
                # axs[0].set_xscale('log')
                axs[1].plot(vae_kl_loss)
                axs[1].set_ylim([0, 1])
                plt.savefig(f"{args.LOGS_FOLDER}/reward_generator_losses.png")
                plt.close()
            # break

        torch.save(rg_model.state_dict(), f"{args.MODEL_SAVE_FOLDER}/rg_model.pth")


        fig, axs = plt.subplots(4, 4, figsize=(20, 20))
        axs = axs.flatten()

        reward_params, random_states, random_states_rewards, mask = sample_reward_function_fre_RG(
            reward_generator, dataset, batch_size=16, num_random_samples=10000
        )

        for b in range(16):
            if args.env_name == 'cheetah':
                axs[b].scatter(random_states[b, :, 8].cpu(), random_states[b, :, 17].cpu(), c=random_states_rewards[b].cpu(), vmin=-1, vmax=1)
            elif args.env_name == 'walker':
                axs[b].scatter(random_states[b, :, 16].cpu(), random_states[b, :, 24].cpu(), c=random_states_rewards[b].cpu(), vmin=-1, vmax=1)
            elif args.env_name == 'antmaze':
                axs[b].scatter(random_states[b, :, 0].cpu(), random_states[b, :, 1].cpu(), c=random_states_rewards[b].cpu(), vmin=-1, vmax=1)

        plt.savefig(f"{args.LOGS_FOLDER}/reward_generator_examples.png")
        plt.close()
    
    elif args.method == 'fre':
        unsupervsied_rewards = UnsupervsiedReward(args, dataset)  

        
        
    ids_to_keep_linear, ids_to_keep_mlp = get_best_reward_function_ids(args, dataset, unsupervsied_rewards)
    
    unsupervsied_rewards.linear_rewards.param_w1 = unsupervsied_rewards.linear_rewards.param_w1[ids_to_keep_linear]
    unsupervsied_rewards.linear_rewards.random_mask = unsupervsied_rewards.linear_rewards.random_mask[ids_to_keep_linear]

    unsupervsied_rewards.mlp_rewards.param_w1 = unsupervsied_rewards.mlp_rewards.param_w1[ids_to_keep_mlp]
    unsupervsied_rewards.mlp_rewards.param_b1 = unsupervsied_rewards.mlp_rewards.param_b1[ids_to_keep_mlp]
    unsupervsied_rewards.mlp_rewards.param_w2 = unsupervsied_rewards.mlp_rewards.param_w2[ids_to_keep_mlp]
    
    print('Unsupervised reward functions filtered !')
    print(f'Linear rewards:', len(unsupervsied_rewards.linear_rewards))
    print(f'MLP rewards:', len(unsupervsied_rewards.mlp_rewards))
        
    
    # FRE ########################################################################################################################################
        
    fre_network = FRENetwork(obs_len=obs_len).to(device)
    if args.encoder_checkpoint:
        fre_network.load_state_dict(torch.load(args.encoder_checkpoint))
        print('FRE encoder loaded')
    optimimizer = torch.optim.Adam(fre_network.parameters(), lr=0.001)

    reward_losses = []
    kl_losses = []


    # [17]:


    num_encode_states = 128
    num_decode_states = 128

    for i in tqdm(range(args.encoder_training_steps), desc='FRE encoder training'):
        
        if args.method == 'fre':
            reward_params, random_states, random_states_rewards = unsupervsied_rewards.sample_reward_function_fre(
                batch_size=256, num_random_samples=(num_encode_states+num_decode_states)
            )
        elif args.method == 'rg':
            reward_params, random_states, random_states_rewards, mask = sample_reward_function_fre_RG(
                reward_generator, dataset, 
                batch_size=256, num_random_samples=(num_encode_states+num_decode_states)
            )
        
        encode_obs = random_states[:, :num_encode_states, :].to(device)
        decode_obs = random_states[:, num_encode_states:, :].to(device)
        
        encode_rewards = random_states_rewards[:, :num_encode_states, None].to(device)
        decode_rewards = random_states_rewards[:, num_encode_states:, None].to(device)

                
        # encode_rewards, goals = goal_rewards(encode_obs, goals=None)
        # decode_rewards, goals = goal_rewards(decode_obs, goals=goals)
        
        reward_state_pairs = torch.concatenate((encode_obs, encode_rewards), axis=-1)
        
        w_mean, w_log_std = fre_network.get_transformer_encoding(reward_state_pairs)
        
        
        # Calculate the loss:
        
        w = w_mean + torch.normal(0, 1, size=w_mean.shape, device=device) * torch.exp(w_log_std)
        # w = w_mean
        rewards_pred = fre_network.get_reward_pred(w, decode_obs)
        
        reward_pred_loss = ((rewards_pred - decode_rewards)**2).mean()
        kl_loss = -0.5 * (1 + w_log_std - w_mean**2 - torch.exp(w_log_std)).mean()
        loss = reward_pred_loss + kl_loss * 0.01
        
        optimimizer.zero_grad()
        loss.backward()
        optimimizer.step()
        
        reward_losses.append(reward_pred_loss.item())
        kl_losses.append(kl_loss.item())
        
        if i % 20 == 0:
            clear_output(True)
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            axs[0].plot(reward_losses)
            # axs[0].set_xscale('log')
            axs[0].set_ylim(0, 0.5)
            axs[1].plot(kl_losses)
            plt.savefig(f"{args.LOGS_FOLDER}/encoder_training_losses.png")
            plt.close()
            
    
    torch.save(fre_network.state_dict(), f"{args.MODEL_SAVE_FOLDER}/fre_network.pth")
    
        
    
    ################################################################################################################ 
    
    num_eval_states = 10_000
    fig, axs = plt.subplots(len(benchmarks), 2, figsize=(10, 4*len(benchmarks)))
    
    for benchmark_id in range(len(benchmarks)):

        if args.method == 'fre':
            _, random_states, _ = unsupervsied_rewards.sample_reward_function_fre(
                batch_size=1, num_random_samples=(128+num_eval_states)
            )
        elif args.method == 'rg':
            _, random_states, _, mask = sample_reward_function_fre_RG(
                reward_generator, dataset,
                batch_size=1, num_random_samples=(128+num_eval_states)
            )

        encode_obs = random_states[:, :128, :].to(device)
        decode_obs = random_states[:, 128:, :].to(device)

        # encode_rewards = random_states_rewards[:, :128, None].to(device)
        # decode_rewards = random_states_rewards[:, 128:, None].to(device)

        
        encode_rewards = benchmarks[benchmark_id][0](encode_obs.cpu(), benchmarks[benchmark_id][2]).unsqueeze(-1).to(device)
        decode_rewards = benchmarks[benchmark_id][0](decode_obs.cpu(), benchmarks[benchmark_id][2]).unsqueeze(-1).to(device)

                
        # encode_rewards, goals = goal_rewards(encode_obs, goals=None)
        # decode_rewards, goals = goal_rewards(decode_obs, goals=goals)

        reward_state_pairs = torch.concatenate((encode_obs, encode_rewards), axis=-1)

        with torch.no_grad():
            w_mean, w_log_std = fre_network.get_transformer_encoding(reward_state_pairs)

            # Calculate the loss:

            w = w_mean + torch.normal(0, 1, size=w_mean.shape, device=device) * torch.exp(w_log_std)
            # w = w_mean
            
            rewards_pred = fre_network.get_reward_pred(w, decode_obs)


        # [168]:


        if args.env_name == 'cheetah':
            axs[benchmark_id, 0].scatter(decode_obs[..., 8].cpu(), decode_obs[..., 17].cpu(), c=rewards_pred.cpu())
            axs[benchmark_id, 1].scatter(decode_obs[..., 8].cpu(), decode_obs[..., 17].cpu(), c=decode_rewards.cpu())
        elif args.env_name == 'walker':
            axs[benchmark_id, 0].scatter(decode_obs[..., 16].cpu(), decode_obs[..., 24].cpu(), c=rewards_pred.cpu())
            axs[benchmark_id, 1].scatter(decode_obs[..., 16].cpu(), decode_obs[..., 24].cpu(), c=decode_rewards.cpu())
        elif args.env_name == 'antmaze':
            axs[benchmark_id, 0].scatter(decode_obs[..., 0].cpu(), decode_obs[..., 1].cpu(), c=rewards_pred.cpu())
            axs[benchmark_id, 1].scatter(decode_obs[..., 0].cpu(), decode_obs[..., 1].cpu(), c=decode_rewards.cpu())
        
    plt.savefig(f"{args.LOGS_FOLDER}/FRE_reconstruction.png")
    
    
    ################################################################################################################


    if args.env_name == 'antmaze':
        iql_agent = IQL(state_dim=args.state_dim, action_dim=args.action_dim, args=args).to(device)
    elif args.env_name == 'cheetah':
        iql_agent = IQL(state_dim=args.state_dim-1, action_dim=args.action_dim, args=args).to(device)
    elif args.env_name == 'walker':
        iql_agent = IQL(state_dim=args.state_dim-3, action_dim=args.action_dim, args=args).to(device)
    
    if args.iql_checkpoint:
        iql_agent.load_state_dict(torch.load(args.iql_checkpoint))
        print('IQL agent loaded')


    actor_losses = []
    v_losses, q_losses = [], []
    mse_errors = []
    stds = []
    rewards_logs = np.zeros((len(benchmarks), 0),)


    # [451]:


    config = {
        'expectile': 0.9,
        'temperature': 3.0,
        'discount': 0.99,
        'tau': 0.005,
        'bc_coefficient': 0.01
    }

    iql_batch_size = 1
    iql_num_states = 1024


    for timestep in tqdm(range(1, args.iql_training_steps+1)):

        
        if args.method == 'fre':
            reward_params, random_states, random_states_rewards = unsupervsied_rewards.sample_reward_function_fre(
                batch_size=iql_batch_size, num_random_samples=128
            )
        elif args.method == 'rg':
            reward_params, random_states, random_states_rewards, mask = sample_reward_function_fre_RG(
                reward_generator, dataset, 
                batch_size=iql_batch_size, num_random_samples=128
            )
        
        
        
        encode_obs = random_states[:, :128, :].to(device)
        encode_rewards = random_states_rewards[:, :128, None].to(device)
        reward_state_pairs = torch.concatenate((encode_obs, encode_rewards), axis=-1)
        
        
        batch = get_iql_training_data(
            dataset=dataset,
            batch_size=iql_batch_size, 
            num_states=iql_num_states
        )
        
        with torch.no_grad():
            w_mean, _ = fre_network.get_transformer_encoding(reward_state_pairs)
            # w_mean = torch.zeros_like(w_mean) ############################################################################### !!!!!!!!!
        
            # if args.method == 'fre':
            #     batch['rewards'] = unsupervsied_rewards.get_reward(reward_params=reward_params, random_states=batch['states']).unsqueeze(-1)
            # elif args.method == 'rg':
            #     batch['rewards'] = get_reward_RG(reward_generator, reward_params=reward_params, mask=mask, random_states=batch['states']).unsqueeze(-1)
            
            benchmark_id = 3
            batch['rewards'] = benchmarks[benchmark_id][0](
                batch['next_states'].flatten(0, 1).unsqueeze(0).cpu(), benchmarks[benchmark_id][2]
            ).reshape(iql_batch_size, iql_num_states, 1).to(device)
            
            # batch['rewards'] = benchmarks[3][0](batch['states'].flatten(0, 1).unsqueeze(0), 2).reshape(iql_batch_size, iql_num_states, 1)
            

        # Implicit Q-Learning
        
        if args.env_name == 'cheetah':
            batch['states'] = batch['states'][..., :-1]
            batch['next_states'] = batch['next_states'][..., :-1]
        elif args.env_name == 'walker':
            batch['states'] = batch['states'][..., :-3]
            batch['next_states'] = batch['next_states'][..., :-3]
        
        
        observations = batch['states']
        next_observations = batch['next_states']
        actions = batch['actions']
        terminals = (1. - batch['masks'].float())
        rewards = batch['rewards']
        
        w_target = w_mean.unsqueeze(1).repeat(1, batch['states'].shape[1], 1)
        
        
        with torch.no_grad():
            target_q1, target_q2 = iql_agent.get_target_critic(w_target, observations, actions)
            target_q = torch.min(target_q1, target_q2)
            next_v = iql_agent.get_value(w_target, next_observations).detach()
        
        
        # Value Loss: Update V towards expectile of min(q1, q2).
        
        v = iql_agent.get_value(w_target, observations)
        adv = target_q - v    
        v_loss = expectile_loss(adv, config['expectile']).mean()
        iql_agent.value_optim.zero_grad(set_to_none=True)
        v_loss.backward()
        iql_agent.value_optim.step()



        # Critic Loss. Update Q = r #############################
        targets = rewards + (1. - terminals.float()) * config['discount'] * next_v.detach()
        q1, q2 = iql_agent.get_critic(w_target, observations, actions)
        q_loss = (F.mse_loss(q1, targets) + F.mse_loss(q2, targets)) / 2 
        iql_agent.critic_optim.zero_grad(set_to_none=True)
        q_loss.backward()
        iql_agent.critic_optim.step()

        # if timestep % 10 == 0:
        update_target_critic(iql_agent.critic, iql_agent.target_critic, config['tau'])


        # Actor Loss ############################################

        if args.policy_extraction_method == 'awr':
            exp_adv = torch.exp(config['temperature'] * adv.detach()).clamp(max=100.)        
            policy_out = iql_agent.get_actor(w_target, observations)
            bc_losses = -policy_out.log_prob(actions).unsqueeze(-1)
            print((exp_adv * bc_losses).shape)
            policy_loss = torch.mean(exp_adv * bc_losses)
        
        elif args.policy_extraction_method == 'ddpg':
            policy_out = iql_agent.get_actor(w_target, observations)
            q1, q2 = iql_agent.get_critic(w_target, observations, policy_out.mean)
            q = (q1 + q2) / 2
            q_loss_ = -q.mean()
            log_probs = policy_out.log_prob(actions)
            bc_loss = -((config['bc_coefficient'] * log_probs)).mean()
            policy_loss = torch.mean(q_loss_ + bc_loss)
            
            print(q1.shape, q2.shape, q_loss_.shape, log_probs.shape, bc_loss.shape)
        
        
        iql_agent.actor_optim.zero_grad(set_to_none=True)
        policy_loss.backward()
        iql_agent.actor_optim.step()
        iql_agent.actor_lr_schedule.step()
        

        actor_losses.append(policy_loss.item())
        v_losses.append(v_loss.item())
        q_losses.append(q_loss.item())
        

        ########################################################################################

        
        
        if timestep % 5000 == 0:
            clear_output(True)
            fig, axs = plt.subplots(1, 3, figsize=(18, 5))
            axs[0].plot(smooth_and_downsample(actor_losses))
            axs[0].set_title("Actor Loss")
            
            axs[1].plot(smooth_and_downsample(v_losses))
            axs[1].set_ylim(0,max(v_losses[-100:]))
            axs[1].set_title("V Loss")
            
            axs[2].plot(smooth_and_downsample(q_losses))
            axs[2].set_ylim(0,max(q_losses[-100:]))
            axs[2].set_title("Q Loss")

            plt.savefig(f"{args.LOGS_FOLDER}/iql_training_losses.png")
            plt.close()
            
            
        if (args.iql_training_steps < 10) or (timestep % (args.iql_training_steps // 100) == 0):
            benchmark_rewards = run_benchmark(args, env, dataset, fre_network, iql_agent, benchmarks, steps=timestep, num_evals=args.num_evals)
            
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
            
            plt.savefig(f"{args.LOGS_FOLDER}/rewards.png")
            plt.close()
            
                      
            torch.save(iql_agent.state_dict(), f"{args.MODEL_SAVE_FOLDER}/iql_agent.pth")

    
    if args.iql_training_steps == 0:
        run_benchmark(args, env, dataset, fre_network, iql_agent, benchmarks, steps=0, num_evals=args.num_evals)            
        torch.save(iql_agent.state_dict(), f"{args.MODEL_SAVE_FOLDER}/iql_agent.pth")
    
    
    ################################################################################################################
    
    

    
    return







import argparse
import os
from datetime import datetime

def get_args():

    """
    
    python main_v1_var_test.py --env_name walker --method fre --policy_extraction_method ddpg\
            --reward_generator_training_steps 20000 --rg_dropout 0.5 \
            --topk_rewards 1000 --encoder_training_steps 10 \
            --iql_training_steps 2 \
            --num_evals 5               
    
    """
               
    parser = argparse.ArgumentParser(description="Training and Evaluation Parameters")
    parser.add_argument('--env_name', type=str, required=True, choices=['antmaze', 'cheetah', 'walker'])
    parser.add_argument('--method', type=str, choices=['fre', 'rg'], required=True)
    parser.add_argument('--policy_extraction_method', type=str, choices=['awr', 'ddpg'], required=True)
    
    parser.add_argument('--reward_generator_training_steps', type=int, required=True)
    parser.add_argument('--rg_dropout', type=float, default=0.5, help='the dropout probality of masking features during reward generation')
    parser.add_argument('--rg_checkpoint', type=str)
    
    parser.add_argument('--topk_rewards', type=int, required=True)
    parser.add_argument('--encoder_training_steps', type=int, required=True)
    parser.add_argument('--encoder_checkpoint', type=str)
    
    parser.add_argument('--iql_training_steps', type=int, required=True)
    parser.add_argument('--iql_checkpoint', type=str)
    
    parser.add_argument('--num_evals', type=int, required=True)
    parser.add_argument('--file_suffix', type=str)
    
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(args)
        
        
    now = datetime.now()
    date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    
    exp_name = f'{args.env_name}-topkR:{args.topk_rewards}'
    if args.file_suffix:
        exp_name = f'{exp_name}-{args.file_suffix}'
        
    LOGS_FOLDER = f'./logs/{date_time_str}_{exp_name}'
    MODEL_SAVE_FOLDER = f'./models/{date_time_str}_{exp_name}'

    os.makedirs(LOGS_FOLDER)
    os.makedirs(MODEL_SAVE_FOLDER)

    print('LOGS_FOLDER:', LOGS_FOLDER)
    print('MODEL_SAVE_FOLDER:', MODEL_SAVE_FOLDER)
    
    args.LOGS_FOLDER = LOGS_FOLDER
    args.MODEL_SAVE_FOLDER = MODEL_SAVE_FOLDER
    
    for key, value in vars(args).items():
        print(f'{key}: {value}')
    
    
    main(args)