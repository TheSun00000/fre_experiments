import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import numpy as np
import random

from tqdm import tqdm
from IPython.display import clear_output
import os
from datetime import datetime


import argparse
import gym
import d4rl # Import required to register environments, you may need to also import the submodule

from utils.reward_alignement import *
from utils.antmaze_benchmark import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)



TRAJECTORY_LEN = 1001
STATE_DIM = 29
# FEATURES_TO_CONSIDER = [0, 1, 15, 16]
FEATURES_TO_CONSIDER = torch.arange(29)


EPISODE_LENGTH = TRAJECTORY_LEN
Z_DIM = 128







# Create the environment
env = gym.make('antmaze-large-diverse-v2')
dataset = env.get_dataset()




dataset_trajectories = torch.tensor(dataset['observations'])
dataset_trajectories = dataset_trajectories[..., :STATE_DIM]

dataset_actions = torch.tensor(dataset['actions'])
dataset_terminals = torch.tensor(dataset['terminals'])
dataset_timeouts = torch.tensor(dataset['timeouts'])

dataset_goals = torch.tensor(dataset['infos/goal'])



# [:999*1001]

dataset_trajectories = dataset_trajectories[:999*1001].reshape(-1, 1001, STATE_DIM)
dataset_actions = dataset_actions[:999*1001].reshape(-1, 1001, 8)
dataset_terminals = dataset_terminals[:999*1001]
dataset_timeouts = dataset_timeouts[:999*1001]



num_trajectories, len_trajectory, obs_dim = dataset_trajectories.shape




dataset_mean = dataset_trajectories.mean([0, 1])
dataset_std = dataset_trajectories.std([0, 1])


def normalize_dataset_coords(dataset_, features_to_consider_only=False):
    is_numpy = isinstance(dataset_, np.ndarray)
    if is_numpy: dataset_ = torch.tensor(dataset_)
    dataset = dataset_.clone()
    if not features_to_consider_only:
        dataset = (dataset - dataset_mean) / dataset_std
    else:
        dataset = (dataset - dataset_mean[FEATURES_TO_CONSIDER]) / dataset_std[FEATURES_TO_CONSIDER]
    if is_numpy: dataset = np.array(dataset.cpu())
    return dataset

def denormalize_dataset_coords(dataset_, features_to_consider_only=False):
    is_numpy = isinstance(dataset_, np.ndarray)
    if is_numpy: dataset_ = torch.tensor(dataset_)
    dataset = dataset_.clone()
    if not features_to_consider_only:
        dataset = dataset * dataset_std + dataset_mean
    else:
        dataset = dataset * dataset_std[FEATURES_TO_CONSIDER] + dataset_mean[FEATURES_TO_CONSIDER]
    if is_numpy: dataset = np.array(dataset.cpu())
    return dataset

dataset_trajectories = normalize_dataset_coords(dataset_trajectories)
dataset_trajectories_cuda = dataset_trajectories.to(device)




def get_training_data(batch_size, num_anchors, num_states, trajectories_idx_=None):
    
    num_trajectories, len_trajectory, obs_dim = dataset_trajectories.shape
    
    if trajectories_idx_ is not None:
        batch_size = len(trajectories_idx_)
        
    if trajectories_idx_ is None: 
        trajectories_idx_ = torch.randint(0, num_trajectories, (batch_size,))      
        
    trajectories_idx_ = trajectories_idx_.unsqueeze(-1).long()    
    
    
    trajectories_idx = trajectories_idx_.repeat(1, num_anchors).reshape(-1)
    states_idx = torch.linspace(0, len_trajectory-1, num_anchors).long().repeat(batch_size)
    anchors = dataset_trajectories[trajectories_idx, states_idx].reshape(batch_size, num_anchors, -1)
    anchors_actions = dataset_actions[trajectories_idx, states_idx].reshape(batch_size, num_anchors, -1)
    
    
    trajectories_idx = trajectories_idx_.repeat(1, num_states).reshape(-1)
    states_idx = torch.linspace(0, len_trajectory-1, num_states).long().repeat(batch_size)
    states = dataset_trajectories[trajectories_idx, states_idx].reshape(batch_size, num_states, -1)
    actions = dataset_actions[trajectories_idx, states_idx].reshape(batch_size, num_states, -1)
    
    
    return (anchors, anchors_actions), (states, actions), {'trajectories_idx': trajectories_idx_}







        
        
        
        
class RewardGenerator:
    def __init__(self, obs_dim, fre_network: RewardGeneratorTransformer, min_num_anchors, max_num_anchors):
        self.obs_dim = obs_dim
        
        self.fre_network = fre_network.to(device)
        self.optimimizer = torch.optim.Adam(self.fre_network.parameters(), lr=0.001)
        
        self.emperical_mean = torch.zeros((Z_DIM,), dtype=torch.float32, device=device)
        self.emperical_std  = torch.ones((Z_DIM,), dtype=torch.float32, device=device)    
        
        self.len_params = Z_DIM
        
        self.min_num_anchors = min_num_anchors
        self.max_num_anchors = max_num_anchors
        
        self.resampling_weights = None
        
        self.pre_computed_zs = None

    
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
    
    
    def get_training_data(self, batch_size, min_num_anchors, max_num_anchors, num_states, num_intermediate_anchors=10, from_new_states=False, trajectories_idx_=None,
                          anchors_from_same_trajectory=True):
        assert min_num_anchors <= max_num_anchors <= num_states

        obs_dim = dataset_trajectories.shape[-1]
        

        # buffer = dataset_trajectories[..., :2].reshape(-1, 2)
        buffer = dataset_trajectories[..., :].reshape(-1, obs_dim)
        
        num_states = max_num_anchors

        anchors = torch.zeros((batch_size, max_num_anchors, obs_dim), dtype=torch.float32)

        idx = self.get_importance_sampling_indices(batch_size*max_num_anchors,)
        # anchors = buffer[idx, :2]
        anchors = buffer[idx, :]
        anchors = anchors.reshape(batch_size, max_num_anchors, obs_dim)

            
        anchors_rewards = torch.zeros((batch_size, num_states), dtype=torch.float32)
        pad_mask = torch.ones((batch_size, max_num_anchors), dtype=torch.bool)

        # Generate random number of anchors for each batch element
        num_anchors = torch.randint(min_num_anchors, max_num_anchors + 1, (batch_size,))

        reward_indices = torch.arange(num_states).unsqueeze(0)
        reward_mask = reward_indices < num_anchors.unsqueeze(1)

        
        candidates = torch.tensor([-1, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.])
        
        anchors_rewards[reward_mask] = candidates[torch.randint(0, candidates.shape[0], (reward_mask.sum(),))]
        
        # Goal reaching reward functions:
        random_rows = torch.tensor([n for n in range(batch_size) if random.random() < 0.3], dtype=torch.long)
        anchors_rewards[random_rows] = 0.
        anchors_rewards[random_rows, 0] = 1.
            
        # rewards[reward_mask] = torch.exp(2*(rewards[reward_mask] - 1))
        anchors_rewards = anchors_rewards.unsqueeze(-1)

        pad_mask_indices = torch.arange(max_num_anchors).unsqueeze(0)
        pad_mask = pad_mask_indices < num_anchors.unsqueeze(1)
        pad_mask = ~pad_mask
                
        
        anchors = anchors[..., FEATURES_TO_CONSIDER].float()
        anchors_rewards = anchors_rewards.float()
        pad_mask = pad_mask
        
        return (anchors, anchors_rewards, pad_mask), {}
    

    def generate_boolean_mask(self, args, batch_size, length, p=0.9):
        vecs = (torch.rand(batch_size, length) > p).bool()
        mask = ~vecs.any(dim=1)
        if mask.any():
            rows = mask.nonzero(as_tuple=False).squeeze(1)
            cols = torch.randint(0, length, (rows.size(0),))
            vecs[rows, cols] = True
            
        if args.keep_only_coords:
            vecs = torch.zeros_like(vecs)
            vecs[:, :2] = True
            
        return vecs * 1.
    
        
    def train_step_VAE(self, args, batch_size, min_num_anchors, max_num_anchors, num_states, from_new_states=False, non_anchor_coef=1, anchors_from_same_trajectory=True):
        self.fre_network.train()
        
        (anchors, anchors_rewards, pad_mask), info = self.get_training_data(
            batch_size=batch_size, 
            min_num_anchors=min_num_anchors, 
            max_num_anchors=max_num_anchors,
            num_states=num_states,
            from_new_states=from_new_states,
            anchors_from_same_trajectory=anchors_from_same_trajectory
        )
        anchors = anchors.to(device)
        anchors_rewards = anchors_rewards.to(device)
        pad_mask = pad_mask.to(device)
        
        mask = self.generate_boolean_mask(args, batch_size, len(FEATURES_TO_CONSIDER), p=args.vae_dropout_p)
        mask = mask.unsqueeze(1).repeat(1, max_num_anchors, 1)
        mask = mask.to(device)
        anchors = anchors * mask
        
        
        # print(anchors)
        # print(anchors.sum())
        w_mean, w_log_std = self.fre_network.get_transformer_encoding(anchors, anchors_rewards, pad_mask=pad_mask)
        
        
        # Calculate the loss:
        
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
        
    
    def get_z_from_random_anchors(self, batch_size: int, min_num_anchors:int, max_num_anchors:int, anchors_from_same_trajectory=True, mask=None):
        self.fre_network.eval()
        
        (anchors, anchors_rewards, pad_mask), info = self.get_training_data(
            batch_size=batch_size, 
            min_num_anchors=min_num_anchors, 
            max_num_anchors=max_num_anchors,
            from_new_states=True,
            num_states=max_num_anchors+1,
            anchors_from_same_trajectory=anchors_from_same_trajectory
        )
        anchors = anchors.to(device)
        anchors_rewards = anchors_rewards.to(device)
        pad_mask = pad_mask.to(device)
        
        if mask is None:
            mask = self.generate_boolean_mask(args, batch_size, len(FEATURES_TO_CONSIDER), p=0.0)
        
        assert mask.shape == (batch_size, len(FEATURES_TO_CONSIDER))
        
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
        
        eps = torch.normal(0, 1, (batch_size, self.emperical_mean.shape[0]), device=device)
        # z = w_mean + eps * torch.exp(w_log_std)
        z = w_mean
        
        return z, {'anchors': anchors.cpu()}
            
    
    def get_importance_sampling_indices(self, N):
        indices = torch.multinomial(self.resampling_weights, N, replacement=True)
        return indices
    
    
    
    
    

    # break






def visualize_rewards_and_trajectories(eval_z, reward_generator, anchors=None, anchors_rewards=None, pad_mask=None, mask=None, trajectories=None, select_max_trajectory=False,
                                       mode='', plot_ax=None):
    # eval_z, _ = reward_generator.get_z_from_anchors(anchors, pad_mask)

    # dataset_trajectories[:, 0].min(), dataset_trajectories[:, 0].max(), dataset_trajectories[:, 1].min(), dataset_trajectories[:, 1].max()

    s = 100
    x1_vals = np.linspace(dataset_trajectories[..., 0].min(), dataset_trajectories[..., 0].max(), s)
    x2_vals = np.linspace(dataset_trajectories[..., 1].min(), dataset_trajectories[..., 1].max(), s)

    # x1_vals = np.linspace(-0.3, 0.3, s) * STATE_SCALE
    # x2_vals = np.linspace(-0.3, 0.3, s) * STATE_SCALE

    # x1_vals = np.linspace(-0.3, 0.3, s)
    # x1_vals = np.round((x1_vals + 0.3) / 0.6 * 32)
    # x2_vals = np.linspace(-0.3, 0.3, s)
    # x2_vals = np.round((x2_vals + 0.3) / 0.6 * 32)



    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    # state_x12 = np.column_stack((X1.ravel(), X2.ravel()))
    # state = torch.zeros((s*s, 2), dtype=torch.float32)
    # state[:, :2] = torch.tensor(state_x12)
    # state[:, 2:] = torch.rand((s*s, 2)) * 0.1 - 0.05
    # state = dataset_trajectories[200:300, :1000:10].reshape(-1, obs_dim)
    state = dataset_trajectories[0:300, :1000:10].reshape(-1, obs_dim)
    if mode == 'pos':
        state[..., 2:] = 0.
    if mode == 'vel':
        state[..., :15] = 0.
        state[..., 17:] = 0.
    state = state.to(device)


    num_evals = eval_z.shape[0]
    num_rows = int(np.floor(num_evals/4))
    num_rows += 1 if (num_rows == 0) or num_evals % (num_rows*4) != 0 else 0
    fig, axs = plt.subplots(num_rows, 4, figsize=(20, num_rows*4))
    axs = axs.flatten()
    
    
    if (anchors is not None) and (anchors_rewards is not None) and (pad_mask is not None):
        anchors_rewards = anchors_rewards * 0.5 + 0.5
        anchors_rewards = anchors_rewards.clip(0, 1)
        anchors_rewards = anchors_rewards * 20
        anchors_rewards = anchors_rewards + 0.1
                
        anchors = np.where(~pad_mask[..., None], anchors, None)
        anchors_rewards = np.where(~pad_mask[..., None], anchors_rewards, 0)
        
        

    for i in range(len(eval_z)):
        
        
        
        zi = eval_z[i].unsqueeze(0).repeat(state.shape[0], 1)
        with torch.no_grad():
            x = state[..., FEATURES_TO_CONSIDER]
            if mask is not None:
                x = x * mask[i].unsqueeze(0).repeat(state.shape[0], 1).to(device)
            # x = torch.zeros_like(x)
            r = reward_generator.get_reward(x, zi).cpu()
        
        if plot_ax is not None:
            i = plot_ax
        
        axs[i].scatter(state[:, 0].cpu(), state[:, 1].cpu(), c=r, alpha=0.7, s=20, vmin=-1, vmax=1)
        # axs[i].scatter(coods[:, 0].cpu(), coods[:, 1].cpu(), c='red', alpha=0.3, s=20)
        # axs[i].pcolormesh(X1, X2, r.reshape(X1.shape), shading='gouraud', alpha=1, vmin=-1, vmax=1)
        # axs[i].contourf(X1, X2, r.reshape(X1.shape), levels=20, alpha=1, vmin=0.1, vmax=1.0)
        
        if (anchors is not None) and (anchors_rewards is not None) and (pad_mask is not None):
            axs[i].scatter(anchors[i, :, 0], anchors[i, :, 1], c='red', s=anchors_rewards[i, :, 0].reshape(-1))
        # axs[i].scatter(anchors[i, -1, 0].cpu(), anchors[i, -1, 1].cpu(), c=anchors_rewards[i, -1, 0],  edgecolors='red', vmin=0, vmax=1)
        # axs[i].scatter(anchors[i, 0, 0].cpu(), anchors[i, 0, 1].cpu(), c='red', s=anchors_rewards[i, 0, 0].reshape(-1).cpu()*100)
        # axs[i].scatter(info['get_training_data:info']['base_anchors'][i, -1, 0].cpu(), info['get_training_data:info']['base_anchors'][i, -1, 1].cpu(), c='red', s=anchors_rewards[i, -1, 0].reshape(-1).cpu()*10)
        
        if trajectories is not None:
            if select_max_trajectory:
                # j = np.argmax([trajectories[j][0]['rewards'][TEMP_EPISODE_LENGTH*i:TEMP_EPISODE_LENGTH*(i+1)][-10:].mean() for j in range(len(trajectories))])
                max_j = np.argmax([trajectories[j][0]['rewards'][EPISODE_LENGTH*i:EPISODE_LENGTH*(i+1)].sum() for j in range(len(trajectories))])
                # trajectories = [trajectories[j]]
                
            for j in range(len(trajectories)):
                if select_max_trajectory:
                    j = max_j
                    
                trajectory, _ = trajectories[j]
                coords = trajectory['states'][EPISODE_LENGTH*i:EPISODE_LENGTH*(i+1), :2]
                r_mean = trajectory['rewards'][EPISODE_LENGTH*i:EPISODE_LENGTH*(i+1)][-10:].mean()
                axs[i].scatter(coords[:, 0], coords[:, 1], s=10, c='black')
                
                axs[i].set_xlabel("X-Coordinates")
                axs[i].set_xlabel(f"{r_mean}")
                axs[i].set_ylabel("Y-Coordinates")
                
                if select_max_trajectory:
                    break
        
    

    return fig, axs, {'state':state.cpu(), 'r':r.cpu()} 
    


    
    
class RewardFunctionEncoder(nn.Module):
    def __init__(self, obs_len, num_heads=2, num_layers=2, reward_pairs_emb_dim=Z_DIM):
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

        self.init_state_embed = nn.Linear(2, self.reward_pairs_emb_dim)
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
        )



    def get_transformer_encoding(self, init_state, states, rewards, pad_mask):  
        
        
        batch_size, num_anchors = states.shape[0], states.shape[1]
        
        if pad_mask is None:
            pad_mask = torch.zeros((batch_size, num_anchors), dtype=torch.bool, device=device)
        # pad_mask.shape = [batch, anchors]
        
        # reward_values_idx = torch.floor(rewards * self.num_discrete_embeddings).int()
        reward_values_idx = torch.floor((rewards*0.5+0.5) * self.num_discrete_embeddings).int() # dont forget that rewards are in [-1, 1]
        reward_values_idx = torch.clip(reward_values_idx, 0, self.num_discrete_embeddings - 1)
        
        init_state_emd = self.init_state_embed(init_state).unsqueeze(1)
        state_emb = self.state_embed(states)
        reward_emb = self.reward_embed(reward_values_idx.squeeze(-1))
        state_reward_emd = torch.concat((state_emb, reward_emb), dim=-1)      
        # x = torch.concat((init_state_emd, state_reward_emd), dim=1)
        # pad_mask = torch.concat((pad_mask, pad_mask[:, [0]]), dim=1)
        
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
    
    
    def get_reward_pred(self, w, reward_states): # Reward Pairs: [batch, reward_pairs, obs_dim + 1]
        z_expand = w.unsqueeze(1) # [batch, 1, emb_dim]
        z_expand = z_expand.repeat(1, reward_states.shape[1], 1)        
        
        w_and_obs = torch.concatenate([z_expand, reward_states], axis=-1)
        
        reward_pred = self.reward_predict(w_and_obs)
        
        return reward_pred # [batch, reward_pairs]
        
        
        
    


def get_best_trajectory(start_states, reward_w, reward_generator, pre_computed_zs, num_considered_steps=100):
    
    best_traj_w = []
    best_traj_idx = []
    cumm_r_list = []

    for i in range(reward_w.shape[0]):
        
        start_states = start_states.to(device)
        
        cond = (torch.norm(dataset_trajectories_cuda[:, ::, :2] - start_states[i, :2].unsqueeze(0), p=2, dim=2) < 0.1)

        valid_trajectories_bool = cond.any(dim=1) # True if the trajectory contains the start_state, otherwise False
        valid_trajectories_idx = torch.nonzero(valid_trajectories_bool).flatten() # The indicies of the trajectories containing the start_state
        valid_start_states_idx = cond[valid_trajectories_bool].long().argmax(dim=1) # The index of the start_state in each trajectory (the start of the sub-trajectory)
        # print(valid_trajectories_idx)
        # print(valid_start_states_idx)
        # break

        # The index of the states from each sub-trajectory        
        valid_start_states_idx = valid_start_states_idx.unsqueeze(1)  # (N, 1)
        step_fractions = torch.linspace(0, 1, num_considered_steps, device=valid_start_states_idx.device)  # (S,)
        state_idx = valid_start_states_idx + (TRAJECTORY_LEN - 1 - valid_start_states_idx) * step_fractions  # (N, S)
        state_idx = state_idx.long()  # Optional: use .floor() or .ceil() depending on behavior

        
        # x = torch.stack([dataset_trajectories_cuda[valid_trajectories_idx[i], state_idx[i], :2] for i in range(len(valid_trajectories_idx))])
        # x = dataset_trajectories_cuda[valid_trajectories_idx[:, None], state_idx, :2]
        x = dataset_trajectories_cuda[valid_trajectories_idx[:, None], state_idx][..., FEATURES_TO_CONSIDER]
        
        num_filtered_trajectories, _, obs_dim = x.shape
        
        x = x.reshape(-1, obs_dim).to(device)
        w = reward_w[[i]].repeat(x.shape[0], 1)

        with torch.no_grad():
            r = reward_generator.get_reward(x, w).reshape(num_filtered_trajectories, num_considered_steps)
            
            # Calculate the sum of all reward over the trajectories
            cumm_r = r.sum(-1)

        # Find the index of the best trajectory in the set of valid trajectories
        relative_traj_max_idx = cumm_r.argmax().item()
        
        # Find the index of the best trajectory in the of all trajectories
        traj_max_idx = torch.arange(0, num_trajectories)[valid_trajectories_bool.cpu()][relative_traj_max_idx]
        
        best_traj_idx.append(torch.arange(0, num_trajectories)[valid_trajectories_bool.cpu()][cumm_r.argsort(descending=True).cpu()])
        best_traj_w.append(pre_computed_zs[traj_max_idx])
        cumm_r_list.append(cumm_r.sort(descending=True).values.cpu())
    
    # The representations of the best trajectories:
    best_traj_w = torch.stack(best_traj_w).to(device)
    
    return best_traj_w, {'best_traj_idx': best_traj_idx, 'cumm_r': cumm_r_list}



  
    
def get_best_trajectory_for_benchmark(start_states, reward_function, param, pre_computed_zs, num_considered_steps=100):
    
    best_traj_w = []
    best_traj_idx = []

    for i in range(start_states.shape[0]):
        
        start_states = start_states.to(device)
        
        cond = (torch.norm(dataset_trajectories_cuda[:, ::, :2] - start_states[i, :2].unsqueeze(0), p=2, dim=2) < 0.1)

        valid_trajectories_bool = cond.any(dim=1) # True if the trajectory contains the start_state, otherwise False
        valid_trajectories_idx = torch.nonzero(valid_trajectories_bool).flatten() # The indicies of the trajectories containing the start_state
        valid_start_states_idx = cond[valid_trajectories_bool].long().argmax(dim=1) # The index of the start_state in each trajectory (the start of the sub-trajectory)
        # print(valid_trajectories_idx)
        # print(valid_start_states_idx)
        # break

        # The index of the states from each sub-trajectory        
        valid_start_states_idx = valid_start_states_idx.unsqueeze(1)  # (N, 1)
        step_fractions = torch.linspace(0, 1, num_considered_steps, device=valid_start_states_idx.device)  # (S,)
        state_idx = valid_start_states_idx + (TRAJECTORY_LEN - 1 - valid_start_states_idx) * step_fractions  # (N, S)
        state_idx = state_idx.long()  # Optional: use .floor() or .ceil() depending on behavior

        
        # x = torch.stack([dataset_trajectories_cuda[valid_trajectories_idx[i], state_idx[i], :2] for i in range(len(valid_trajectories_idx))])
        # x = dataset_trajectories_cuda[valid_trajectories_idx[:, None], state_idx, :2]
        x = dataset_trajectories_cuda[valid_trajectories_idx[:, None], state_idx, :]
        x = denormalize_dataset_coords(x.cpu()).to(device)
        
        num_filtered_trajectories, _, _ = x.shape
        
        x = x.reshape(-1, obs_dim).cpu()

        with torch.no_grad():
            # r = reward_generator.get_reward(x, w).reshape(num_filtered_trajectories, num_considered_steps)
            
            
            r = reward_function(x, param).reshape(num_filtered_trajectories, num_considered_steps)
            
            
            # Calculate the sum of all reward over the trajectories
            cumm_r = r.sum(-1)

        # Find the index of the best trajectory in the set of valid trajectories
        relative_traj_max_idx = cumm_r.argmax().item()
        
        # Find the index of the best trajectory in the of all trajectories
        traj_max_idx = torch.arange(0, num_trajectories)[valid_trajectories_bool.cpu()][relative_traj_max_idx]
        
        best_traj_idx.append(torch.arange(0, num_trajectories)[valid_trajectories_bool.cpu()][cumm_r.argsort(descending=True).cpu()])
        best_traj_w.append(pre_computed_zs[traj_max_idx])
    
    # The representations of the best trajectories:
    best_traj_w = torch.stack(best_traj_w).to(device)
    
    return best_traj_w, {'best_traj_idx': best_traj_idx, 'rewards': r[cumm_r.argsort(descending=True).cpu()]}




        
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




def run_test(benchmark_id, num_evals, num_eval_anchors, frenetwork, reward_encoder, reward_generator, pre_computed_zs):

    benchmark_reward_function, benchmark_test_label, benchmark_param = benchmarks[benchmark_id]


    reward_z_list = []
    start_states_list = []
    produced_trajectories_list = []
    best_trajectories_idx_list = []
    best_trajectories_bench_idx_list = []
    best_trajectories_from_alignement = []


    for num_eval in range(num_evals):
        (anchors, _, pad_mask), _ = reward_generator.get_training_data(
            batch_size=1, 
            min_num_anchors=num_eval_anchors, 
            max_num_anchors=num_eval_anchors,
            from_new_states=True,
            num_states=num_eval_anchors+1,
            anchors_from_same_trajectory=False
        )
        
        # Replace some of the anchors with anchors from the 
        if 'goal' in benchmark_test_label:
            norm_goal_state = (torch.tensor(benchmark_param) - dataset_mean[:2]) / dataset_std[:2]
            goal_state = norm_goal_state.to(device).float()
            cond = (torch.norm(dataset_trajectories_cuda[:, ::, :2] - goal_state[:2], p=2, dim=2) < 0.5)
            anchors[0, :num_eval_anchors//3] = dataset_trajectories_cuda[cond][torch.randint(0, cond.sum(), (num_eval_anchors//3,))][..., FEATURES_TO_CONSIDER]
            # print(anchors)

        anchors_rewards = benchmark_reward_function(
            denormalize_dataset_coords(anchors, features_to_consider_only=True),
            benchmark_param
        ).to(device)

        anchors = anchors.to(device)
        pad_mask = pad_mask.to(device)
        
        if args.keep_only_coords:
            mask = torch.zeros((1, num_eval_anchors, len(FEATURES_TO_CONSIDER)), device=device)
            mask[..., [0, 1]] = 1
            anchors = anchors * mask

        reward_z, _ = reward_generator.get_z_from_anchors(anchors, anchors_rewards, pad_mask)
                
        
        




        env.reset()
        # location = np.array(env.unwrapped._wrapped_env._get_reset_location())
        location = (20, 15)
        start_state = reset_to_location(env, location)
        start_state = normalize_dataset_coords(start_state)[..., :]
        state = start_state

        tensor_state = torch.tensor(state).reshape(1, -1).to(device).float()

        # Get the predicted representation of optimmal trajectory `trajectory_z`:
        # trajectory_z = R2T_model(tensor_state[:, FEATURES_TO_CONSIDER], reward_z)
        
        # print(tensor_state.shape, anchors.shape, anchors_rewards.shape, pad_mask.shape)
        # print(anchors.min(), anchors.max(), anchors_rewards.min(), anchors_rewards.max(), tensor_state)
        trajectory_z, _ = reward_encoder.get_transformer_encoding(
            tensor_state[..., :2],
            anchors, 
            anchors_rewards, 
            pad_mask=pad_mask 
        )
        
        
        diffs = (trajectory_z[:, None].cpu() - pre_computed_zs[None]).pow(2).sum(-1)
        tmp = diffs.flatten().argsort()
        best_trajectories_from_alignement.append(tmp)


        # Get the best trajectory from the dataset:
        predcited_best_w, get_best_trajectory_info = get_best_trajectory(tensor_state[:, :2].cpu(), reward_z, reward_generator, pre_computed_zs)    
        best_traj_from_benchmark, get_best_trajectory_info_bench = get_best_trajectory_for_benchmark(
            tensor_state[:, :2].cpu(), benchmark_reward_function, benchmark_param, pre_computed_zs
        )


        # Run the agent conditioned on the predicted trajectory representation:

        produced_trajectory = []

        for step in tqdm(range(2000)):
            
            produced_trajectory.append(state)
            
            if step % 10 == 0:
                tensor_state = torch.tensor(state).reshape(1, -1).to(device).float()
                # predcited_best_w, get_best_trajectory_info = get_best_trajectory(tensor_state[:, :2].cpu(), reward_z)
                # trajectory_z = R2T_model(tensor_state[:, :], reward_z)
            
            with torch.no_grad():
                tensor_state = torch.tensor(state).reshape(1, 1, -1).to(device).float()
                action = frenetwork.get_action_pred(predcited_best_w, tensor_state).cpu()
                # action = frenetwork.get_action_pred(trajectory_z, tensor_state).cpu()
                action = np.array(action[0, 0])
                
            new_state, _, _, _ = env.step(action)
            
            state = normalize_dataset_coords(new_state)[..., :]
            
            
            
        produced_trajectory = np.stack(produced_trajectory)
        
        ##############
        reward_z_list.append(reward_z)
        start_states_list.append(start_state)
        produced_trajectories_list.append(produced_trajectory)
        best_trajectories_idx_list.append(get_best_trajectory_info['best_traj_idx'][0][:10])
        best_trajectories_bench_idx_list.append(get_best_trajectory_info_bench['best_traj_idx'][0][:10])
        
        
        
        
        
    reward_z_list = torch.concat(reward_z_list)
    start_states_list = np.stack(start_states_list)
    produced_trajectories_list = np.stack(produced_trajectories_list)
    best_trajectories_idx_list = np.array(best_trajectories_idx_list)
    best_trajectories_bench_idx_list = np.array(best_trajectories_bench_idx_list)
    best_trajectories_from_alignement = torch.concat(best_trajectories_from_alignement)
    
    return (
        reward_z_list, start_states_list, produced_trajectories_list, best_trajectories_idx_list,
        best_trajectories_bench_idx_list, best_trajectories_from_alignement, 
        {'anchors': anchors, 'anchors_rewards': anchors_rewards, 'pad_mask': pad_mask}
    )
    
    


def visualize_eval(eval_z, reward_generator, axs):
    # eval_z, _ = reward_generator.get_z_from_anchors(anchors, pad_mask)

    # dataset_trajectories[:, 0].min(), dataset_trajectories[:, 0].max(), dataset_trajectories[:, 1].min(), dataset_trajectories[:, 1].max()

    states = dataset_trajectories[0:300, :1000:10].reshape(-1, obs_dim)
    states = states.to(device)
        

    # for i in range(len(eval_z)):
        
        
    zi = eval_z[0].unsqueeze(0).repeat(states.shape[0], 1)
    with torch.no_grad():
        states_rewards = reward_generator.get_reward(states[..., FEATURES_TO_CONSIDER], zi).cpu()
    
    
    # axs[2].scatter(states[:, 0].cpu(), states[:, 1].cpu(), c=states_rewards, alpha=0.7, s=20, vmin=-1, vmax=1)


    return fig, axs, states.cpu(), states_rewards.cpu().flatten()
    
    
    

def main(args):
    
    frenetwork = FRENetwork(state_dim=obs_dim, action_dim=8, num_layers=2, num_heads=2).to(device)
    frenetwork.load_state_dict(torch.load('models/offline_agent_norm.pth'))

    num_trajectories = dataset_trajectories.shape[0]
    batch_size = 512
    indicies = torch.arange(0, num_trajectories, dtype=torch.long)

    z_list = []

    for i in tqdm(range(0, num_trajectories, batch_size)):
        traj_idx = indicies[i:i+batch_size]
        (anchors, anchor_actions), _, info = get_training_data(
            batch_size=-1,
            num_anchors=200,
            num_states=1,
            trajectories_idx_=traj_idx
        )
        anchors, anchor_actions = anchors.to(device), anchor_actions.to(device)
        with torch.no_grad():
            frenetwork.eval()
            z, _ = frenetwork.get_transformer_encoding(anchors, anchor_actions, pad_mask=None)
        z_list.append(z)
    pre_computed_zs = torch.concat(z_list).cpu()

    ################################################################################################

    MIN_NUM_ANCHORS = args.min_num_anchors
    MAX_NUM_ANCHORS = args.max_num_anchors

    rg_model = RewardGeneratorTransformer(obs_len=len(FEATURES_TO_CONSIDER))
    # rg_model.load_state_dict(torch.load('models/offline_reward_generator.pth'))
    # rg_model.load_state_dict(torch.load('models/offline_reward_generator_with_mask.pth'))

    reward_generator = RewardGenerator(
        obs_dim=len(FEATURES_TO_CONSIDER),
        fre_network=rg_model,
        min_num_anchors=MIN_NUM_ANCHORS,
        max_num_anchors=MAX_NUM_ANCHORS,
    )

    resampler = RNDResampling(state_dim=2)
    rnd_dataset = dataset_trajectories[..., :2].reshape(-1, 2)
    resampler_losses = resampler.fit(rnd_dataset, epochs=1000)
    resampling_weights = resampler.get_resampling_weights(rnd_dataset, alpha=1.2)
    reward_generator.resampling_weights = resampling_weights


    vae_loss, vae_kl_loss = [], []





    print('VAE training...')
    for step in tqdm(range(args.training_epochs), desc='VAE training', leave=False):
        
        vae_loss_dict = reward_generator.train_step_VAE(
            args=args,
            batch_size=256,
            min_num_anchors=MIN_NUM_ANCHORS,
            max_num_anchors=MAX_NUM_ANCHORS,
            from_new_states=True,
            num_states=MAX_NUM_ANCHORS+1,
            non_anchor_coef=0.0,
            anchors_from_same_trajectory=False
        )
        vae_loss.append(vae_loss_dict['loss'])
        vae_kl_loss.append(vae_loss_dict['kl_loss'])   
        
        if step % 100 == 0:
            clear_output(True)
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            axs[0].plot(vae_loss)
            axs[0].set_ylim([0, 0.5])
            axs[0].set_xscale('log')
            axs[1].plot(vae_kl_loss)
            axs[1].set_ylim([0, 1])
            plt.savefig(f"{args.LOGS_FOLDER}/reward_generator_loss.png")
    
        if i % 1000 == 0:
            torch.save(rg_model.state_dict(), f"{MODEL_SAVE_FOLDER}/reward_generator.pth")
            
    torch.save(rg_model.state_dict(), f"{MODEL_SAVE_FOLDER}/reward_generator.pth")  
        
    
    
    eval_num_envs = 16
    num_anchors = 16

    mask = reward_generator.generate_boolean_mask(args, eval_num_envs, len(FEATURES_TO_CONSIDER), p=0.0)
    mask = torch.zeros_like(mask)
    mask[..., [0, 1]] = 1.
    with torch.no_grad():
        w, info = reward_generator.get_z_from_random_anchors(eval_num_envs, min_num_anchors=num_anchors, max_num_anchors=num_anchors, anchors_from_same_trajectory=False, mask=mask)
        
    fig, axs, info = visualize_rewards_and_trajectories(w, reward_generator, mask=mask)

    plt.savefig(f"{args.LOGS_FOLDER}/Random rewards.png")
    
    
    ################################################################################################
    

    dataset_trajectories_cuda = dataset_trajectories.to(device)


    re_batch_size = args.re_batch_size
    num_random_samples = args.num_random_samples
    num_trajectories, len_trajectory, obs_len = dataset_trajectories.shape



    reward_encoder = RewardFunctionEncoder(obs_len=len(FEATURES_TO_CONSIDER), num_heads=2, num_layers=2).to(device)
    reward_encoder_optimizer = torch.optim.Adam(reward_encoder.parameters(), lr=0.001)
    reward_encoder_loss = []

    

    for epoch in tqdm(range(args.training_epochs_re), desc='Reward Alignement Traning'):
        
        # start_states = dataset_trajectories[
        #     torch.randint(0, num_trajectories, (re_batch_size,)), 
        #     torch.randint(0, len_trajectory, (re_batch_size,)), 
        #     :2
        # ]
        
        start_states = torch.tensor([[0.01172762, 0.14960599]]).repeat(re_batch_size, 1)
        
        # get a random reward function from the reward generator:
        with torch.no_grad():
            mask = reward_generator.generate_boolean_mask(args, re_batch_size, len(FEATURES_TO_CONSIDER), p=args.vae_dropout_p)
            if args.keep_only_coords:
                mask = torch.zeros_like(mask)
                mask[:, [0, 1]] = 1.
            reward_w, info = reward_generator.get_z_from_random_anchors(re_batch_size, min_num_anchors=8, max_num_anchors=8, anchors_from_same_trajectory=False, mask=mask)
        
        
        # Get the representation of the reward function based on random states:
        abs_states_idx = reward_generator.get_importance_sampling_indices(re_batch_size*num_random_samples)
        trajectories_idx, states_idx = abs_states_idx // len_trajectory, abs_states_idx % len_trajectory
        random_states = dataset_trajectories[trajectories_idx, states_idx] # get the random states
        random_states = random_states.reshape(re_batch_size, num_random_samples, len(FEATURES_TO_CONSIDER)).to(device)


        # Calculate the rewards of the selected random states using the trained Reward Function Generator:
        with torch.no_grad():
            random_states_rewards = reward_generator.get_reward(
                random_states.reshape(-1, len(FEATURES_TO_CONSIDER)) * mask.repeat(num_random_samples, 1).to(device),
                reward_w.unsqueeze(1).repeat(1, num_random_samples, 1).reshape(-1, Z_DIM)
            )
        random_states_rewards = random_states_rewards.reshape(re_batch_size, num_random_samples)

        # Get the new representation of the reward function:
        pad_mask = torch.full((re_batch_size, num_random_samples), fill_value=False, device=device)
        new_reward_w, _ = reward_encoder.get_transformer_encoding(
            start_states.to(device),
            random_states, 
            random_states_rewards, 
            pad_mask=pad_mask
        )
        

        # Get the best trajectory:
        best_traj_w, get_best_trajectory_info = get_best_trajectory(start_states, reward_w, reward_generator, pre_computed_zs)
        
        loss = (best_traj_w - new_reward_w).pow(2).sum(-1).mean()
        

        reward_encoder_optimizer.zero_grad()
        loss.backward()
        reward_encoder_optimizer.step()


        reward_encoder_loss.append(loss.item())
        
        
        if epoch % 10 == 0:
            clear_output(True)
            fig, ax = plt.subplots(1, 1, figsize=(5, 4))
            ax.plot(reward_encoder_loss)
            ax.set_xscale('log')
            plt.savefig(f"{LOGS_FOLDER}/reward_alignement_loss.png")
            
        if i % 1000 == 0:
            torch.save(reward_encoder.state_dict(), f"{MODEL_SAVE_FOLDER}/reward_encoder.pth")
            
    torch.save(reward_encoder.state_dict(), f"{MODEL_SAVE_FOLDER}/reward_encoder.pth")  
            
        
    
    ################################################################################################
    fig, all_axs = plt.subplots(len(benchmarks), 6, figsize=(6*5, len(benchmarks)*4))

    reconstruction_losses = []

    for benchmark_id in range(len(benchmarks)):
    # for benchmark_id in range():
        
        axs = all_axs[benchmark_id]
        # benchmark_id = 11

        benchmark_reward_function, benchmark_test_label, benchmark_param = benchmarks[benchmark_id]
        
        print(benchmark_test_label)

        reward_z_list, start_states_list, produced_trajectories_list, best_trajectories_idx_list, best_trajectories_bench_idx_list, best_trajectories_from_alignement, info = run_test(
            benchmark_id, num_evals=5, num_eval_anchors=args.num_random_samples, 
            frenetwork=frenetwork, reward_encoder=reward_encoder, reward_generator=reward_generator, pre_computed_zs=pre_computed_zs
        )



        
        states = dataset_trajectories[0:300, :1000:10].reshape(-1, obs_dim)
        states = states.to(device)
            
        zi = reward_z_list[0].unsqueeze(0).repeat(states.shape[0], 1)
        with torch.no_grad():
            states_rewards = reward_generator.get_reward(states[..., FEATURES_TO_CONSIDER], zi).cpu()
        
        eval_states, eval_states_rewards = states.cpu(), states_rewards.cpu().flatten()

        

        axs[0].set_title(f'{benchmark_test_label}')
        axs[1].set_title('Random anchors')
        axs[2].set_title('Reconstructed reward function')
        axs[3].set_title('Alignement trajectory')
        axs[4].set_title('Best trajectory')
        axs[5].set_title('Agent trajectory')
        
        add_largest_maze_walls(axs[0])
        add_largest_maze_walls(axs[1])
        add_largest_maze_walls(axs[3])
        add_largest_maze_walls(axs[4])
        add_largest_maze_walls(axs[2])
        add_largest_maze_walls(axs[5])

        # x = dataset_trajectories[get_best_trajectory_info['best_traj_idx'][0][0]]
        # x = dataset_trajectories[best_trajectories_idx_list[i]]
        # axs[i].scatter(x[:, 0], x[:, 1], c='orange', s=1, alpha=np.linspace(0.5, 1, 1001))

        for j in range(5):
            x = dataset_trajectories[best_trajectories_bench_idx_list[0][j]]
            axs[4].scatter(x[:, 0], x[:, 1], s=1, c=np.linspace(0.0, 0.8, 1001), cmap='hot', vmin=-1, vmax=1)
            
        
        for j in range(5):
            x = dataset_trajectories[best_trajectories_from_alignement[j]]
            axs[3].scatter(x[:, 0], x[:, 1], s=1, c=np.linspace(0.0, 0.8, 1001), cmap='hot', vmin=-1, vmax=1)
    
            
            

        anchors, anchors_rewards = info['anchors'], info['anchors_rewards']
        axs[1].scatter(anchors[..., 0].cpu(), anchors[..., 1].cpu(), c=anchors_rewards.cpu()[0], vmin=-1, vmax=1)    

        rews = benchmark_reward_function(denormalize_dataset_coords(dataset_trajectories[:, ::50]), benchmark_param)
        axs[0].scatter(dataset_trajectories[:, ::50, 0], dataset_trajectories[:, ::50, 1], c=rews, vmin=-1, vmax=1)

        # Reconstruction loss:
        true_rewards = benchmark_reward_function(denormalize_dataset_coords(eval_states), benchmark_param)
        reconstruction_loss = (eval_states_rewards - true_rewards).abs().mean().item()
        print(f'reconstruction_loss: {reconstruction_loss}')
        
        reconstruction_losses.append(reconstruction_loss)
        
        
        
        ################################################################################################################################
        
        anchors = info['anchors'].to(device)
        anchors_rewards = benchmark_reward_function(denormalize_dataset_coords(anchors.cpu()), benchmark_param).to(device)
        pad_mask = info['pad_mask'].to(device)

        mask = reward_generator.generate_boolean_mask(args, 1, len(FEATURES_TO_CONSIDER), p=0.0)
        if args.keep_only_coords:
            mask = torch.zeros_like(mask)
            mask[..., [0, 1]] = 1.
        mask = mask.to(device)
        anchors = anchors * mask.unsqueeze(1).repeat(1, args.num_random_samples, 1)

        eval_z, _ = reward_generator.get_z_from_anchors(anchors, anchors_rewards, pad_mask)

        with torch.no_grad():
            states_rewards = reward_generator.get_reward(
                eval_states[..., FEATURES_TO_CONSIDER].to(device) * mask.repeat(eval_states.shape[0], 1), 
                # eval_states[..., FEATURES_TO_CONSIDER].to(device), 
                eval_z.repeat(30000, 1)
            ).cpu()
            
        axs[2].scatter(eval_states[:, 0], eval_states[:, 1], c=states_rewards)
        
        ################################################################################################################################
        
        axs[5].scatter(produced_trajectories_list[:, :, 0], produced_trajectories_list[:, :, 1], c='blue', s=1, label='Agent trajectory')
        axs[5].scatter(start_states_list[0, 0], start_states_list[0, 1], c='red', marker='x')
        

        # break
            
    plt.savefig(f"{args.LOGS_FOLDER}/benchmark.png")
        
    print(f'mean reconstruction losses: {np.mean(reconstruction_losses)}')
    print(f'std reconstruction losses: {np.std(reconstruction_losses)}')

      
  
    
    return




def get_args():
    parser = argparse.ArgumentParser(description="Training and Evaluation Parameters")

    # Training parameters
    
    parser.add_argument('--training_epochs', type=int, default=100_000, help='Number of training vae epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for vae training')
    parser.add_argument('--min_num_anchors', type=int, default=16)
    parser.add_argument('--max_num_anchors', type=int, default=16)
    parser.add_argument('--vae_dropout_p', type=int, default=0.5)
    
    parser.add_argument('--training_epochs_re', type=int, default=10_000)
    parser.add_argument('--re_batch_size', type=int, default=32)
    parser.add_argument('--num_random_samples', type=int, default=200)
    parser.add_argument('--num_anchors_robust', type=int, default=64)
    
    
    
    
    parser.add_argument('--keep_only_coords', action='store_true', default=False, help='Whether to mask all eatures except coordinates')
        


    return parser.parse_args()



if __name__ == "__main__":
    args = get_args()
    print(args)
    
    for key, value in vars(args).items():
        print(f'{key}: {value}')
        
        
    now = datetime.now()
    date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    exp_name = f'keep_only_coords:{args.keep_only_coords}'
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