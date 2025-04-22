import torch
import torch.nn as nn
from utils.networks import FRENetwork


device = 'cuda' if torch.cuda.is_available() else 'cpu'


Z_DIM = 128

class RewardGenerator:
    def __init__(self, obs_dim, fre_network: FRENetwork, min_num_anchors, max_num_anchors, from_buffer, max_buffer_size=1e6):
        self.obs_dim = obs_dim
        
        self.fre_network = fre_network.to(device)
        self.optimimizer = torch.optim.Adam(self.fre_network.parameters(), lr=0.001)
        
        self.emperical_mean = torch.zeros((Z_DIM,), dtype=torch.float32, device=device)
        self.emperical_std  = torch.ones((Z_DIM,), dtype=torch.float32, device=device)    
    
        self.from_buffer = from_buffer
        self.new_states_buffer = None
        self.states_buffer = None
        self.max_buffer_size = max_buffer_size
        
        self.std = 1
        
        self.len_params = Z_DIM
        
        self.min_num_anchors = min_num_anchors
        self.max_num_anchors = max_num_anchors
        
    def update_states_buffer(self, new_states):

        # num_trajectories, len_trajectory, obs_dim = new_states.shape
        # for i in range(num_trajectories):
        #     if len(self.states_buffer) < self.max_buffer_size:
        #         self.states_buffer.append(new_states[i].cpu())
        #     else:
        #         j = random.randint(0, self.max_buffer_size-1)
        #         self.states_buffer[j] = new_states[i].cpu()
        
        if self.states_buffer is None:
            self.states_buffer = new_states
        else:
            self.states_buffer = torch.concatenate((self.states_buffer, new_states), dim=0)
        
    
    def update_new_states_buffer(self, new_states):
        if self.new_states_buffer is None:
            self.new_states_buffer = new_states
        else:
            self.new_states_buffer = torch.concatenate((self.new_states_buffer, new_states), dim=0)

    
    
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
    
    
    
    def get_intermediate_anchors(self, anchors, anchors_rewards, num_intermediate_anchors:int):
        
        batch_size, num_anchors = anchors.shape[0], anchors.shape[1]

        intermediate_anchors = []
        intermediate_rewards = []
        for b in range(batch_size):
            # alpha = torch.rand((num_intermediate_anchors, num_anchors-1, 1))
            alpha = torch.linspace(0, 1, num_intermediate_anchors+2)[1:-1].unsqueeze(-1).repeat(1, num_anchors-1).unsqueeze(-1)
            
            x = (1-alpha)*anchors[b, :-1] + alpha*anchors[b, 1:]
            x = x.permute(1, 0, 2)
            
            r = (1-alpha)*anchors_rewards[b, :-1] + alpha*anchors_rewards[b, 1:]
            r = r.permute(1, 0, 2)
            
            x = x.reshape(-1, 2)
            r = r.reshape(-1, 1)
            
            intermediate_anchors.append(x)
            intermediate_rewards.append(r)
            
        intermediate_anchors = torch.stack(intermediate_anchors)
        intermediate_rewards = torch.stack(intermediate_rewards)

        return intermediate_anchors, intermediate_rewards
        # anchors.shape, intermediate_anchors.shape, intermediate_rewards.shape
    
    def get_importance_sampling_indices(self, N):
        indices = torch.multinomial(self.resampling_weights, N, replacement=True)
        return indices
    
    def get_training_data(self, batch_size, min_num_anchors, max_num_anchors, num_states, num_intermediate_anchors=10, from_new_states=False):
        assert min_num_anchors <= max_num_anchors <= num_states

        obs_dim = self.obs_dim
        
        all_states = torch.zeros((batch_size, num_states, obs_dim), dtype=torch.float32)
        anchors = torch.zeros((batch_size, max_num_anchors, obs_dim), dtype=torch.float32)
        
        buffer = self.new_states_buffer if from_new_states else self.states_buffer
        assert buffer is not None
        
        num_trajectories, trajectory_length = buffer.shape[0], buffer.shape[1]
        
        # Get anchors:
        # trajectories_idx_ = torch.randint(0, num_trajectories, (batch_size, 1))
        trajectories_idx_ = self.get_importance_sampling_indices(batch_size).unsqueeze(-1)
        trajectories_idx = trajectories_idx_.repeat(1, max_num_anchors).reshape(-1)
        # states_idx       = torch.randint(0, buffer.shape[1], (max_num_anchors*batch_size,))
        states_idx       = torch.linspace(0, trajectory_length-1, max_num_anchors).long().repeat(batch_size)
        anchors = buffer[trajectories_idx, states_idx]
        anchors = anchors.reshape(batch_size, max_num_anchors, obs_dim)
        all_states[:, :max_num_anchors, :] = anchors
        
        # Get non anchors
        num_non_anchors = num_states - max_num_anchors
        # trajectories_idx = torch.randint(0, buffer.shape[0], (num_non_anchors*batch_size,))
        trajectories_idx = self.get_importance_sampling_indices(num_non_anchors*batch_size)
        states_idx       = torch.randint(0, buffer.shape[1], (num_non_anchors*batch_size,))
        non_anchors = buffer[trajectories_idx, states_idx]
        non_anchors = non_anchors.reshape(batch_size, num_non_anchors, 2)
        all_states[:, max_num_anchors:, :] = non_anchors
        
            
        rewards = torch.zeros((batch_size, num_states), dtype=torch.float32)
        pad_mask = torch.ones((batch_size, max_num_anchors), dtype=torch.bool)
        
        # Generate random number of anchors for each batch element
        num_anchors = torch.randint(min_num_anchors, max_num_anchors + 1, (batch_size,))
        
        reward_indices = torch.arange(num_states).unsqueeze(0)
        reward_mask = reward_indices < num_anchors.unsqueeze(1)
        rewards[reward_mask] = torch.linspace(0.3, 1, max_num_anchors).unsqueeze(0).repeat(batch_size, 1).reshape(-1)
        # rewards[reward_mask] = torch.exp(2*(rewards[reward_mask] - 1))
        rewards = rewards.unsqueeze(-1)
        
        pad_mask_indices = torch.arange(max_num_anchors).unsqueeze(0)
        pad_mask = pad_mask_indices < num_anchors.unsqueeze(1)
        pad_mask = ~pad_mask
        
        anchors_rewards = rewards[:, :max_num_anchors]
        
        
        intermediate_anchors, intermediate_rewards = self.get_intermediate_anchors(anchors, anchors_rewards, num_intermediate_anchors=num_intermediate_anchors)
        
        
        return (anchors, anchors_rewards, pad_mask), (all_states, rewards), {
            'trajectories_idx': trajectories_idx_.reshape(-1),
            'intermediate_anchors': intermediate_anchors,
            'intermediate_rewards': intermediate_rewards,
        }
    

        
    def train_step_VAE(self, batch_size, min_num_anchors, max_num_anchors, num_states, from_new_states=False, non_anchor_coef=1):
        self.fre_network.train()
        
        (anchors, anchors_rewards, pad_mask), (all_states, rewards), info = self.get_training_data(
            batch_size=batch_size, 
            min_num_anchors=min_num_anchors, 
            max_num_anchors=max_num_anchors,
            num_states=num_states,
            from_new_states=from_new_states
        )
        anchors = anchors.to(device)
        anchors_rewards = anchors_rewards.to(device)
        pad_mask = pad_mask.to(device)
        all_states = all_states.to(device)
        rewards = rewards.to(device)
        
        
        # print(anchors)
        # print(anchors.sum())
        w_mean, w_log_std = self.fre_network.get_transformer_encoding(anchors, anchors_rewards, pad_mask=pad_mask)
        
        
        # Calculate the loss:
        
        w = w_mean + torch.normal(0, 1, size=w_mean.shape, device=device) * torch.exp(w_log_std)
        # w = w_mean
        rewards_pred = self.fre_network.get_reward_pred(w, all_states)
        # inter_rewards_pred = self.fre_network.get_reward_pred(w, info['intermediate_anchors'].to(device))
        # rewards_pred = torch.clip(rewards_pred, 0, 1)
        
        is_anchor = (rewards != 0)
        # reward_pred_loss = ((rewards_pred - rewards)**2).mean()
        reward_pred_loss = ((rewards_pred[is_anchor] - rewards[is_anchor])**2).mean() + ((rewards_pred[~is_anchor] - rewards[~is_anchor])**2).mean() * non_anchor_coef
        # reward_pred_loss += ((inter_rewards_pred - info['intermediate_rewards'].to(device))**2).mean()
        # print(info['intermediate_anchors'].shape, info['intermediate_rewards'].shape, all_states.shape)
        # reward_pred_loss = ((rewards_pred[ is_anchor] - 1)**2).mean() + ((rewards_pred[~is_anchor] - 0)**2).mean() * non_anchor_coef
        # reward_pred_loss = ((rewards_pred[ is_anchor] - rewards[ is_anchor])**2).mean() + \
                        #    ((rewards_pred[~is_anchor] - rewards[~is_anchor])**2).mean() * non_anchor_coef
        # reward_pred_loss = ((rewards_pred - anchors_rewards)**2).mean()
                        
        
        kl_loss = -0.5 * (1 + 2*w_log_std - w_mean**2 - torch.exp(w_log_std)**2).mean()
        loss = reward_pred_loss + kl_loss * 0.01
        
        
        self.optimimizer.zero_grad()
        loss.backward()
        self.optimimizer.step()
        
        return {
            'loss': loss.item(),
            'reward_pred_loss': reward_pred_loss.item(),
            'kl_loss': kl_loss.item(),
            'get_training_data:info': info
        }
        
        
    
    def estimate_mean_std(self, steps=10000, num_anchors=None):
        # assert len(self.states_buffer) > 0
        
        self.fre_network.eval()
        
        all_means, all_vars = [], []
        # for  _ in tqdm(range(steps), desc='Estimate_mean_std'):
        for  _ in range(steps):
            (anchors, anchors_rewards, pad_mask), (all_states, rewards), info = self.get_training_data(
                batch_size=1, 
                min_num_anchors=self.min_num_anchors if (num_anchors is None) else num_anchors, 
                max_num_anchors=self.max_num_anchors if (num_anchors is None) else num_anchors,
            )
            anchors = anchors.to(device)
            anchors_rewards = anchors_rewards.to(device)
            pad_mask = pad_mask.to(device)

            with torch.no_grad():
                w_mean, w_log_std = self.fre_network.get_transformer_encoding(anchors, anchors_rewards, pad_mask)   
                var = torch.exp(2*w_log_std)
            all_means.append(w_mean[0].cpu())
            all_vars.append(var[0].cpu())

        emperical_mean = torch.stack(all_means).mean(dim=0).to(device)

        sigma_hat = (torch.stack(all_vars).to(device) + torch.stack(all_means).to(device)**2).mean(dim=0) - emperical_mean**2
        emperical_std = sigma_hat ** 0.5
        
        self.emperical_std  = emperical_std
        self.emperical_mean = emperical_mean
        
        return emperical_mean, emperical_std
    
    
    def get_z_from_prior(self, batch_size: int):
        eps = torch.normal(0, 1, (batch_size, self.emperical_mean.shape[0]), device=device)
        z = self.emperical_mean + eps * self.emperical_std
        return z, {}
    
    
    def get_z_from_random_anchors(self, batch_size: int, min_num_anchors:int, max_num_anchors:int):
        assert self.new_states_buffer is not None
        self.fre_network.train()
        
        (anchors, anchors_rewards, pad_mask), (all_states, rewards), info = self.get_training_data(
            batch_size=batch_size, 
            min_num_anchors=min_num_anchors, 
            max_num_anchors=max_num_anchors,
            from_new_states=True,
            num_states=max_num_anchors+1,
        )
        anchors = anchors.to(device)
        anchors_rewards = anchors_rewards.to(device)
        pad_mask = pad_mask.to(device)

        eval_z, _ = self.get_z_from_anchors(anchors, anchors_rewards, pad_mask)
        return eval_z, {'anchors': anchors.cpu(), 'anchors_rewards': anchors_rewards.cpu(), 'get_training_data:info': info}
        
            
    def get_z_from_anchors(self, anchors: torch.Tensor, anchors_rewards: torch.Tensor, pad_mask: torch.Tensor):    
        assert anchors.shape[:-1] == pad_mask.shape
        self.fre_network.train()
        
        batch_size = anchors.shape[0]
        
        with torch.no_grad():
            w_mean, w_log_std = self.fre_network.get_transformer_encoding(anchors, anchors_rewards, pad_mask) 
        
        eps = torch.normal(0, 1, (batch_size, self.emperical_mean.shape[0]), device=device)
        z = w_mean + eps * torch.exp(w_log_std)
        
        return z, {'anchors': anchors.cpu()}

        


class RNDModule(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )
        
    def forward(self, x):
        return self.model(x)

class RNDResampling:
    def __init__(self):
        self.current = RNDModule().to(device)
        self.current_optimizer = torch.optim.SGD(self.current.parameters())
        self.target = RNDModule().to(device)
        self.target.requires_grad_ = False
        self.rnd_losses = []


    def fit(self, dataset, epochs=1000, batch_size=16):

        for _ in range(epochs):
        
            x = dataset[torch.randint(0, dataset.shape[0], (batch_size,))]
            x = x.to(device)

            yc = self.current(x)
            with torch.no_grad():
                yt = self.target(x)

            loss = (yc - yt).pow(2).sum(-1).mean()

            self.current_optimizer.zero_grad()
            loss.backward()
            self.current_optimizer.step()
            
            self.rnd_losses.append(loss.item())
        
        return self.rnd_losses
    
    
    def get_resampling_weights(self, x):
        
        x = x.to(device)
        with torch.no_grad():
            yc = self.current(x)
            yt = self.target(x)
            w = (yc - yt).pow(2).sum(-1).cpu()


        w = w ** 0.8
        w = w / w.sum()

        return w