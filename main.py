import torch
import torch.nn as nn

from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output

from utils.envs import PointMaze, TorchWrapper
from utils.reward_generator import RewardGenerator, RNDResampling
from utils.networks import FRENetwork, ActorCriticContinuous
from utils.ppo_utils import collect_trajectories, shufffle_trajectory, ppo_optimization
from utils.logs import add_largest_maze_walls

import os
from datetime import datetime

print('[INFO] Finished imports')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

print('device:', device)





MIN_NUM_ANCHORS = 32
MAX_NUM_ANCHORS = 32
EPISODE_LENGTH = 400
NUM_RANDOM_STEPS = 200
STATE_SCALE = 10
Z_DIM = 128

DISCOUNT_FACTOR = 0.95

X1_RANGE, X2_RANGE = 0.25 * STATE_SCALE, 0.175 * STATE_SCALE
# X1_RANGE, X2_RANGE = 0.3 * STATE_SCALE, 0.3 * STATE_SCALE




now = datetime.now()
date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
LOGS_FOLDER = f'./logs/{date_time_str}'
MODEL_SAVE_FOLDER = f'./models/{date_time_str}'

# Create folder
os.makedirs(LOGS_FOLDER)
os.makedirs(MODEL_SAVE_FOLDER)





def train_on_new_states(
    env: TorchWrapper, 
    model: ActorCriticContinuous, 
    optimizer: torch.optim.Optimizer,
    reward_generator: RewardGenerator,
    num_policy_steps: int
):
    (anchors, anchors_rewards, pad_mask), (all_states, rewards), info = reward_generator.get_training_data(
        batch_size=num_envs, 
        min_num_anchors=MIN_NUM_ANCHORS, 
        max_num_anchors=MAX_NUM_ANCHORS,
        from_new_states=True,
        num_states=MAX_NUM_ANCHORS+1,
    )
    anchors = anchors.to(device)
    anchors_rewards = anchors_rewards.to(device)
    pad_mask = pad_mask.to(device)

    z, _ = reward_generator.get_z_from_anchors(anchors, anchors_rewards, pad_mask)

    trajectory, _ = collect_trajectories(env, z, model, n_steps=num_policy_steps, reward_generator=reward_generator)
    shuffled_trajectory = shufffle_trajectory(trajectory)
    for _  in range(4):
        entropy = ppo_optimization(shuffled_trajectory, model, optimizer, epochs=1, batch_size=512)
        
    avg_reward = trajectory['rewards'][len(trajectory['rewards'])//num_envs-1::len(trajectory['rewards'])//num_envs].mean()

        
    return avg_reward, entropy, {'anchors': anchors, 'trajectory': trajectory, 'get_training_data:info': info}


def filter_only_new_states(new, old, steps):
    assert len(new.shape) == len(old.shape) == 2
    
    obs_len = new.shape[-1]
    
    class DiffModel(nn.Module):
        def __init__(self, ):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(obs_len, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                # nn.Linear(512, 512),
                # nn.ReLU(),
                nn.Linear(512, 1),
                nn.Sigmoid()
            )
        def forward(self, x):
            return self.model(x)

    detector = DiffModel()
    detector_optim = torch.optim.Adam(detector.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    detector_losses = []

    # new = reward_generator.new_states_buffer.reshape(-1, 2)
    # old = reward_generator.states_buffer.reshape(-1, 2)

    for i in tqdm(range(steps), desc='filter new trajectories'):
        
        x = torch.concat(
            (
                new[torch.randint(0, new.shape[0], (512,))],
                old[torch.randint(0, old.shape[0], (512,))]
            ), dim=0
        )
        y = torch.concat(
            (
                torch.ones(512, dtype=torch.float32),
                torch.zeros(512, dtype=torch.float32),
            )
        )
        
        pred = detector(x).squeeze(-1)
        loss = criterion(pred, y)
        
        detector_optim.zero_grad()
        loss.backward()
        detector_optim.step()
        
        detector_losses.append(loss.item())
        
        # if i % 10 == 0:
        #     clear_output(True)
        #     plt.plot(detector_losses)
        #     plt.show()
    
    with torch.no_grad():
        old_score = detector(old).detach().cpu().reshape(-1)
        new_score = detector(new).detach().cpu().reshape(-1)
    cond = new_score > 0.8
    to_keep = new[cond]
    
    # plt.scatter(to_keep[:, 0], to_keep[:, 1], c=new_score[cond], vmin=0, vmax=1)
    # plt.scatter(old[:, 0], old[:, 1], c=old_score)
        
    return to_keep, {'loss': detector_losses, 'old_score':old_score, 'new_score':new_score}


def get_new_states(
    iterations: int,
    env: TorchWrapper, 
    model: ActorCriticContinuous, 
    reward_generator: RewardGenerator, 
    from_prior: bool,
    num_policy_steps=1000,
    num_random_steps=100,
):
    # Get new states:
    
    
    list_trajectory, list_random_states, list_get_new_states_info = [], [], []

    for _ in tqdm(range(iterations), desc='Get new states', leave=False):
        
        if from_prior:
            z, info = reward_generator.get_z_from_prior(num_envs)
        else:
            z, info = reward_generator.get_z_from_random_anchors(num_envs, reward_generator.min_num_anchors, reward_generator.max_num_anchors)
        
        trajectory, random_states = collect_trajectories(
            env, z, model, 
            n_steps=num_policy_steps, 
            reward_generator=reward_generator, 
            num_random_steps=num_random_steps
        )
        
        list_trajectory.append(trajectory)
        list_random_states.append(random_states)
        list_get_new_states_info.append(info)

    trajectory = {key: torch.concat([traj[key] for traj in list_trajectory], dim=0) for key in trajectory}
    info = {key: torch.concat([info[key] for info in list_get_new_states_info], dim=0) if 'info' not in key else [info[key] for info in list_get_new_states_info] for key in info}
    random_states = torch.concat(list_random_states, dim=0)
    
    return trajectory, random_states, info




def plot_logs():
    clear_output(True)
    fig, axs = plt.subplots(3, 3, figsize=(18, 15))
        
    axs[0, 0].plot(rewards_list_1)
    axs[0, 0].set_title('New functions reward')
    axs[0, 1].plot(entropies_1)
    axs[0, 1].set_title('New functions entropy')
        
    axs[1, 0].plot(rewards_list_2)
    axs[1, 0].set_title('Rehearsal reward')
    axs[1, 1].plot(entropies_2)
    axs[1, 1].set_title('Rehearsal entropy')
        
    axs[2, 0].plot(vae_loss)
    axs[2, 0].set_title('VAE loss')
    axs[2, 0].set_ylim([0, 0.5])
    axs[2, 1].plot(vae_kl_loss)
    axs[2, 1].set_title('VAE KL loss')
    axs[2, 1].set_ylim([0, 2])


    # viz_new_states_buffer = reward_generator.new_states_buffer.reshape(-1, 2)
    # viz_states_buffer = reward_generator.states_buffer.reshape(-1, 2)
    # viz_states_buffer = parcoured_states.reshape(-1, 2)
    
    if viz_old_state is not None:
        axs[0, 2].scatter(viz_old_state[:, 0], viz_old_state[:, 1], c='blue', alpha=0.1, s=10)
    if viz_new_state is not None:
        axs[0, 2].scatter(viz_new_state[:, 0], viz_new_state[:, 1], c='red', alpha=0.1, s=10)
        
    add_largest_maze_walls(axs[0, 2])
    axs[0, 2].set_xlim([-X1_RANGE, X1_RANGE])
    axs[0, 2].set_ylim([-X2_RANGE, X2_RANGE])
    axs[0, 2].set_title('States coverage')
    
    if viz_new_state is not None:
        axs[1, 2].scatter(viz_new_state[:, 0], viz_new_state[:, 1], c='blue', alpha=0.1, s=10)
    if anchors_list:
        anchors = torch.concat(anchors_list).reshape(-1, 2).cpu().detach()
        axs[1, 2].scatter(anchors[:, 0], anchors[:, 1], marker='x', c='red')
    add_largest_maze_walls(axs[1, 2])
    axs[1, 2].set_xlim([-X1_RANGE, X1_RANGE])
    axs[1, 2].set_ylim([-X2_RANGE, X2_RANGE])
    axs[1, 2].set_title('Training Anchors')
    
    
    if viz_random_state is not None:
        axs[2, 2].scatter(viz_random_state[:, 0], viz_random_state[:, 1], c='blue', s=10)
    if viz_policy_reaches is not None:
        axs[2, 2].scatter(viz_policy_reaches[:, 0], viz_policy_reaches[:, 1], c='orange', s=10)
    add_largest_maze_walls(axs[2, 2])
    axs[2, 2].set_xlim([-X1_RANGE, X1_RANGE])
    axs[2, 2].set_ylim([-X2_RANGE, X2_RANGE])
    axs[2, 2].set_title('Policy reaching')
        
    plt.savefig(f"{LOGS_FOLDER}/epoch_{epoch}.png")




# path = 'mazes/point_mass_maze_empty.xml'
path = 'mazes/point_mass_maze_hardest.xml'
base_env = PointMaze(path)

num_envs = 128
env = TorchWrapper(base_env, num_envs=num_envs)

# state, _ = env.reset()
# action = torch.zeros((128, 2)).float()
# next_state, reward, done, truncated, info = env.step(action)

# eval_num_envs = 16
# eval_env = TorchWrapper(base_env, num_envs=eval_num_envs)


print('[INFO] Env imported')




fre_network = FRENetwork(obs_len=2)
# fre_network.load_state_dict(torch.load('models/fre_network_flexible.pth'))

reward_generator = RewardGenerator(
    obs_dim=2,
    fre_network=fre_network,
    min_num_anchors=MIN_NUM_ANCHORS,
    max_num_anchors=MAX_NUM_ANCHORS,
    from_buffer=True
)
# reward_generator.states_buffer = torch.rand((33, 1000, 2)) * 0.6 - 0.3
# empericial_mean, empericial_std = reward_generator.estimate_mean_std(steps=1000)

# x = torch.rand((1, 10000, 2)) * 0.6 - 0.3
# x = x * STATE_SCALE
# x = torch.round((x + 0.3) / 0.6 * 32)

# x1_vals = np.linspace(-0.25, 0.25, 100) 
# x2_vals = np.linspace(-0.175, 0.175, 100)
# X1, X2 = np.meshgrid(x1_vals, x2_vals)
# state_x12 = np.column_stack((X1.ravel(), X2.ravel()))
# x = torch.tensor(state_x12, dtype=torch.float32) * STATE_SCALE
# x = x.reshape(1, -1, 2)

# reward_generator.update_new_states_buffer(x)
# reward_generator.update_states_buffer    (x)


model = ActorCriticContinuous(
    state_dim=4,
    z_dim=reward_generator.len_params,
    action_dim=2,
    actor_hidden_layers=[512, 512, 512, 512],
    critic_hidden_layers=[512, 512, 512, 512]
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

vae_loss, vae_kl_loss = [], []
rewards_list_1, entropies_1 = [], []
rewards_list_2, entropies_2 = [], []

viz_new_state, viz_old_state, viz_random_state = None, None, None
viz_random_state, viz_policy_reaches = None, None

states_buffer_history = []

anchors_list = []
parcoured_states = None

training_steps = 0








for epoch in tqdm(range(100)):
    
    ##### Explore new states using random actions ##################################################################
    from_prior = True if (reward_generator.new_states_buffer is None) else False
    print(f'Getting new states... from_prior={from_prior}')
    trajectory, random_states, get_new_states_info = get_new_states(
        5,
        env, model, reward_generator, 
        num_policy_steps=EPISODE_LENGTH, 
        num_random_steps=NUM_RANDOM_STEPS, 
        from_prior=from_prior
    )
    
    
    if reward_generator.new_states_buffer is not None:
        potential_targets = random_states[:, -1, :2]
        old_targets = reward_generator.new_states_buffer[:, -1, :2]
        
        _, filter_info = filter_only_new_states(potential_targets, old_targets, steps=2000)
        to_keep = filter_info['new_score'] > 0.8
    
        parcoured_states_and_random_states = torch.concat(
            (
                trajectory['states'].reshape(-1, EPISODE_LENGTH, 4)[to_keep],
                random_states.reshape(-1, NUM_RANDOM_STEPS, 4)[to_keep]
            ),
            dim=1
        )
    
    
    else:
        parcoured_states_and_random_states = torch.concat(
            (
                trajectory['states'].reshape(-1, EPISODE_LENGTH, 4),
                random_states.reshape(-1, NUM_RANDOM_STEPS, 4)
            ),
            dim=1
        )


    if reward_generator.new_states_buffer is not None:
        viz_random_state = reward_generator.new_states_buffer[:, -1, :2].reshape(-1, 2)
    viz_policy_reaches = trajectory['states'].reshape(-1, EPISODE_LENGTH, 4)[:, -1, :2].reshape(-1, 2)
    
    
    # reward_generator.update_states_buffer(parcoured_states_and_random_states)
    reward_generator.update_new_states_buffer(parcoured_states_and_random_states[..., :2])
    # reward_generator.new_states_buffer = parcoured_states_and_random_states[..., :2]
    
    
    resampler = RNDResampling()
    dataset = reward_generator.new_states_buffer[:, -1, :].reshape(-1, 2)
    resampler_losses = resampler.fit(dataset, epochs=1000)
    resampling_weights = resampler.get_resampling_weights(dataset)
    reward_generator.resampling_weights = resampling_weights
    
    
    # Remove the random first please:
    # if training_steps == 1:
    #     reward_generator.new_states_buffer = reward_generator.new_states_buffer[640:]
    
    
    # break
    

    viz_old_state = viz_new_state
    # viz_new_state = trajectory['states'][..., :2].reshape(-1, 2)
    viz_new_state = reward_generator.new_states_buffer[:, -1, :2].reshape(-1, 2)
    # viz_random_state = random_states[..., :2].reshape(-1, 2)

    # break
    # continue
    
    plot_logs()

    
    
    ##### train the VAE: ###########################################################################################
    print('VAE training...')
    for _ in tqdm(range(2000), desc='VAE training', leave=False):
        vae_loss_dict = reward_generator.train_step_VAE(
            batch_size=512,
            min_num_anchors=MIN_NUM_ANCHORS,
            max_num_anchors=MAX_NUM_ANCHORS,
            from_new_states=True,
            num_states=64,
            non_anchor_coef=0.5,
        )
        vae_loss.append(vae_loss_dict['loss'])
        vae_kl_loss.append(vae_loss_dict['kl_loss'])    
    plot_logs()


    # ##### Estimate the emperical means and std to train the policy: ################################################
    # print('Mean and std estimation...')
    # empericial_mean, empericial_std = reward_generator.estimate_mean_std(steps=1000)    
    
    
    
    ##### Train the policy on new states (Exploration) #############################################################
    print('Policy training on new states...')
    
    model.action_var.data = torch.full((model.action_dim,), model.action_std**2, requires_grad=True, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    anchors_list = []
    for _ in tqdm(range(50), desc='Policy training', leave=False):
        avg_reward_1, entropy_1, info = train_on_new_states(env, model, optimizer, reward_generator, num_policy_steps=EPISODE_LENGTH)
        anchors_list.append(info['anchors'])
        rewards_list_1.append(avg_reward_1)
        entropies_1.append(entropy_1)
    plot_logs()
    
    
    
    # ##### Train the policy on learned reward functions (Rehearsal) #################################################
    # print('Policy rehearsal...')
    # for _ in range(10):
    #     avg_reward_2, entropy_2 = policy_rehearsal(env, model, reward_generator, num_policy_steps=EPISODE_LENGTH)
    #     rewards_list_2.append(avg_reward_2)
    #     entropies_2.append(entropy_2)
        
    

    # states_buffer_history.append((reward_generator.states_buff    er, reward_generator.new_states_buffer, filtered_new))

    # break
    
    
    training_steps += 1
    
    
    
    
    os.makedirs(f"{MODEL_SAVE_FOLDER}/epoch_{epoch}")
    torch.save(reward_generator.fre_network.state_dict(), f"{MODEL_SAVE_FOLDER}/epoch_{epoch}/fre_network.pth")
    torch.save(reward_generator.optimimizer.state_dict(), f"{MODEL_SAVE_FOLDER}/epoch_{epoch}/fre_network_optimizer.pth")
    torch.save(model.state_dict(), f"{MODEL_SAVE_FOLDER}/epoch_{epoch}/model.pth")
    torch.save(optimizer.state_dict(), f"{MODEL_SAVE_FOLDER}/epoch_{epoch}/optimizer.pth")
    torch.save({'new_states_buffer': reward_generator.new_states_buffer}, f"{MODEL_SAVE_FOLDER}/new_states_buffer.pth")


    os.system('nvidia-smi')
