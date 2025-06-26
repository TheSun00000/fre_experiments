import torch
import opensimplex
import numpy as np
from tqdm import tqdm


# Goal reaching rewards:
def goal_reaching_reward(state, goal):
    return (torch.norm(state[..., :2] - goal, p=2, dim=-1) < 2).long() * 2 - 1



# Simplex rewards:
import opensimplex

class SimplexRewardFunction: 
    def __init__(self, num_simplex):
    
        self.simplex_size = num_simplex
        self.simplex_seeds_pos = np.zeros((self.simplex_size, 36, 25))
        self.simplex_seeds_xvel = np.zeros((self.simplex_size, 36, 25))
        self.simplex_seeds_yvel = np.zeros((self.simplex_size, 36, 25))
        self.simplex_best_xy = np.zeros((self.simplex_size, 10, 2))
        print("Generating simplex seeds")
        xi = np.arange(36)
        yi = np.arange(25)  
        for r in tqdm(range(self.simplex_size)):
            opensimplex.seed(r)
            self.simplex_seeds_pos[r] = opensimplex.noise2array(x=xi/20.0, y=yi/20.0).T
            opensimplex.seed(r + self.simplex_size)
            self.simplex_seeds_xvel[r] = opensimplex.noise2array(x=xi/20.0, y=yi/20.0).T
            opensimplex.seed(r + self.simplex_size * 2)
            self.simplex_seeds_yvel[r] = opensimplex.noise2array(x=xi/20.0, y=yi/20.0).T

            best_topn = np.argpartition(self.simplex_seeds_pos[r].flatten(), -10)[-10:] # (10,)
            best_xy = np.array(np.unravel_index(best_topn, self.simplex_seeds_pos[r].shape)).T # (10, 2)
            self.simplex_best_xy[r] = best_xy
        self.simplex_seeds_xvel[np.abs(self.simplex_seeds_xvel) < 0.5] = 0
        self.simplex_seeds_yvel[np.abs(self.simplex_seeds_yvel) < 0.5] = 0
        
        self.simplex_seeds_pos = torch.tensor(self.simplex_seeds_pos)
        self.simplex_seeds_xvel = torch.tensor(self.simplex_seeds_xvel)
        self.simplex_seeds_yvel = torch.tensor(self.simplex_seeds_yvel)
        self.simplex_best_xy = torch.tensor(self.simplex_best_xy)
        
        
        
    def compute_reward(self, states, params):
        
        if isinstance(params, int):
            params = torch.full((*states.shape[:-1], 1), fill_value=params)
        
        assert len(states.shape) == len(params.shape), states.shape # (batch_size, obs_dim) OR (batch_size, num_pairs, obs_dim)
        
        simplex_id = params[..., 0].long()
        x = states[..., 0].long().clip(0, 35)
        y = states[..., 1].long().clip(0, 24)
        simplex = self.simplex_seeds_pos[simplex_id, x, y]
        simplex_xvel = self.simplex_seeds_xvel[simplex_id, x, y]
        simplex_yvel = self.simplex_seeds_yvel[simplex_id, x, y]
        rews = -1 + (simplex > 0.3).float() * 0.5
        xy_vels = states[..., 15:17] * 0.33820298
        rews += xy_vels[...,0] * simplex_xvel + xy_vels[...,1] * simplex_yvel
        # rews = (simplex > 0.3).float()
        
        rews = ((rews + 1) * 2).clip(-1, 1)
        
        return rews # (batch_size,) 
    


# Velocity rewards:

class VelocityRewardFunction:
    def __init__(self):
        """
        [0, 1] up
        [0, -1] down
        [0, 1] right
        [0, -1] left
        """
        pass
    
    def compute_reward(self, states, direction):
        
        if isinstance(direction, list):
            direction = torch.concat((
                torch.full((*states.shape[:-1], 1), fill_value=direction[0]),
                torch.full((*states.shape[:-1], 1), fill_value=direction[1])
            ), dim=-1)

        assert len(states.shape) == len(direction.shape), states.shape # (batch_size, obs_dim) OR (batch_size, num_pairs, obs_dim)
        xy_vels = states[..., 15:17] * 0.33820298
        
        
        return torch.sum(xy_vels * direction, dim=-1) # (batch_size,)
    
 
 
# Path rewards:   
    
class TestRewMatrix:
    def __init__(self):
        self.pos = torch.zeros((36, 25))
        self.xvel = torch.zeros((36, 25))
        self.yvel = torch.zeros((36, 25))

    def compute_reward(self, s, *args):
        rews = torch.zeros_like(s[..., 0]) # (batch, examples)
        # XY Vel Reward
        xy_vels = s[..., 15:17] * 0.33820298
        
        x = s[..., 0].long().clip(0, 35)
        y = s[..., 1].long().clip(0, 23)
        simplex = self.pos[x, y]
        simplex_xvel = self.xvel[x, y]
        simplex_yvel = self.yvel[x, y]
        rews = (simplex > 0.3).float() * 0.5
        # rews = (simplex > 0.3).float() * 2 - 1
        rews += xy_vels[...,0] * simplex_xvel + xy_vels[...,1] * simplex_yvel

        return rews


class TestRewPath(TestRewMatrix):
    def __init__(self):
        super().__init__()
        self.pos[3:21, 7:10] = 1
        self.xvel[3:21, 7:10] = -1

        self.pos[0:3, 3:10] = 1
        self.yvel[0:3, 3:10] = -1

        self.pos[0:18, 0:3] = 1
        self.xvel[0:18, 0:3] = 1
        
        
class TestRewLoop(TestRewMatrix):
    def __init__(self):
        super().__init__()
        self.pos[22:33, 14:18] = 1
        self.xvel[22:33, 14:18] = -1

        self.pos[21:, 0:3] = 1
        self.xvel[21:, 0:3] = 1

        self.pos[33:, 3:18] = 1
        self.yvel[33:, 3:18] = 1

        self.pos[18:21, 0:7] = 1
        self.yvel[18:21, 0:7] = -1
        
        
class TestRewMatrixEdges(TestRewMatrix):
    def __init__(self):
        super().__init__()
        self.pos[:3, :] = 1
        self.pos[-3:, :] = 1
        self.pos[:, :3] = 1
        self.pos[:, -3:] = 1

    def compute_reward(self, s, *args):
        rews = torch.zeros_like(s[..., 0]) # (batch, examples)
        
        x = s[..., 0].long().clip(0, 35)
        y = s[..., 1].long().clip(0, 23)
        simplex = self.pos[x, y]
        rews = (simplex > 0.3).float() * 2 - 1

        return rews
        
        