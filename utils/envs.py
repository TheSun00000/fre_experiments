import torch
import random


import jax
import jax.numpy as jnp
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
import jax
from brax import base
from brax.io import mjcf
from dm_control import mujoco


STATE_SCALE = 10 # Pointmaze


class PointMaze(PipelineEnv):

    def __init__(self, path):

        backend='mjx' #'generalized'
        kwargs={}
        
        # self.path = 'mazes/point_mass_maze_empty.xml'
        # self.path = 'mazes/point_mass_maze_hardest.xml'
        self.path = path
        sys = mjcf.load(self.path)
        
        n_frames = 1
        kwargs['n_frames'] = kwargs.get('n_frames', n_frames)
        super().__init__(sys=sys, backend=backend, **kwargs)
        
        # path = epath.resource_path('brax') / 'envs/assets/half_cheetah.xml'
        
        
    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        _reset_noise_scale = 0.001

        low, hi = -_reset_noise_scale, _reset_noise_scale
        # qpos = self.sys.init_q + jax.random.uniform(
        #     rng1, (self.sys.q_size(),), minval=low, maxval=hi
        # )
        # qpos = self.sys.init_q
        qpos = jnp.array([0.025, 0.])
        # qpos = jnp.array([-0.2, 0.15])
        # print(self.sys.init_q, qpos)
        qvel = hi * jax.random.normal(rng2, (self.sys.qd_size(),))
        
        pipeline_state = self.pipeline_init(qpos, qvel)

        
        obs = self._get_obs(pipeline_state)
        reward, done, zero = jnp.zeros(3)
        metrics = {
            'x_position': zero,
            'y_position': zero,
        }
        return State(pipeline_state, obs, reward, done, metrics)
    
    def step(self, state: State, action: jax.Array) -> State:
        pipeline_state0 = state.pipeline_state
        assert pipeline_state0 is not None
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        obs = self._get_obs(pipeline_state)
        reward = 0

        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward)
    
    def _get_obs(self, pipeline_state):
        qpos, qvel = pipeline_state.qpos, pipeline_state.qvel
        return jnp.concatenate((
            qpos * STATE_SCALE,  
            qvel
            ), axis=-1)
        
        


class Ant(PipelineEnv):

  # pyformat: enable


  def __init__(
      self,
      ctrl_cost_weight=0.5,
      use_contact_forces=False,
      contact_cost_weight=5e-4,
      healthy_reward=1.0,
      terminate_when_unhealthy=True,
      healthy_z_range=(0.2, 1.0),
      contact_force_range=(-1.0, 1.0),
      reset_noise_scale=0.1,
      exclude_current_positions_from_observation=True,
      backend='generalized',
      **kwargs,
  ):
      
    backend='mjx'
    path = 'mazes/antmaze_hardest.xml'
    # path = 'mazes/ant.xml'
    sys = mjcf.load(path)

    n_frames = 5

    if backend in ['spring', 'positional']:
      sys = sys.tree_replace({'opt.timestep': 0.005})
      n_frames = 10

    if backend == 'mjx':
      sys = sys.tree_replace({
          'opt.solver': mujoco.mjtSolver.mjSOL_NEWTON,
          'opt.disableflags': mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
          'opt.iterations': 1,
          'opt.ls_iterations': 4,
      })

    if backend == 'positional':
      # TODO: does the same actuator strength work as in spring
      sys = sys.replace(
          actuator=sys.actuator.replace(
              gear=200 * jnp.ones_like(sys.actuator.gear)
          )
      )

    kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

    super().__init__(sys=sys, backend=backend, **kwargs)

    self._ctrl_cost_weight = ctrl_cost_weight
    self._use_contact_forces = use_contact_forces
    self._contact_cost_weight = contact_cost_weight
    self._healthy_reward = healthy_reward
    self._terminate_when_unhealthy = terminate_when_unhealthy
    self._healthy_z_range = healthy_z_range
    self._contact_force_range = contact_force_range
    self._reset_noise_scale = reset_noise_scale
    self._exclude_current_positions_from_observation = (
        exclude_current_positions_from_observation
    )

    if self._use_contact_forces:
      raise NotImplementedError('use_contact_forces not implemented.')
  
  def reset(self, rng: jax.Array) -> State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2 = jax.random.split(rng, 3)

    low, hi = -self._reset_noise_scale, self._reset_noise_scale
    # q = self.sys.init_q + jax.random.uniform(
    #     rng1, (self.sys.q_size(),), minval=low, maxval=hi
    # )
    qpos = self.sys.init_q
    # print(qpos.shape)
    qpos = qpos.at[:2].set(jnp.array([2.5, 0.0]))
    qvel = hi * jax.random.normal(rng2, (self.sys.qd_size(),))

    pipeline_state = self.pipeline_init(qpos, qvel)
    obs = self._get_obs(pipeline_state)

    reward, done, zero = jnp.zeros(3)
    metrics = {
        'x_position': zero,
        'y_position': zero,
    }
    return State(pipeline_state, obs, reward, done, metrics)

  def step(self, state: State, action: jax.Array) -> State:
    """Run one timestep of the environment's dynamics."""
    pipeline_state0 = state.pipeline_state
    assert pipeline_state0 is not None
    pipeline_state = self.pipeline_step(pipeline_state0, action)

    obs = self._get_obs(pipeline_state)
    reward = 0
    
    state.metrics.update(
        x_position=pipeline_state.x.pos[0, 0],
        y_position=pipeline_state.x.pos[0, 1],
    )
    
    return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward)

  def _get_obs(self, pipeline_state: base.State) -> jax.Array:
    """Observe ant body position and velocities."""
    qpos = pipeline_state.q
    qvel = pipeline_state.qd
    
    # if self._exclude_current_positions_from_observation:
    #   qpos = pipeline_state.q[2:]
    # qpos = pipeline_state.x.pos[0, 0]
    
    return jnp.concatenate([qpos] + [qvel])


        
from brax.io import torch as io_torch
from brax.envs.wrappers.training import VmapWrapper 

import random

cpu_device = jax.devices("cpu")[0]
gpu_device = jax.devices("gpu")[0]

class TorchWrapper:
    def __init__(self, env, num_envs, state_dim=None):
        
        self.env = VmapWrapper(env, batch_size=num_envs)
        self.num_envs = num_envs
        self.state_dim = state_dim if state_dim is not None else env.observation_size
        self.action_dim = env.action_size
        
        
        self.reset_fn = jax.jit(self.env.reset)
        self.step_fn = jax.jit(self.env.step)
        
    
    def reset(self, seed=None):
        
        rng = jax.random.PRNGKey(random.randint(0, 99999999))
        state = self.reset_fn(rng)
        self.state = state
        return io_torch.jax_to_torch(state.obs), {}
    
    def step(self, action: torch.Tensor):
        
        action = io_torch.torch_to_jax(action)
        action = jax.device_put(action, gpu_device)
        next_state = self.step_fn(self.state, action)
        observation, reward, done = next_state.obs, next_state.reward, next_state.done                
        observation = io_torch.jax_to_torch(observation)
        reward = io_torch.jax_to_torch(reward)
        done = io_torch.jax_to_torch(done)
        
        info = {
            # 'x_coordinate': float(next_state.pipeline_state.x.pos[0, 0][0]),
            # 'y_coordinate': float(next_state.pipeline_state.x.pos[0, 1][0]),
        }
        
        
        self.state = next_state
        truncated = torch.full_like(done, fill_value=False)
        
        
        
        return observation.cpu(), reward.cpu(), done.cpu(), truncated.cpu(), info
        