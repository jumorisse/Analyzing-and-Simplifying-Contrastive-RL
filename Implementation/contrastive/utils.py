# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for the contrastive RL agent."""
import functools
from typing import Dict
from typing import Optional, Sequence

from acme import types
from acme.agents.jax import actors
from acme.jax import networks as network_lib
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.jax.types import PRNGKey
from acme.utils.observers import base as observers_base
from acme.wrappers import base
from acme.wrappers import canonical_spec
from acme.wrappers import gym_wrapper
from acme.wrappers import step_limit
import dm_env
import env_utils
import jax
import numpy as np


def obs_to_goal_1d(obs, start_index, end_index):
  assert len(obs.shape) == 1
  return obs_to_goal_2d(obs[None], start_index, end_index)[0]


def obs_to_goal_2d(obs, start_index, end_index):
  assert len(obs.shape) == 2
  if end_index == -1:
    return obs[:, start_index:]
  else:
    return obs[:, start_index:end_index]


class SuccessObserver(observers_base.EnvLoopObserver):
  """Measures success by whether any of the rewards in an episode are positive.
  """

  def __init__(self):
    self._rewards = []
    self._success = []

  def observe_first(self, env, timestep
                    ):
    """Observes the initial state."""
    if self._rewards:
      success = np.sum(self._rewards) >= 1
      self._success.append(success)
    self._rewards = []

  def observe(self, env, timestep,
              action):
    """Records one environment step."""
    assert timestep.reward in [0, 1]
    self._rewards.append(timestep.reward)

  def get_metrics(self):
    """Returns metrics collected for the current episode."""
    return {
        'success': float(np.sum(self._rewards) >= 1),
        'success_1000': np.mean(self._success[-1000:]),
    }


class DistanceObserver(observers_base.EnvLoopObserver):
  """Observer that measures the L2 distance to the goal."""

  def __init__(self, obs_dim, start_index, end_index,
               smooth = True):
    self._distances = []
    self._obs_dim = obs_dim
    self._obs_to_goal = functools.partial(
        obs_to_goal_1d, start_index=start_index, end_index=end_index)
    self._smooth = smooth
    self._history = {}

  def _get_distance(self, env,
                    timestep):
    if hasattr(env, '_dist'):
      assert env._dist  # pylint: disable=protected-access
      return env._dist[-1]  # pylint: disable=protected-access
    else:
      # Note that the timestep comes from the environment, which has already
      # had some goal coordinates removed.
      obs = timestep.observation[:self._obs_dim]
      goal = timestep.observation[self._obs_dim:]
      dist = np.linalg.norm(self._obs_to_goal(obs) - goal)
      return dist

  def observe_first(self, env, timestep
                    ):
    """Observes the initial state."""
    if self._smooth and self._distances:
      for key, value in self._get_current_metrics().items():
        self._history[key] = self._history.get(key, []) + [value]
    self._distances = [self._get_distance(env, timestep)]

  def observe(self, env, timestep,
              action):
    """Records one environment step."""
    self._distances.append(self._get_distance(env, timestep))

  def _get_current_metrics(self):
    metrics = {
        'init_dist': self._distances[0],
        'final_dist': self._distances[-1],
        'delta_dist': self._distances[0] - self._distances[-1],
        'min_dist': min(self._distances),
    }
    return metrics

  def get_metrics(self):
    """Returns metrics collected for the current episode."""
    metrics = self._get_current_metrics()
    if self._smooth:
      for key, vec in self._history.items():
        for size in [10, 100, 1000]:
          metrics['%s_%d' % (key, size)] = np.nanmean(vec[-size:])
    return metrics


class ObservationFilterWrapper(base.EnvironmentWrapper):
  """Wrapper that exposes just the desired goal coordinates."""

  def __init__(self, environment,
               idx):
    """Initializes a new ObservationFilterWrapper.

    Args:
      environment: Environment to wrap.
      idx: Sequence of indices of coordinates to keep.
    """
    super().__init__(environment)
    self._idx = idx
    observation_spec = environment.observation_spec()
    spec_min = self._convert_observation(observation_spec.minimum)
    spec_max = self._convert_observation(observation_spec.maximum)
    self._observation_spec = dm_env.specs.BoundedArray(
        shape=spec_min.shape,
        dtype=spec_min.dtype,
        minimum=spec_min,
        maximum=spec_max,
        name='state')

  def _convert_observation(self, observation):
    return observation[self._idx]

  def step(self, action):
    timestep = self._environment.step(action)
    return timestep._replace(
        observation=self._convert_observation(timestep.observation))

  def reset(self):
    timestep = self._environment.reset()
    return timestep._replace(
        observation=self._convert_observation(timestep.observation))

  def observation_spec(self):
    return self._observation_spec


def make_environment(env_name, start_index, end_index,
                     seed, return_raw_gym_env = False):
  """Creates the environment.

  Args:
    env_name: name of the environment
    start_index: first index of the observation to use in the goal.
    end_index: final index of the observation to use in the goal. The goal
      is then obs[start_index:goal_index].
    seed: random seed.
  Returns:
    env: the environment
    obs_dim: integer specifying the size of the observations, before
      the start_index/end_index is applied.
  """
  np.random.seed(seed)
  gym_env, obs_dim, max_episode_steps = env_utils.load(env_name)
  goal_indices = obs_dim + obs_to_goal_1d(np.arange(obs_dim), start_index,
                                          end_index)
  indices = np.concatenate([
      np.arange(obs_dim),
      goal_indices
  ])
  env = gym_wrapper.GymWrapper(gym_env)
  if return_raw_gym_env:
    return gym_env, obs_dim
  env = step_limit.StepLimitWrapper(env, step_limit=max_episode_steps)
  env = ObservationFilterWrapper(env, indices)
  if env_name.startswith('ant_'):
    env = canonical_spec.CanonicalSpecWrapper(env)
  return env, obs_dim


class InitiallyRandomActor(actors.GenericActor):
  """Actor that takes actions uniformly at random until the actor is updated.
  """

  def select_action(self,
                    observation):
    # if all biases of first linear layer ar 0, no training has been done yet
    # in this case, replay tables are still being filled -> uniformly sample action
    if (self._params['mlp/~/linear_0']['b'] == 0).all():
      shape = self._params['Normal/~/linear']['b'].shape
      rng, self._state = jax.random.split(self._state)
      action = jax.random.uniform(key=rng, shape=shape,
                                  minval=-1.0, maxval=1.0)
    # else replay table contains min nr of samples -> choose action according to policy network output
    else:
      action, self._state = self._policy(self._params, observation,
                                         self._state)
    return utils.to_numpy(action)
  
class RandomActor(actors.GenericActor):
  """
  Actor that always uniformly samples actions.
  """
  def select_action(self, observation):
    # Randomly sample two actions between the lower and upper bounds
    lower = [-1.0,-1.0]
    upper = [1.0,1.0]
    action = np.random.uniform(lower, upper, 2)
    # convert to float tensor (previously seemed to be double)
    action = np.array(action, dtype=np.float32)

    return action
  
class GreedyActor(actors.GenericActor):
  """
  Actor that chooses the action with the highest critic value.
  Greedy selection implemented in the policy function in networks.py.
  """
  def __init__(self, config, actor_core, random_key, variable_client, adder, backend='cpu'):
    # initialize super class
    super().__init__(actor_core, random_key, variable_client, adder, backend='cpu')
    self._config = config

  def select_action(self, observation):
    # if config says to start with random actions, we fill the replay tables using random actions
    # the critic network is not updated until the replay tables are filled, until then all biases are 0
    if self._config.use_random_actor and (self._params['g_encoder/~/linear_0']['b'] == 0).all():
      action = self.get_random_action(observation)
    
    # with the set probability (epsilon), we choose a random action
    elif np.random.rand() < self._config.random_prob:
        action = self.get_random_action(observation)
    
    # otherwise, we chose an action greedily based on its critic value
    else:
      action, self._state = self._policy(self._params, observation,
                                         self._state)
      action = utils.to_numpy(action)

    #print("Action of greedy actor:", action)
    return action
  
  def get_random_action(self, observation, use_grid=False):
    if not use_grid:
      # Randomly sample two actions between the lower and upper bounds
      lower = [-1.0,-1.0]
      upper = [1.0,1.0]
      action = np.random.uniform(lower, upper, 2)
      # convert to float tensor (previously seemed to be double)
      action = np.array(action, dtype=np.float32)
      return action
    
    elif use_grid:
      '''
      # 3x3 grid of actions
      action_grid = jnp.array([
        [[-1., -1.], [-1., 0.], [-1., 1.]],
        [[ 0., -1.], [ 0., 0.], [ 0., 1.]],
        [[ 1., -1.], [ 1., 0.], [ 1., 1.]],
        ])
      
      '''
      # 5x5 grid of actions
      action_grid = np.array([
        [[-1., -1.], [-1., -0.5], [-1., 0.], [-1., 0.5], [-1., 1.]],
        [[-0.5, -1.], [-0.5, -0.5], [-0.5, 0.], [-0.5, 0.5], [-0.5, 1.]],
        [[ 0., -1.], [ 0.,-0.5], [0., 0.], [0., 0.5], [ 0., 1.]],
        [[ 0.5, -1.], [ 0.5, -0.5], [0.5, 0.], [0.5, 0.5], [0.5, 1.]],
        [[ 1., -1.], [ 1., -0.5], [1., 0.], [1., 0.5], [1., 1.]],
        ])
      
      # 9x9 grid of actions
      '''
      action_grid = jnp.array([
        [[-1., -1.], [-1., -0.75], [-1., -0.5], [-1., -0.25], [-1., 0.], [-1., 0.25], [-1., 0.5], [-1., 0.75], [-1., 1.]],
        [[-0.75, -1.], [-0.75, -0.75], [-0.75, -0.5], [-0.75, -0.25], [-0.75, 0.], [-0.75, 0.25], [-0.75, 0.5], [-0.75, 0.75], [-0.75, 1.]],
        [[-0.5, -1.], [-0.5, -0.75], [-0.5, -0.5], [-0.5, -0.25], [-0.5, 0.], [-0.5, 0.25], [-0.5, 0.5], [-0.5, 0.75], [-0.5, 1.]],
        [[-0.25, -1.], [-0.25, -0.75], [-0.25, -0.5], [-0.25, -0.25], [-0.25, 0.], [-0.25, 0.25], [-0.25, 0.5], [-0.25, 0.75], [-0.25, 1.]],
        [[ 0., -1.], [ 0., -0.75], [ 0., -0.5], [ 0., -0.25], [ 0., 0.], [ 0., 0.25], [ 0., 0.5], [ 0., 0.75], [ 0., 1.]],
        [[ 0.25, -1.], [ 0.25, -0.75], [ 0.25, -0.5], [ 0.25, -0.25], [ 0.25, 0.], [ 0.25, 0.25], [ 0.25, 0.5], [ 0.25, 0.75], [ 0.25, 1.]],
        [[ 0.5, -1.], [ 0.5, -0.75], [ 0.5, -0.5], [ 0.5, -0.25], [ 0.5, 0.], [ 0.5, 0.25], [ 0.5, 0.5], [ 0.5, 0.75], [ 0.5, 1.]],
        [[ 0.75, -1.], [ 0.75, -0.75], [ 0.75, -0.5], [ 0.75, -0.25], [ 0.75, 0.], [ 0.75, 0.25], [ 0.75, 0.5], [ 0.75, 0.75], [ 0.75, 1.]],
        [[ 1., -1.], [ 1., -0.75], [ 1., -0.5], [ 1., -0.25], [ 1., 0.], [ 1., 0.25], [ 1., 0.5], [ 1., 0.75], [ 1., 1.]],
        ])
      '''

      # Randomly sample one of the actions from the grid
      idx = np.random.randint(0, action_grid.shape[0])
      idy = np.random.randint(0, action_grid.shape[1])
      action = action_grid[idx, idy]
      # convert to float tensor (previously seemed to be double)
      action = np.array(action, dtype=np.float32)
      return action
  