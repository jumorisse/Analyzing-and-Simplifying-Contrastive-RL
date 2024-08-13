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

"""Contrastive RL networks definition."""
import dataclasses
from typing import Optional, Tuple, Callable

from acme import specs
from acme.agents.jax import actor_core as actor_core_lib
from acme.jax import networks as networks_lib
from acme.jax import utils
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass
class ContrastiveNetworks:
  """Network and pure functions for the Contrastive RL agent."""
  policy_network: networks_lib.FeedForwardNetwork
  q_network: networks_lib.FeedForwardNetwork
  log_prob: networks_lib.LogProbFn
  repr_fn: Callable[Ellipsis, networks_lib.NetworkOutput]
  sample: networks_lib.SampleFn
  sample_eval: Optional[networks_lib.SampleFn] = None


def apply_policy_and_sample(
    networks,
    eval_mode = False):
  """Returns a function that computes actions by applying the policy network and sampling from the output distribution."""
  sample_fn = networks.sample if not eval_mode else networks.sample_eval
  if not sample_fn:
    raise ValueError('sample function is not provided')

  # if the sample_fn name is "greedy_selection", we are using the greedy actor
  if sample_fn.__name__ == "greedy_selection":
    def apply_and_sample(params, key, obs):
      # setting the action grid here, because function is jitted and needs to know the action grid shape at compile time
      # TODO: find better way to do this
      
      # 3x3 grid of actions
      '''
      action_grid = jnp.array([
        [[-1., -1.], [-1., 0.], [-1., 1.]],
        [[ 0., -1.], [ 0., 0.], [ 0., 1.]],
        [[ 1., -1.], [ 1., 0.], [ 1., 1.]],
        ])
      '''
      # 5x5 grid of actions
      action_grid = jnp.array([
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
      
      # apply the critic network to each action in the action grid to get the value of each action
      value_grid = jnp.apply_along_axis(lambda action: jax.nn.sigmoid(networks.q_network.apply(params, obs, action.reshape(1,-1))), 2, action_grid).reshape(action_grid.shape[0], action_grid.shape[1])

      return sample_fn(value_grid, key)
  
  # if the sample_fn is not "greedy_selection", we are using the parameterized or random actor
  else:
    def apply_and_sample(params, key, obs):
      return sample_fn(networks.policy_network.apply(params, obs), key)
    
  
  return apply_and_sample


def make_networks(
    spec,
    obs_dim,
    repr_dim = 64,
    repr_norm = False,
    repr_norm_temp = True,
    hidden_layer_sizes = (256, 256),
    actor_min_std = 1e-6,
    twin_q = False,
    use_image_obs = False,
    only_sa_encoder = False,
    sample_action = True,
    config = None):
  """Creates networks used by the agent."""

  num_dimensions = np.prod(spec.actions.shape, dtype=int)
  TORSO = networks_lib.AtariTorso  # pylint: disable=invalid-name

  # make config a global variable
  global training_config
  training_config = config

  def _unflatten_obs(obs):
    state = jnp.reshape(obs[:, :obs_dim], (-1, 64, 64, 3)) / 255.0
    goal = jnp.reshape(obs[:, obs_dim:], (-1, 64, 64, 3)) / 255.0
    return state, goal

  def _repr_fn(obs, action, hidden=None):
    # The optional input hidden is the image representations. We include this
    # as an input for the second Q value when twin_q = True, so that the two Q
    # values use the same underlying image representation.

    # obs contains both state and goal, therefore needs to be split
    if hidden is None:
      if use_image_obs:
        state, goal = _unflatten_obs(obs)
        img_encoder = TORSO()
        state = img_encoder(state)
        goal = img_encoder(goal)
      else:
        state = obs[:, :obs_dim]
        goal = obs[:, obs_dim:]
    else:
      state, goal = hidden

    ### Defining SA-Encoder ###
    # Simple MLP with relu activations and defined layer sizes + defined output size (given by representation dimension)
    sa_encoder = hk.nets.MLP(
        list(hidden_layer_sizes) + [repr_dim],
        w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
        activation=jax.nn.relu,
        name='sa_encoder')
    sa_repr = sa_encoder(jnp.concatenate([state, action], axis=-1)) # Is this called to trace the function?

    ### Defining G-Encoder ###
    # Simple MLP with relu activations and defined layer sizes + defined output size (given by representation dimension)
    if not only_sa_encoder:
      g_encoder = hk.nets.MLP(
          list(hidden_layer_sizes) + [repr_dim],
          w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
          activation=jax.nn.relu,
          name='g_encoder')
      g_repr = g_encoder(goal) # called to trace the function?
    # Experimental versions of the g_encoder, where the goal is encoded using the state-action encoder
    elif only_sa_encoder:
      if not sample_action:
        # use sa_encoder with fixed action [0,0] as the goal encoder
        g_repr = sa_encoder(jnp.concatenate([goal, jnp.zeros_like(action)], axis=-1))
      elif sample_action:
        key = jax.random.PRNGKey(0)
        action_key, key = jax.random.split(key)
        # currently only works for point_envs where actions are between [-1, 1]
        action = jax.random.uniform(key=action_key, shape=action.shape, minval=-1, maxval=1)
        # use sa_encoder with sampled action as the goal encoder
        g_repr = sa_encoder(jnp.concatenate([goal, action], axis=-1))

        # g_repr is negative of sa_repr
        #g_repr = -sa_repr(jnp.concatenate([goal, action], axis=-1))

    # Normalizing the representations
    # I think Eysenbach et al. found this to hurt performance and its off by default
    if repr_norm:
      sa_repr = sa_repr / jnp.linalg.norm(sa_repr, axis=1, keepdims=True)
      g_repr = g_repr / jnp.linalg.norm(g_repr, axis=1, keepdims=True)

      if repr_norm_temp:
        log_scale = hk.get_parameter('repr_log_scale', [], dtype=sa_repr.dtype,
                                     init=jnp.zeros)
        sa_repr = sa_repr / jnp.exp(log_scale)

    return sa_repr, g_repr, (state, goal)

  def _combine_repr(sa_repr, g_repr):
    """
    Function that computes the similarity between SA- and G-representations.
    Similarity is computed as the inner/dot product between the two representations.
    """
    return jax.numpy.einsum('ik,jk->ij', sa_repr, g_repr)

  def _critic_fn(obs, action=None):
    """
    Critic function.
    Unless twin_q==True, it simply returns the inner/dot product between the SA- and G-representations.
    """
    #print("Arguments passed to critic_fn: ", obs, action)
    # when critic is called as "policy" in the greedy actor, the first argument will be a tuple of (obs, action) and action will be None
    if type(obs) == tuple:
      try:
        obs, action = obs
        #print("Arguments unpacked: ", obs, action)
      except ValueError:
        raise ValueError("Critic function was called with invalid arguments")
    
    #print("Obs and action in critic_fn: ", obs, action)
    #print("Obs shape: ", obs.shape)
    #print("Action shape: ", action.shape)
    sa_repr, g_repr, hidden = _repr_fn(obs, action)
    outer = _combine_repr(sa_repr, g_repr)
    if twin_q:
      sa_repr2, g_repr2, _ = _repr_fn(obs, action, hidden=hidden)
      outer2 = _combine_repr(sa_repr2, g_repr2)
      # outer.shape = [batch_size, batch_size, 2]
      outer = jnp.stack([outer, outer2], axis=-1)
    return outer

  def _actor_fn(obs):
    # if using parameterized actor, we need an actor function that outputs a distribution for each action
    if config.actor == 'parameterized' or config.actor == 'random':
      if use_image_obs:
        state, goal = _unflatten_obs(obs)
        obs = jnp.concatenate([state, goal], axis=-1)
        obs = TORSO()(obs)
      network = hk.Sequential([
          hk.nets.MLP(
              list(hidden_layer_sizes),
              w_init=hk.initializers.VarianceScaling(1.0, 'fan_in', 'uniform'),
              activation=jax.nn.relu,
              activate_final=True),
          networks_lib.NormalTanhDistribution(num_dimensions,
                                              min_scale=actor_min_std),
      ])
      return network(obs)
    
    # if using greedy actor, we don't need an actor function
    # the greedy actor uses the critic function to select the action (implemented in GreedyAgent calss in utils.py and apply_policy_and_sample() above)
    elif config.actor == 'greedy':
      pass
    
    elif config.actor == 'random':
      # random action selection is taken care of action selection method of RandomAgent class in utils.py
      pass

    else:
      raise ValueError('Unknown actor type')

  def greedy_selection(value_grid, key):
    """
    Function that is used as sample and/or sample_eval functions when using the greedy actor.
    """
    action_grid = jnp.array([
        [[-1., -1.], [-1., -0.5], [-1., 0.], [-1., 0.5], [-1., 1.]],
        [[-0.5, -1.], [-0.5, -0.5], [-0.5, 0.], [-0.5, 0.5], [-0.5, 1.]],
        [[ 0., -1.], [ 0.,-0.5], [0., 0.], [0., 0.5], [ 0., 1.]],
        [[ 0.5, -1.], [ 0.5, -0.5], [0.5, 0.], [0.5, 0.5], [0.5, 1.]],
        [[ 1., -1.], [ 1., -0.5], [1., 0.], [1., 0.5], [1., 1.]],
        ])

    # find index of max value (in a flattened array of value_grid)
    max_index_1d = jnp.argmax(value_grid, axis=None)

    # convert the flattened index to 2D index
    max_index_2d = jnp.unravel_index(max_index_1d, value_grid.shape)
    

    # select one of the maximum value indices randomly
    #selected_index = max_indices[jax.random.randint(key, shape=max_indices.shape[0])]
    action = action_grid[max_index_2d]

    # add batch dimension
    action = jnp.expand_dims(action, axis=0)
    return action

  critic = hk.without_apply_rng(hk.transform(_critic_fn))
  repr_fn = hk.without_apply_rng(hk.transform(_repr_fn))
  if config.actor == 'greedy':
    # we set the policy network to be the critic network
    # this way we can access the critic network's output in the greedy actor implemented in builder.py
    print("##################################################")
    print("Using Greedy Actor")
    print("##################################################")
    #policy = critic
    policy = hk.without_apply_rng(hk.transform(_actor_fn))
  else:
    policy = hk.without_apply_rng(hk.transform(_actor_fn))

  # Create dummy observations and actions to create network parameters.
  dummy_action = utils.zeros_like(spec.actions)
  dummy_obs = utils.zeros_like(spec.observations)
  dummy_action = utils.add_batch_dim(dummy_action)
  dummy_obs = utils.add_batch_dim(dummy_obs)

  if config.actor == 'greedy':
    # when using the greedy actor, we set the policy network to be the critic network
    # this way we can access the critic network within the greedy actor implemented in builder.py
    return ContrastiveNetworks(
        policy_network=networks_lib.FeedForwardNetwork(
            lambda key: critic.init(key, dummy_obs, dummy_action), critic.apply),
        q_network=networks_lib.FeedForwardNetwork(
            lambda key: critic.init(key, dummy_obs, dummy_action), critic.apply),
        repr_fn=repr_fn.apply,
        log_prob=lambda params, actions: params.log_prob(actions),
        sample= greedy_selection,
        sample_eval=greedy_selection
        )
  else:
    return ContrastiveNetworks(
        policy_network=networks_lib.FeedForwardNetwork(
            lambda key: policy.init(key, dummy_obs), policy.apply),
        q_network=networks_lib.FeedForwardNetwork(
            lambda key: critic.init(key, dummy_obs, dummy_action), critic.apply),
        repr_fn=repr_fn.apply,
        log_prob=lambda params, actions: params.log_prob(actions),
        sample=lambda params, key: params.sample(seed=key),
        sample_eval=lambda params, key: params.mode(),
        )
