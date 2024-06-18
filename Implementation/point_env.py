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

"""Utility for loading the 2D navigation environments."""
from typing import Optional

import gym
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.patches as patches


WALLS = {
    'Small':  # max_goal_dist = 3
        np.array([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]]),
    'Cross':  # max_goal_dist = 9
        np.array([[0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0]]),
    'FourRooms':  # max_goal_dist = 14
        np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]),
    'U':  # max_goal_dist = 14
        np.array([[0, 0, 0],
                  [0, 1, 0],
                  [0, 1, 0],
                  [0, 1, 0],
                  [1, 1, 0],
                  [0, 1, 0],
                  [0, 1, 0],
                  [0, 1, 0],
                  [0, 0, 0]]),
    'Spiral11x11':  # max_goal_dist = 45
        np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                  [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0],
                  [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
                  [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
                  [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
                  [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                  [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]]),
    'Maze11x11':  # max_goal_dist = 49
        np.array([[0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                  [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                  [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0],
                  [1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
}


def resize_walls(walls, factor):
  """
  Resizes the walls array by a given factor.

  Args:
    walls (numpy.ndarray): The input walls array.
    factor (int): The resizing factor.

  Returns:
    numpy.ndarray: The resized walls array.

  Raises:
    AssertionError: If the shape of the resized walls array is not equal to (factor * height, factor * width).
  """
  (height, width) = walls.shape
  row_indices = np.array([i for i in range(height) for _ in range(factor)])  # pylint: disable=g-complex-comprehension
  col_indices = np.array([i for i in range(width) for _ in range(factor)])  # pylint: disable=g-complex-comprehension
  walls = walls[row_indices]
  walls = walls[:, col_indices]
  assert walls.shape == (factor * height, factor * width)
  return walls


class PointEnv(gym.Env):
  """Abstract class for 2D navigation environments."""

  def __init__(self,
               walls = None, resize_factor = 1):
    """Initialize the point environment.

    Args:
      walls: (str or array) binary, H x W array indicating locations of walls.
        Can also be the name of one of the maps defined above.
      resize_factor: (int) Scale the map by this factor.
    """
    if resize_factor > 1:
      self._walls = resize_walls(WALLS[walls], resize_factor)
    else:
      self._walls = WALLS[walls]
    (height, width) = self._walls.shape
    self._height = height
    self._width = width
    self._action_noise = 0.01
    self.action_space = gym.spaces.Box(
        low=np.array([-1.0, -1.0]),
        high=np.array([1.0, 1.0]),
        dtype=np.float32)
    self.observation_space = gym.spaces.Box(
        low=np.array([0, 0, 0, 0]),
        high=np.array([height, width, height, width]),
        dtype=np.float32)
    self.reset()

  def _sample_empty_state(self):
    """
    Samples an empty state from the environment.

    Returns:
      state (np.array): The sampled empty state.
    """
    # Step 1: Find all candidate states where the walls are not present
    candidate_states = np.where(self._walls == 0)
    # Step 2: Get the number of candidate states
    num_candidate_states = len(candidate_states[0])
    # Step 3: Choose a random index from the candidate states
    state_index = np.random.choice(num_candidate_states)
    # Step 4: Get the coordinates of the chosen state
    state = np.array([candidate_states[0][state_index],
                      candidate_states[1][state_index]],
                     dtype=float)
    # Step 5: Add random noise to the state coordinates
    state += np.random.uniform(size=2)
    # Step 6: Check if the state is blocked by walls
    assert not self._is_blocked(state)
    # Step 7: Return the sampled empty state
    return state

  def _set_obs(self, obs):
    # split observation in the middle
    self.state = obs[:obs.shape[0] // 2]
    self.goal = obs[obs.shape[0] // 2:]

  def _get_obs(self):
    return np.concatenate([self.state, self.goal]).astype(np.float32)

  def reset(self):
    self.goal = self._sample_empty_state()
    self.state = self._sample_empty_state()
    return self._get_obs()

  def _discretize_state(self, state, resolution=1.0):
    """
    Discretizes the given state based on the specified resolution.
    Discretization is done by flooring the state values and clipping them to the walls shape.
    Example: state [1.34, 2.56] with resolution 1.0 will be discretized to [1, 2].

    Parameters:
      state (numpy.ndarray): The state to be discretized.
      resolution (float): The resolution used for discretization. Default is 1.0.

    Returns:
      numpy.ndarray: The discretized state.

    """
    ij = np.floor(resolution * state).astype(int)
    ij = np.clip(ij, np.zeros(2), np.array(self.walls.shape) - 1)
    return ij.astype(int)

  def _is_blocked(self, state):
    """
    Checks if the given state is blocked by a wall.

    Parameters:
    state (numpy.ndarray): The state to check for blockage.

    Returns:
    bool: True if the state is blocked, False otherwise.
    """
    assert len(state) == 2
    if (np.any(state < self.observation_space.low[:2])
      or np.any(state > self.observation_space.high[:2])):
      return True
    (i, j) = self._discretize_state(state)
    return (self._walls[i, j] == 1)

  def step(self, action):
    """
    Executes a step in the environment.

    Args:
      action (ndarray): The action to take in the environment.

    Returns:
      tuple: A tuple containing the following elements:
        - obs (ndarray): The observation of the environment after the step.
        - rew (float): The reward obtained from the step.
        - done (bool): A flag indicating if the episode is done.
        - info (dict): Additional information about the step.
    """
    # Make a copy of the action
    action = action.copy()

    # Clip the action if it is outside the action space
    if not self.action_space.contains(action):
      print('WARNING: clipping invalid action:', action)
    action = np.clip(action, self.action_space.low, self.action_space.high)
    assert self.action_space.contains(action)

    # Add noise to the action
    if self._action_noise > 0:
      action += np.random.normal(0, self._action_noise, (2,))

    # Clip the action again after adding noise
    action = np.clip(action, self.action_space.low, self.action_space.high)
    assert self.action_space.contains(action)

    # Perform multiple substeps to simulate continuous movement
    num_substeps = 10
    dt = 1.0 / num_substeps
    num_axis = len(action)
    for _ in np.linspace(0, 1, num_substeps):
      for axis in range(num_axis):
        # Calculate the new state based on the action and current state
        new_state = self.state.copy()
        new_state[axis] += dt * action[axis]

        # Check if the new state is blocked by walls
        if not self._is_blocked(new_state):
          self.state = new_state

    # Set the done flag to False since the episode is not done yet
    done = False

    # Get the observation after the step
    obs = self._get_obs()

    # Calculate the distance between the goal and the current state
    dist = np.linalg.norm(self.goal - self.state)

    # Assign a reward based on the distance to the goal
    rew = float(dist < 2.0)

    # Return the observation, reward, done flag, and additional information
    return obs, rew, done, {}
  
  def _get_env_img(self, color=True, scale=30):
    """
    Returns an image of the environment without any agent location.

    Args:
      color (bool): Whether to return the image in color (white path, black walls) or grayscale (gray path, black walls).
      scale (int): Factor by which original walls ndarray is scaled up.

    Returns: 
      np.ndarray: The image of the environment.
    """
    # Step 1: Scale the walls image
    img = resize_walls(self.walls, scale)
    # Step 2: Convert the walls image to grayscale
    if color:
      img = 1 - img
    else:
      img = 0.5 * (1 - img)
    # Step 3: Add color channels
    if color:
      img = np.stack([img, img, img], axis=-1)*255
    # Step 4: Return the resulting image
    return img
  
  def _draw_state(self, env_img, state, color=None, scale=30, radius=None):
    """
    Draws the agent's location on the environment image.
    If color is provided, the agent location will be colored in the specified color.

    Args:
      env_img (np.ndarray): The environment image (can already contain drawn states).
      state (np.ndarray): The agent's location.
      color (np.array): The RGB color to use for the agent location.

    Returns:
      np.ndarray: The environment image with the agent's location drawn.
    """
    # set radius to draw around agent coordinates
    if radius is None:
      radius = 1
    # get image shape
    (height, width, channels) = env_img.shape
    # calculate coordinates of the visible region in the walls image
    # i corresponds to the y-axis and j corresponds to the x-axis
    low_i, low_j = np.clip((state * scale).astype(int) - radius, [0, 0], [height, width])
    high_i, high_j = np.clip((state * scale).astype(int) + radius, [0, 0], [height, width])
    # set the visible region in the walls image to white or to specified color
    if color is not None:
      env_img[low_i:high_i, low_j:high_j, 0] = color[0]
      env_img[low_i:high_i, low_j:high_j, 1] = color[1]
      env_img[low_i:high_i, low_j:high_j, 2] = color[2]
    else:
      env_img[low_i:high_i, low_j:high_j, 0] = 255
      env_img[low_i:high_i, low_j:high_j, 1] = 255
      env_img[low_i:high_i, low_j:high_j, 2] = 255
    return env_img

  def _draw_action(self, env_img, state, action, color=None, scale=30):
    """
    Draws an action on the environment image using matplotlib.
    Actions are represented as arrows drawn from the agent's location.
    If color is provided, the action will be colored in the specified color.

    Args:
      env_img (np.ndarray): The environment image (can already contain drawn states).
      state (np.ndarray): The agent's location.
      action (np.ndarray): The action to draw.
      color (np.array): The RGB color to use for the action.
    
    Returns:
      np.ndarray: The environment image with the action drawn.
    """
    # Convert state and action to plot coordinates
    start_point = np.array(state) * scale
    end_point = start_point + (np.array(action) * scale)
    # convert points to integers
    start_point = start_point.astype(int)
    end_point = end_point.astype(int)

    # get all coordinates of points on the path from start to end
    x_values = [start_point[1]]
    y_values = [start_point[0]]
    dx = end_point[1] - start_point[1]
    dy = end_point[0] - start_point[0]
    steps = max(abs(dx), abs(dy))
    if steps > 0:
      x_step = dx / steps
      y_step = dy / steps
      for i in range(steps):
        x_values.append(x_values[-1] + x_step)
        y_values.append(y_values[-1] + y_step)
    x_values = np.clip(x_values, 0, env_img.shape[1] - 1)
    y_values = np.clip(y_values, 0, env_img.shape[0] - 1)

    # Draw the path on the environment image
    for i in range(len(x_values)):
      x = int(x_values[i])
      y = int(y_values[i])
      env_img[y, x, 0] = color[0]
      env_img[y, x, 1] = color[1]
      env_img[y, x, 2] = color[2]
    
    return env_img



  def _get_img(self, state, color=None, only_state=False):
    """
    Returns an image representation of the environment state.

    Args:
      state (np.ndarray): The current state of the environment.
      color (np.array): The RGB color to use for the visible region in the walls image.

    Returns:
      np.ndarray: The image representation of the environment state.
    """
    # Step 1: Scale the walls image
    scale = 30
    img = resize_walls(self.walls, scale)
    # set 

    # Step 2: Convert the walls image to grayscale
    #img = 0.5 * (1 - img)
    img = 1 - img # so that path is white and walls are black

    # Step 3: If color is not None, convert grayscale ndarray to RGB format
    img_shape = img.shape
    if color is not None:
      img = np.stack([img, img, img], axis=-1)*255

    # Step 3: Define the radius of the agent's visibility (not sure if this is the correct interpretation)
    radius = 1

    # Step 4: Calculate the coordinates of the visible region in the walls image
    low_i, low_j = np.clip((state * scale).astype(int) - radius, [0, 0], img_shape)
    high_i, high_j = np.clip((state * scale).astype(int) + radius, [0, 0], img_shape)

    # Step 5: Set the visible region in the walls image to white or to specified color
    if color is not None:
      # assign color values to respective image channels 
      img[low_i:high_i, low_j:high_j, 0] = color[0]
      img[low_i:high_i, low_j:high_j, 1] = color[1]
      img[low_i:high_i, low_j:high_j, 2] = color[2]
    else:
      img[low_i:high_i, low_j:high_j] = 1

    # Step 6: Resize the walls image to 64x64 pixels
    if color is None:
      (h, w) = img.shape
      img = (255 * img).astype(np.uint8)
      img = scipy.ndimage.zoom(img, (64 / h, 64 / w), order=0)
    else:
      pass

    # Step 7: Convert the walls image to RGB format
    if color is None:
      img = np.stack([img, img, img], axis=-1)*255

    # Step 8: Return the resulting image
    return img
  
  def visualize_states(self, states):
    """
    Visualizes the given states in the environment.

    Args:
      states (np.ndarray): The states to visualize.
    
    Returns:
      np.ndarray: The image representation of the environment containing the given states.
    """
    images = [self._get_img(state) for state in states]
    # overlay images
    img = np.zeros_like(images[0])
    for i in range(len(states)):
      img = np.maximum(img, images[i])
    return img
  
  def steps_to_center(self, state, method="dijkstra"):
    """
    Returns the min number of steps required to reach the center from the given state.
    Assumes that environment is a square grid and that step size is 1.

    Args:
      state (np.ndarray): The input state.
    
    Returns:
      int: The number of steps required to reach the center.
    """
    # discretize the state
    state = self._discretize_state(state)
    #scaled_walls = resize_walls(self._walls, 30)
    scaled_walls = self._walls
    center_coord = np.array([scaled_walls.shape[0] // 2, scaled_walls.shape[1] // 2])

    if self._is_blocked(state):
      print("State is blocked")
      return -1
    if self._is_blocked(center_coord):
      raise ValueError("Invalid Center: Center is blocked")
    
    if method == "dijkstra":
      # Dijkstra's algorithm
      # Create a 2D array to store the distances from the starting point to each point in the grid
      dist = np.full(scaled_walls.shape, np.inf)
      # Set the distance of the starting point to 0
      dist[center_coord[0], center_coord[1]] = 0
      # Create a 2D array to keep track of visited points
      visited = np.zeros(scaled_walls.shape, dtype=bool)
      # Start the main loop until the destination point is visited
      while not visited[state[0], state[1]]:
          # Find the point with the minimum distance from the starting point
          i, j = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
          # Mark the current point as visited
          visited[i, j] = True
          # Explore the neighbors of the current point
          for di, dj in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
              # Calculate the coordinates of the neighbor
              ni, nj = i + di, j + dj
              # Check if the neighbor is within the grid boundaries and is not visited or a wall
              if 0 <= ni < scaled_walls.shape[0] and 0 <= nj < scaled_walls.shape[1]:
                  if not visited[ni, nj] and not scaled_walls[ni, nj]:
                      # Update the distance of the neighbor if it's shorter than the current distance
                      dist[ni, nj] = min(dist[ni, nj], dist[i, j] + 1)
      # Return the distance of the destination point from the starting point
      return int(dist[state[0], state[1]]), center_coord

    

    

    

  @property
  def walls(self):
    return self._walls


class PointImage(PointEnv):
  """An image-based 2D navigation environment."""

  def __init__(self, *args, **kwargs):
    self._dist = []
    self._dist_vec = []
    super(PointImage, self).__init__(*args, **kwargs)
    self.observation_space = gym.spaces.Box(
        low=np.full((64*64*6), 0),
        high=np.full((64*64*6), 255),
        dtype=np.uint8)

  def reset(self):
    self._dist_vec = []
    self._dist = []
    self.goal = self._sample_empty_state()
    self._goal_img = self._get_img(self.goal)
    self.state = self._sample_empty_state()
    self._dist.append(np.linalg.norm(self.state - self.goal))
    return self._get_obs()

  def step(self, action):
    super(PointImage, self).step(action)
    dist = np.linalg.norm(self.state - self.goal)
    self._dist.append(dist)
    s = self._get_obs()
    r = float(dist < 2.0)
    done = False
    info = {}
    return s, r, done, info

  def _get_img(self, state):
    """
    Returns an image representation of the environment state.

    Args:
      state (np.ndarray): The current state of the environment.

    Returns:
      np.ndarray: The image representation of the environment state.
    """
    # Step 1: Scale the walls image
    scale = 30
    img = resize_walls(self.walls, scale)

    # Step 2: Convert the walls image to grayscale
    img = 0.5 * (1 - img)

    # Step 3: Define the radius of the agent's visibility (not sure if this is the correct interpretation)
    radius = 10

    # Step 4: Calculate the coordinates of the visible region in the walls image
    low_i, low_j = np.clip((state * scale).astype(int) - radius, [0, 0], img.shape)
    high_i, high_j = np.clip((state * scale).astype(int) + radius, [0, 0], img.shape)

    # Step 5: Set the visible region in the walls image to white
    img[low_i:high_i, low_j:high_j] = 1

    # Step 6: Resize the walls image to 64x64 pixels
    (h, w) = img.shape
    img = (255 * img).astype(np.uint8)
    img = scipy.ndimage.zoom(img, (64 / h, 64 / w), order=0)

    # Step 7: Convert the walls image to RGB format
    img = np.stack([img, img, img], axis=-1)

    # Step 8: Return the resulting image
    return img

  def _get_obs(self):
    return np.concatenate([
        self._get_img(self.state).flatten(),
        self._goal_img.flatten()
    ])
