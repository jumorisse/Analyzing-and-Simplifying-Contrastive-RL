import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import TransformedDistribution, TanhTransform, Normal
from experiment_utils import get_env, sample_tasks, load_encoders, load_policy_net, generate_env_states
import matplotlib.pyplot as plt
import matplotlib.patches as patches

basic_action_grid = np.array([
                        [[-1.,-1.], [-1.,0.], [-1.,1.]],
                        [[0.,-1.], [0.,0.], [0.,1.]],
                        [[1.,-1.], [1.,0.], [1.,1.]],
                        ])

def print_grid(grid):
    """
    Prints a numpy 3d array in a readable format.
    """
    print("Grid shape:", grid.shape)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            print(grid[i,j], end=" ")
        print()

def scale_action_grid(action_grid, factor):
    """
    Scales action grid up, i.e. creates a more fine-grained grid with more cells.
    Scaling up is done by interpolating neighboring (8-neighbors) cells.
    Factor is the number of times interpolation is done.

    Parameters:
        action_grid: 3D numpy array of actions, shape (3,3,2)
        factor: int, number of times to interpolate neighboring cells
    
    Returns:
        3D numpy array of tuples, scaled up action grid of shape (action rows, action cols, 2)
    """
    for _ in range(factor):
        current_shape = action_grid.shape
        new_shape = (current_shape[0]+current_shape[0]-1, current_shape[1]+current_shape[1]-1, current_shape[2])
        new_action_grid = np.zeros(new_shape, dtype=np.float32)
        # fill new grid with values from old grid
        for i in range(current_shape[0]):
            for j in range(current_shape[1]):
                new_action_grid[2*i, 2*j] = action_grid[i,j]
        # interpolate values
        # in even rows including (0), odd cells are interpolated by their row neighbors
        for i in range(0, new_shape[0], 2):
            for j in range(1, new_shape[1], 2):
                new_action_grid[i,j] = (new_action_grid[i,j-1] + new_action_grid[i,j+1]) / 2
        # odd rows are completely interpolated by their row neighbors
        for i in range(1, new_shape[0], 2):
            for j in range(new_shape[1]):
                new_action_grid[i,j] = (new_action_grid[i-1,j] + new_action_grid[i+1,j]) / 2
        action_grid = new_action_grid    
    print("Scaled 3x3x2 grid to ", action_grid.shape)
    print("Grid contains ", action_grid.size/2, " 2d actions.")
    return action_grid

def get_action_strengths(action_grid):
    """
    Calculates the strength of each action in the action grid.
    The strength of an action is the sum of the absolute values of the action tuple.

    Parameters:
        action_grid: 3D numpy array of actions, shape (action rows, action cols, 2)
    
    Returns:
        2D numpy array of floats, shape (action rows, action cols)
    """
    strength_grid = np.zeros((action_grid.shape[0], action_grid.shape[1]))
    for i in range(action_grid.shape[0]):
        for j in range(action_grid.shape[1]):
            strength_grid[i,j] = np.sum(np.abs(action_grid[i,j]))
    return strength_grid


def evaluate_performances(trajectories, agent_names):
    """
    Evaluates the performances of the agents on the tasks.

    Parameters:
        trajectories: List of lists of tuples, containing the trajectories of the agents on the tasks.
        agent_names: List of strings, containing the names of the agents.
    """
    success_rates = []
    # print the performances of each agent for each task (store whether it was successful or not)
    for i in range(len(trajectories)):
        print("Agent:", agent_names[i])
        agent_success_rates = []
        for j in range(len(trajectories[i])):
            final_reward = trajectories[i][j][-1][2]
            print("Task", j+1, ":", "Reached goal:", final_reward)
            agent_success_rates.append(final_reward)
            print()
        success_rates.append(agent_success_rates)

    # print summary, i.e. average success rate overall and of each agent
    print("Summary:")
    for i in range(len(trajectories)):
        print(agent_names[i], "success rate:", np.mean(success_rates[i]))


class ContrastiveCritic:
    def __init__(self, env, params_path):
        self.env = env
        self.params_path = params_path
        self.sa_encoder, self.g_encoder = self.load_encoder_nets()
    
    def load_encoder_nets(self):
        sa_encoder, g_encoder = load_encoders(self.env, self.params_path+"/critic_params.txt")

        # return networks in eval mode
        return sa_encoder.eval(), g_encoder.eval()

    def encode_sa(self, state, action):
        # concatenate state and action
        s_a = np.concatenate([state, action])

        # turn np.array into torch tensor
        s_a = torch.tensor(s_a, dtype=torch.float32)

        # encode state-action pair
        sa_encoding = self.sa_encoder(s_a).detach().numpy()

        return sa_encoding

    def encode_g(self, goal):
        # turn goal array into torch tensor
        goal = torch.tensor(goal, dtype=torch.float32)

        # encode goal
        g_encoding = self.g_encoder(goal).detach().numpy()


        return g_encoding


    def value_function(self, obs, action):
        # split obs into state and goal
        state = obs[:obs_dim]
        goal = obs[obs_dim:]

        # encode state-action pair
        sa_encoding = self.encode_sa(state, action)
        # encode goal
        g_encoding = self.encode_g(goal)

        # calculate value, i.d. inner product of the two encodings
        inner_product = np.dot(sa_encoding, g_encoding)
        # inner_product = np.einsum('ik,jk->ij', sa_encoding, g_encoding)

        return inner_product


class GreedyAgent:
    def __init__(self, env, action_grid, params_path, obs_dim, epsilon=0.05, value_type="contrastive_critic"):
        self.env = env
        self.params_path = params_path
        self.action_grid = action_grid
        self.value_type = value_type
        self.evaluator = self.load_evaluator()
        self.obs_dim = obs_dim
        self.epsilon = epsilon

    def load_evaluator(self):
        if self.value_type == "contrastive_critic":
            return ContrastiveCritic(self.env, self.params_path)

    def value_function(self, obs):
        """
        Value function for greedy selection.
        Produces an evaluation grid containing the values of each state-action pair given the current goal.
        """
        if self.value_type == "contrastive_critic":
            value_grid = np.zeros(self.action_grid.shape[:2])
            # iterating through every action-grid cell and updating its corresponding value-grid cell
            for row in range(self.action_grid.shape[0]):
                for column in range(self.action_grid.shape[1]):
                    value_grid[row][column] = self.evaluator.value_function(obs, self.action_grid[row][column])
        
        return value_grid


    def select_action(self, obs):
        """
        Selects an action from the action grid based on the critic values.
        Has an epsilon-greedy policy, i.e. with probability epsilon a random action is selected.

        Parameters:
            obs: The current onbservation, i.e. the current state and the goal
        
        Returns:
            A tuple representing the greedy action.
        """
        if np.random.rand() < self.epsilon:
            # select random action
            random_row_index = np.random.randint(self.action_grid.shape[0])
            random_column_index = np.random.randint(self.action_grid.shape[1])
            return self.action_grid[random_row_index, random_column_index]
        else:
            # produce value grid
            value_grid = self.value_function(obs)
            # identify indices of maximum value
            max_indices = np.argwhere(value_grid == np.max(value_grid))
            # select one of the maximum value indices randomly
            selected_index = max_indices[np.random.randint(max_indices.shape[0])]
            # return the corresponding action
            return self.action_grid[selected_index[0], selected_index[1]]

    def solve_task(self, task):
        """
        Gets a task (initial state and goal) and solves it using the greedy agent.

        Parameters:
            task: 2D numpy array of shape (1, 2*obs_dim)
        
        Returns:
            A tuple containing the trajectory and a boolean indicating whether the goal was reached.
        """
        # set task to be environment's current observation
        env._set_obs(task)

        # split task into intial state and goal
        start_state = task[:obs_dim]
        goal = task[obs_dim:]

        # initialize trajectory
        trajectory = []

        # reached goal flag
        reached_goal = False

        current_state = start_state
        # iterate through the task
        for i in range(max_episode_length):
            # select action
            action = self.select_action(np.concatenate((current_state, goal)))
            # apply action
            next_obs, reward, done, _ = env.step(action)
            next_state = next_obs[:obs_dim]
            # check if goal is reached
            if reward == 1:
                reached_goal = True
            # update start state
            current_state = next_state
            # append to trajectory
            trajectory.append((current_state, action, reward, done))
            # check if task is done
            if done:
                break
        
        return trajectory, reached_goal


class RandomAgent:
    def __init__(self, action_grid, obs_dim):
        self.action_grid = action_grid
        self.obs_dim = obs_dim

    def select_action(self, obs=None):
        """
        Selects an action randomly from the action grid.

        Returns:
            A tuple representing the random action.
        """
        random_row_index = np.random.randint(self.action_grid.shape[0])
        random_column_index = np.random.randint(self.action_grid.shape[1])
        return self.action_grid[random_row_index, random_column_index]
    
    def solve_task(self, task):
        # set task to be environment's current observation
        env._set_obs(task)

        # split task into intial state and goal
        start_state = task[:obs_dim]
        goal = task[obs_dim:]

        # initialize trajectory
        trajectory = []

        # reached goal flag
        reached_goal = False

        current_state = start_state
        # iterate through the task
        for i in range(max_episode_length):
            # select action
            action = self.select_action()
            # apply action
            next_obs, reward, done, _ = env.step(action)
            next_state = next_obs[:self.obs_dim]
            # check if goal is reached
            if reward == 1:
                reached_goal = True
            # update current state
            current_state = next_state
            # append to trajectory
            trajectory.append((current_state, action, reward, done))
            # check if task is done
            if done:
                break

        return trajectory, reached_goal


class TanhTransformedDistribution(torch.distributions.transformed_distribution.TransformedDistribution):
    """
    Distribution followed by tanh.
    PyTorch counterpart to TanhTransformedDistribution from acme.jax.networks.distributional.py
    """

    def __init__(self, distribution, threshold=0.999):
        """Initialize the distribution.

        Args:
          distribution: The distribution to transform.
          threshold: Clipping value of the action when computing the logprob.
        """
        super().__init__(
            distribution=distribution,
            transform=TanhTransform(),
        )
        self._threshold = threshold
        inverse_threshold = torch.tanh(threshold)
        log_epsilon = torch.log(1. - threshold)

        # Compute log probabilities for left and right tails
        self._log_prob_left = distribution.cdf(-inverse_threshold).log() - log_epsilon
        self._log_prob_right = distribution.icdf(torch.tensor(1. - threshold)).log() - log_epsilon

    def log_prob(self, value):
        clipped_value = torch.clamp(value, -self._threshold, self._threshold)
        log_prob = torch.where(
            clipped_value <= -self._threshold, self._log_prob_left,
            torch.where(clipped_value >= self._threshold, self._log_prob_right,
                        super().log_prob(clipped_value))
        )
        return log_prob

    def mode(self):
        """
        This function returns the mode of the distribution.
        """
        return self.base_dist.icdf(torch.tensor(0.))

    def entropy(self):
        return self.base_dist.entropy() + self.transform.log_abs_det_jacobian(self.base_dist.icdf(torch.tensor(0.)), torch.tensor(0.))


class ContrastiveActor:
    def __init__(self, params_path, obs_dim, eval_mode):
        self.params_path = params_path
        self.policy_net = self.load_net(params_path)
        self.obs_dim = obs_dim
        self.eval_mode = eval_mode
    
    def load_net(self, params_path):
        policy_net = load_policy_net(params_path+"/policy_params.txt")
        print("Policy Net Type:", type(policy_net))

        return policy_net.eval()

    def sample_action(self, policy_net_outputs, min_scale=1e-6):
        # the policy network 
        actions = []
        # assumes that first output corresponds to loc and second to scale
        for loc, scale in zip(policy_net_outputs[0], policy_net_outputs[1]):
            # apply softplus activation to scale (makes it non-negative) and add minimum scale
            scale = np.log(1 + np.exp(scale)) + min_scale
            # create PyTorch normal distribution
            dist = torch.distributions.Normal(loc, scale)
            # tanh transformed dist
            #tanh_dist = torch.distributions.independent.Independent(TanhTransformedDistribution(dist), reinterpreted_batch_ndims=1)
            transform = torch.distributions.transforms.TanhTransform()
            tanh_sample = transform(dist.sample())
            # store action sampled for current action dimension
            actions.append(tanh_sample.detach().numpy())
        
        return actions
    
    def get_mode_action(self, policy_net_outputs):
        actions = []
        # assumes that first output corresponds to loc and second to scale
        for loc, scale in zip(policy_net_outputs[0], policy_net_outputs[1]):
            # since the policy network outputs the mean of the normal distribution, the mode is the mean
            actions.append(loc)
        
        return actions
            

    def select_action(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32)
        network_output = self.policy_net(obs)
        network_output = [output.detach().numpy() for output in network_output]
        if self.eval_mode:
            sampled_action = self.get_mode_action(network_output)
        else:
            sampled_action = self.sample_action(network_output)

        return sampled_action

class ContrastiveAgent:
    def __init__(self, env, action_grid, params_path, obs_dim, eval_mode=False):
        self.env = env,
        self.obs_dim = obs_dim
        self.action_grid = action_grid
        self.eval_mode = eval_mode
        self.evaluator = ContrastiveCritic(env, params_path)
        self.actor = self.get_actor(params_path)
    
    def get_actor(self, params_path):
        return ContrastiveActor(params_path, self.obs_dim, self.eval_mode)

    def select_action(self, obs):
        return self.actor.select_action(obs)

    def solve_task(self, task):
        # set task to be environment's current observation
        env._set_obs(task)

        # split task into intial state and goal
        start_state = task[:obs_dim]
        goal = task[obs_dim:]

        # initialize trajectory
        trajectory = []

        # reached goal flag
        reached_goal = False

        # iterate through the task
        current_state = start_state
        for i in range(max_episode_length):
            # select action
            action = self.actor.select_action(np.concatenate((current_state, goal)))
            # apply action
            next_obs, reward, done, _ = env.step(action)
            next_state = next_obs[:self.obs_dim]
            # check if goal is reached
            if reward == 1:
                reached_goal = True
            # update start state
            current_state = next_state
            # append to trajectory
            trajectory.append((current_state, action, reward, done))
            # check if task is done
            if done:
                break

        return trajectory, reached_goal
    

def compare_actions(env, task, leading_agent, trailing_agent, obs_dim):
    """
    Compares the actions of two agents on a task.

    Parameters:
        task: 2D numpy array of shape (1, 2*obs_dim)
        leading_agent: An agent object, the leading agent whose actions determine the next state
        trailing_agent: An agent object, the trailing agent whose action selection is compared to the leading agent's
    
    Returns:
        Returns two lists, the first containing the euclidean distances between the actions and the second containing the cosine similarities.
    """
    # split task into intial state and goal
    start_state = task[:obs_dim]
    goal = task[obs_dim:]

    # initialize action stores
    leading_actions = []
    trailing_actions = []

    # attempt to solve task
    current_state = start_state
    for i in range(max_episode_length):
        # select action
        leading_action = leading_agent.select_action(np.concatenate((current_state, goal)))
        trailing_action = trailing_agent.select_action(np.concatenate((current_state, goal)))
        # store actions
        leading_actions.append(leading_action)
        trailing_actions.append(trailing_action)
        # apply leading action
        next_obs, _, done, _ = env.step(leading_action)
        next_state = next_obs[:obs_dim]

        # update start state
        current_state = next_state
        # check if task is done
        if done:
            break
    
    # converting all actions to numpy arrays
    leading_actions = [np.array(action) for action in leading_actions]
    trailing_actions = [np.array(action) for action in trailing_actions]

    # What are the euclidean distances between the actions of the two agents?
    distances = []
    for i in range(len(leading_actions)):
        distances.append(np.linalg.norm(leading_actions[i] - trailing_actions[i]))

    # What are the cosine similarities between the actions of the two agents?
    similarities = []
    for i in range(len(leading_actions)):
        # check whether the norm of the actions is zero
        if np.linalg.norm(leading_actions[i]) != 0 and np.linalg.norm(trailing_actions[i]) != 0:
            similarities.append(np.dot(leading_actions[i], trailing_actions[i]) / (np.linalg.norm(leading_actions[i]) * np.linalg.norm(trailing_actions[i])))
        # in case the norm of one of the actions is zero, we divide by a very small number to avoid division by zero
        else:
            similarities.append(np.dot(leading_actions[i], trailing_actions[i]) / (0.0000001))

    return distances, similarities

def visualize_actions(actions, agent_names, states, goal, env, colors, save_path):
    """
    Draws the actions of different agents for a given set of states.

    Parameters:
        actions: List of lists of tuples, containing the actions of the agents for each state.
        agent_names: List of strings, containing the names of the agents.
        states: 2D numpy array of states.
        goal: 1D numpy array of goal.
        colors: List of colors, each color corresponds to an agent.
        save_path: String, path to save the plot.
    
    Returns:
        None
    """
    # get base environment image
    env_img = env._get_env_img()
    print("Number of cells in base environment image:", env_img.size)
    

    # draw actions for each state
    for i, state in enumerate(states):
        # draw state
        env_img = env._draw_state(env_img, state, color=[0,255,0], scale=30, radius=2)
        # draw actions of each agent
        for j in range(len(actions)):
            env_img = env._draw_action(env_img, state, actions[j][i], color=colors[j])

    # add legend (showing agent names and colors)
    plt.imshow(env_img)
    plt.axis("off")
    legend_elements = []
    for i in range(len(agent_names)):
        legend_elements.append(patches.Patch(color=np.array(colors[i])/255, label=agent_names[i]))
    plt.legend(handles=legend_elements, loc="lower left")
    plt.savefig(save_path)

if __name__ == "__main__":
    environment_name = "point_Spiral11x11"
    n_tasks = 10
    max_episode_length = 100
    params_path = "manual_checkpoints/two_encoders/point_Spiral11x11/1_million_steps_original"
    seed = 42
    np.random.seed(seed)
    contrastive_eval_mode = False

    evaluating_performances = False
    computing_action_distances = True
    visualizing_actions = True

    # get the action grid, i.e. the actions the greedy agent can select from
    action_grid = scale_action_grid(basic_action_grid, 5)
    strength_grid = get_action_strengths(action_grid)
    # load the environment
    env, obs_dim = get_env(environment_name, return_obs_dim=True, seed=42, return_raw_gym_env=True)
    # sample tasks
    tasks = sample_tasks(env, n_tasks)
    # create agents
    greedy_agent_00 = GreedyAgent(env=env, action_grid=action_grid, params_path=params_path, obs_dim=obs_dim, epsilon=0.0, value_type="contrastive_critic")
    greedy_agent_01 = GreedyAgent(env=env, action_grid=action_grid, params_path=params_path, obs_dim=obs_dim, epsilon=0.01, value_type="contrastive_critic")
    greedy_agent_05 = GreedyAgent(env=env, action_grid=action_grid, params_path=params_path, obs_dim=obs_dim, epsilon=0.05, value_type="contrastive_critic")
    greedy_agent_10 = GreedyAgent(env=env, action_grid=action_grid, params_path=params_path, obs_dim=obs_dim, epsilon=0.1, value_type="contrastive_critic")
    random_agent = RandomAgent(action_grid, obs_dim)
    contrastive_agent = ContrastiveAgent(env=env, action_grid=action_grid, params_path=params_path, obs_dim=obs_dim, eval_mode=contrastive_eval_mode)

    if evaluating_performances:
        print("##########################")
        print("Evaluating Performances")
        print("##########################")
        # record agent performances on tasks
        greedy_00_trajectories = []
        greedy_01_trajectories = []
        greedy_05_trajectories = []
        greedy_10_trajectories = []
        random_trajectories = []
        contrastive_trajectories = []
        for i,task in enumerate(tasks):
            print("Solving Task, ", i+1)
            greedy_00_trajectories.append(greedy_agent_00.solve_task(task)[0])
            greedy_01_trajectories.append(greedy_agent_01.solve_task(task)[0])
            greedy_05_trajectories.append(greedy_agent_05.solve_task(task)[0])
            greedy_10_trajectories.append(greedy_agent_10.solve_task(task)[0])
            random_trajectories.append(random_agent.solve_task(task)[0])
            contrastive_trajectories.append(contrastive_agent.solve_task(task)[0])

        # evaluate agent performances
        #evaluate_performances([greedy_trajectories, random_trajectories, contrastive_trajectories], ["Greedy", "Random", "Contrastive"])
        #evaluate_performances([greedy_trajectories, random_trajectories], ["Greedy", "Random"])
        #evaluate_performances([contrastive_trajectories], ["Contrastive"])
        evaluate_performances([greedy_00_trajectories, greedy_01_trajectories, greedy_05_trajectories, greedy_10_trajectories, random_trajectories, contrastive_trajectories], ["Greedy_00", "Greedy_01", "Greedy_05", "Greedy_10", "Random", "Contrastive"])

    if computing_action_distances:
        print("##########################")
        print("Comparing Actions")
        print("##########################")
        #action_pairs = [(contrastive_agent, greedy_agent_00), (contrastive_agent, random_agent), (contrastive_agent, contrastive_agent)]
        action_pairs = [(contrastive_agent, random_agent), (contrastive_agent, contrastive_agent)]
        for (agent_1, agent_2) in action_pairs:
            print("Comparing Actions of", agent_1, "(leading) and", agent_2, "(trailing) on", n_tasks, "tasks with", max_episode_length, "max episode steps.")
            euclidean_distances = []
            cosine_similarities = []
            for i,task in enumerate(tasks):
                #print("Comparing Actions for Task:", i+1)
                eucl_dists, cos_sims = compare_actions(env=env, task=task, leading_agent=agent_1, trailing_agent=agent_2, obs_dim=obs_dim)
                euclidean_distances.append(eucl_dists)
                cosine_similarities.append(cos_sims)
            # computing mean, max, and min euclidean distances and cosine similarities
            # acros all actions for all tasks
            euclidean_distances = np.array(euclidean_distances)
            cosine_similarities = np.array(cosine_similarities)
            print("###################################")
            print("### Comparison Across All Tasks ###")
            print("###################################")
            print("Mean Euclidean Distance between Actions:", np.mean(euclidean_distances))
            print("Max Euclidean Distance between Actions:", np.max(euclidean_distances))
            print("Min Euclidean Distance between Actions:", np.min(euclidean_distances))
            print()
            print("Mean Cosine Similarity between Actions:", np.mean(cosine_similarities))
            print("Max Cosine Similarity between Actions:", np.max(cosine_similarities))
            print("Min Cosine Similarity between Actions:", np.min(cosine_similarities))
            print()

    if visualizing_actions:
        print("##########################")
        print("Visualizing Actions of Greedy_00 Agent and Contrastive Agent")
        print("##########################")
        # path of image to save
        save_path = "experiment_results/experiment_02/action_comparison.png"
        # compare GreedyAgent and ContrastiveAgent on evenly generated states with goal at the center
        goal = np.array([5.5, 5.5])
        states = generate_env_states(env, environment_name, 1)
        greedy_actions = []
        contrastive_actions = []
        for state in states:
            # construct task
            task = np.concatenate((state, goal))
            # get actions
            greedy_action = greedy_agent_00.select_action(task)
            contrastive_action = contrastive_agent.select_action(task)
            # store actions
            greedy_actions.append(greedy_action)
            contrastive_actions.append(contrastive_action)
        # Visualize actions
        visualize_actions([greedy_actions, contrastive_actions], ["GreedyAgent", "ContrastiveAgent"], states, goal, env, [[255,0,0], [0,0,255]], save_path)


    """
    100 tasks with 100 max episode steps
    Seed 42
    Summary:
    Greedy_00 success rate: 0.59
    Greedy_01 success rate: 0.59
    Greedy_05 success rate: 0.64
    Greedy_10 success rate: 0.63
    Random success rate: 0.13
    Contrastive success rate: 0.66

    Comparing Actions of ContrastiveAgent (leading) and GreedyAgent (trailing) on 10 tasks with 100 max episode steps.
    ###################################
    ### Comparison Across All Tasks ###
    ###################################
    Mean Euclidean Distance between Actions: 0.90950465
    Max Euclidean Distance between Actions: 2.828427
    Min Euclidean Distance between Actions: 0.0

    Mean Cosine Similarity between Actions: 0.39543292
    Max Cosine Similarity between Actions: 1.0000001
    Min Cosine Similarity between Actions: -1.0000001
    """