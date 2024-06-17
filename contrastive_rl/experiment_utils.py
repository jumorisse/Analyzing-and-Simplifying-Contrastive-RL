'''
Script containing functions used throughout my thesis experiments.
Also contains functions that automatically run the whole experiment pipeline or single experiments.

The experiments are implemented in the experiment_XX.ipynb notebooks.
'''
import json
import numpy as np
import torch
from torch import nn
from contrastive import utils as contrastive_utils
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from point_env import WALLS, PointEnv, PointImage
import pandas as pd

def inspect_params(params_path):
    # load parameter dict
    with open(params_path, "r") as f:
        # read raw string and replace single quotes with double quotes
        param_string = f.read().replace("'", "\"")
        params_dict = json.loads(param_string)
    
    # inspect parameter dict
    # print keys
    param_keys = params_dict.keys()
    # print keys nested dicts
    for key in param_keys:
        if isinstance(params_dict[key], dict):
            print("Layer:", key)
            for sub_key in params_dict[key].keys():
                print("2nd Level:",sub_key, " with shape:", np.array(params_dict[key][sub_key]).shape)
        else:
            print("Layer:", key)
            print("Value shape:", np.array(params_dict[key]).shape)

def inspect_network(network):
    """
    Takes in a Pytorch neural network and prints the architecture of the network.
    Additionally, prints the three firt weights and biases.
    """
    # print architecture
    print(network)

def create_encoder(params, activation=nn.ReLU()):
    """
    Function that receives a dictionary of parameters and creates a PyTorch feed-forward neural network from it.
    The dictionary has the following format:
    {
        "layer_0": {
            "b": np.ndarray,
            "w": np.ndarray
        },
        "layer_1": {
            "b": np.ndarray,
            "w": np.ndarray
        },
        ...
    }
    The function returns a PyTorch neural network following the architecture specified in the dictionary and
    loading the weights and biases from the dictionary.

    Args:
        params: Dictionary containing the parameters of the network.
        activation: Activation function to use in the network on all layers except the last one.

    Returns:
        nn.Sequential: PyTorch feed-forward neural network.
    """
    # create list of layers
    layers = []
    for key in params.keys():
        # extract layer number from key
        layer_nr = int(key.split("_")[1])
        # extract weights and biases
        w = torch.tensor(params[key]["w"])
        b = torch.tensor(params[key]["b"])
        # create layer
        layers.append(nn.Linear(w.shape[0], w.shape[1]))
        # load weights and biases
        layers[-1].weight.data = torch.transpose(w, 0, 1) # weight matrix needs to be transposed (PyTorch uses outxin format)
        layers[-1].bias.data = b
        # add activation function
        if layer_nr < len(params.keys()) - 1:
            layers.append(activation)

    # create encoder
    encoder = nn.Sequential(*layers)

    return encoder

def load_encoders(env, critic_path=None):
    if critic_path is None:
        critic_path = f"manual_checkpoints/{env}/critic_params.txt"

    # load parameter dict
    with open(critic_path, "r") as f:
        # read raw string and replace single quotes with double quotes
        param_string = f.read().replace("'", "\"")
        params_dict = json.loads(param_string)
    
    # split parameters into state-action and goal encoder parameters
    sa_encoder_params = {}
    g_encoder_params = {}
    for key in params_dict.keys():
        layer = key.split("/")[2]
        if "sa_encoder" in key:
            sa_encoder_params[layer] = params_dict[key]
        elif "g_encoder" in key:
            g_encoder_params[layer] = params_dict[key]
    
    # create encoder from parameters
    sa_encoder = create_encoder(sa_encoder_params)
    g_encoder = create_encoder(g_encoder_params)

    return sa_encoder, g_encoder


class MultiHeadNet(nn.Module):
    def __init__(self, linear_layers, output_layers):
        super(MultiHeadNet, self).__init__()
        self.linear_layers = nn.ModuleList(linear_layers)
        self.output_layers = nn.ModuleList(output_layers)

    def forward(self, x):
        for layer in self.linear_layers:
            x = layer(x)
        outputs = []
        for layer in self.output_layers:
            outputs.append(layer(x))
        return outputs

def create_policy_net(params):
    """
    Creates policy neural network which takes in a state and outputs two tensors.
    The first tensor contains the means of Gaussian distributions, one for each action dimension.
    The second tensor contains the standard deviations of the Gaussian distributions, one for each action dimension.

    Args:
        params: Dictionary containing the parameters of the network's layers in this order [output1, output2, first linear, second linear].
                The network is connected as first linear -> activation -> second linear -> activation -> output1, output2.
    
    Returns:
        nn.Sequential: PyTorch feed-forward neural network.
    """
    # split params dict in params for linear layers and the output layers
    linear_params = {}
    output_params = {}
    for i,key in enumerate(params.keys()):
        if "mlp/~/linear" in key:
            linear_params["linear"+key.split("_")[1]] = params[key]
        elif "Normal/~/linear" in key:
            output_params["output_"+str(i)] = params[key]
    
    # create list of layers
    linear_layers = []
    for key in linear_params.keys():
        # extract weights and biases
        w = torch.tensor(linear_params[key]["w"])
        b = torch.tensor(linear_params[key]["b"])
        # create layer
        linear_layers.append(nn.Linear(w.shape[0], w.shape[1]))
        # load weights and biases
        linear_layers[-1].weight.data = torch.transpose(w, 0, 1)
        linear_layers[-1].bias.data = b
        # add activation function
        linear_layers.append(nn.ReLU())
    
    # create output layers
    output_layers = []
    for key in output_params.keys():
        # extract weights and biases
        w = torch.tensor(output_params[key]["w"])
        b = torch.tensor(output_params[key]["b"])
        # create layer
        output_layers.append(nn.Linear(w.shape[0], w.shape[1]))
        # load weights and biases
        output_layers[-1].weight.data = torch.transpose(w, 0, 1)
        output_layers[-1].bias.data = b
    
    # create multi-head actor
    policy_net = MultiHeadNet(linear_layers, output_layers)

    return policy_net


def load_policy_net(policy_path):
    # load parameter dict
    with open(policy_path, "r") as f:
        # read raw string and replace single quotes with double quotes
        param_string = f.read().replace("'", "\"")
        params_dict = json.loads(param_string)
    
    # create actor from parameters
    policy_net = create_policy_net(params_dict)

    return policy_net

def load_trained_networks(environment):
    checkpoints_path = f"manual_checkpoints/{environment}"

    sa_encoder, g_encoder = load_encoders(f"{checkpoints_path}/critic_params.txt")
    policy_net = load_policy_net(f"{checkpoints_path}/policy_params.txt")

    # networks not trained, thus return in eval mode
    sa_encoder.eval()
    g_encoder.eval()
    policy_net.eval()

    return sa_encoder, g_encoder, policy_net

def get_env(environment_name, return_obs_dim=False, seed=42, return_raw_gym_env=False):
    env, obs_dim = contrastive_utils.make_environment(
        env_name=environment_name,
        start_index=0,
        end_index=-1,
        seed=seed,
        return_raw_gym_env=return_raw_gym_env)
    if return_obs_dim:
        return env, obs_dim
    else:
        return env

def sample_env_states(env, n):
    """
    Samples n states of the environment.

    Args:
        env: Gym environment.
        n: Number of states to sample.
    """
    states = []
    for _ in range(n):
        env.reset()
        states.append(np.array(env.state))
    return states

def sort_states(env, env_name, states, ascending=False):
    """
    Sorts states based on their steps to the center.

    Args:
        env: Environment object
        env_name: Environment name
        states: List of states to sort.
        ascending: Boolean indicating whether to sort in ascending order.
    
    Returns:
        List of sorted states.
    """
    # get steps to center for each state
    steps = get_steps_to_center(env, states, env_name)
    # sort states based on steps
    sorted_states = [state for _, state in sorted(zip(steps, states), key=lambda x: x[0], reverse=not ascending)]
    return sorted_states

def generate_env_states(env, env_name, factor):
    """
    Generates environment states for a gridworld environment.
    If factor is 1, center of every non-wall grid cell is returned.
    If factor is higher than 1, states between grid cells are also returned.
    Factor 2 means that grid cell centers and the midpoints between grid cell centers are returned.
    Factor 3 means that grid cell centers, midpoints between grid cell centers, and midpoints between midpoints are returned.
    And so on...

    Args:
        env: Gridworld environment.
        factor: Factor to determine how many states are generated.
    
    Returns:
        List of states in the environment.
    """
    # get gridworld layour
    layout = WALLS[env_name.split("_")[1]]
    # get gridworld size
    size = layout.shape
    # get non-wall grid cell centers
    states = []
    for i in range(size[0]):
        for j in range(size[1]):
            if layout[i,j] == 0:
                states.append(np.array([i+0.5,j+0.5]))
    # sort states, because midpoint generation assumes states are in order from exit to center or center to exit
    states = sort_states(env, env_name, states)

    for i in range(factor-1):
        new_states = []
        for j in range(len(states)-1):
            new_states.append(states[j])
            new_states.append((states[j]+states[j+1])/2)
        new_states.append(states[-1])
        states = new_states
    
    return states

def get_sa_encodings(sa_encoder, states, actions):
    """"
    Uses a state-action encoder to encode states and actions.

    Args:
        sa_encoder: PyTorch neural network (simple feed-forward MLP) that encodes states and actions.
        states: List of states to encode, each state is a numpy array of state dimensionality.
        default_action: Can be list of actions (corresponding to list of states) or single action (used for all states).
    
    Returns:
        List of state-action encodings containing numpy arrays.
    """
    # convert list of np.array states to list of torch tensors
    states = [torch.tensor(state, dtype=torch.float32) for state in states]
    # convert list of np.array actions to list of torch tensors
    actions = [torch.tensor(action, dtype=torch.float32) for action in actions]
    # concatenate states and actions
    if len(actions) == len(states):
        # if number of dimensions is the same for states and actions, i.e. each state has one corresponding action
        if len(actions[0].shape) == len(states[0].shape):
            sa_pairs = [torch.cat([states[i], actions[i]]) for i in range(len(states))]
        else:
            sa_pairs = []
            for i in range(len(states)):
                for j in range(len(actions[0])):
                    sa_pairs.append(torch.cat([states[i], actions[i][j]]))
                    
    elif len(actions) == 1:
        sa_pairs = [torch.cat([state, actions[0]]) for state in states]
    
    # encode state-action pairs
    #sa_encodings = [sa_encoder(sa_pair) for sa_pair in sa_pairs]
    sa_encodings = []
    for sa_pair in sa_pairs:
        # get encoding and drop batch dimension
        encoding = sa_encoder(sa_pair).squeeze()
        sa_encodings.append(encoding)

    # return encodings as list of numpy arrays
    return [encoding.detach().numpy() for encoding in sa_encodings]

def get_g_encodings(g_encoder, goals):
    """"
    Uses a goal encoder to encode goals.

    Args:
        g_encoder: PyTorch neural network (simple feed-forward MLP) that encodes goals.
        goals: List of goals to encode, each goal is a numpy array of goal dimensionality.
    
    Returns:
        List of goal encodings containing numpy arrays.
    """
    # convert list of np.array goals to list of torch tensors
    goals = [torch.tensor(goal, dtype=torch.float32) for goal in goals]
    
    # encode goals
    g_encodings = [g_encoder(goal) for goal in goals]

    # return encodings as list of numpy arrays
    return [encoding.detach().numpy() for encoding in g_encodings]

def reduce_dim(encodings, method="TSNE", target_dim=2, perplexity=20, distance_metric="euclidean"):
    """
    Reduces the dimensionality of the encodings using the specified method.

    Args:
        encodings: List of encodings to reduce.
        method: Method to use for dimensionality reduction.
        target_dim: Target dimensionality of the reduced encodings.
        perplexity: Perplexity parameter for TSNE.
        distance_metric: Distance metric for TSNE.

    Returns:
        List of encodings with reduced dimensionality.
    """
    if method == "TSNE":
        tsne = TSNE(n_components=target_dim, perplexity=perplexity, metric=distance_metric)
        encodings_2d = tsne.fit_transform(np.array(encodings))
    elif method == "PCA":
        pca = PCA(n_components=target_dim)
        encodings_2d = pca.fit_transform(np.array(encodings))
    else:
        raise ValueError("Invalid method specified. Use 'TSNE' or 'PCA'.")

    return encodings_2d

def plot_encodings(encodings, path, state_colors=None, colorbar=None, colorbar_ticks=None):
    """
    Plots the encodings using the specified method.

    Args:
        encodings: List of encodings to plot, already mapped to 2d.
        path: Path to save the plot.
    
    Returns:
        None
    """
    if state_colors is None:
        plt.scatter(encodings[:, 0], encodings[:, 1])
        plt.savefig(path)
        plt.close()
    else:
        plt.scatter(encodings[:, 0], encodings[:, 1], c=np.array(state_colors)/255)
        if colorbar is not None and colorbar_ticks is not None:
            plt.colorbar(colorbar, label="Steps to Center").set_ticklabels(list(map(str, list(colorbar_ticks))))
        plt.savefig(path)
        plt.close()

def get_steps_to_center(env, states, env_name="point_Spiral11x11"):
    if env_name=="point_Spiral11x11":
        distances = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                              [1, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
                              [1, 37, 1, 1, 1, 1, 1, 1, 1, 1, 48],
                              [1, 36, 1, 12, 13, 14, 15, 16, 17, 1, 49],
                              [1, 35, 1, 11, 1, 1, 1, 1, 18, 1, 50],
                              [1, 34, 1, 10, 1, 2, 3, 1, 19, 1, 51],
                              [1, 33, 1, 9, 1, 1, 4, 1, 20, 1, 52],
                              [1, 32, 1, 8, 7, 6, 5, 1, 21, 1, 53],
                              [1, 31, 1, 1, 1, 1, 1, 1, 22, 1, 54],
                              [1, 30, 29, 28, 27, 26, 25, 24, 23, 1, 55],
                              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 56]])
    else:
        raise ValueError("Invalid environment name. Use 'point_Spiral11x11'.")
    print("Distances shape:", distances.shape)

    discrete_states = [env._discretize_state(state) for state in states]
    steps = []
    for state in discrete_states:
        try:
            # -2 because walls have value 1 and step distances start counting from 2
            steps.append(distances[state[0]][state[1]]-2)
        except IndexError:
            print("Index out of bounds")
            print(state)
            raise IndexError
    return steps

def visualize_states(env, states, color_meaning="steps", save_path=None, state_colors=None, steps=None):
    """
    Visualizes the given states in the environment.

    Args:
      states (np.ndarray): The states to visualize.
    
    Returns:
        None (if no save_path is given)
        or
        colors (list): List of rgb colors corresponding to the states.
        colorbar: Colorbar object for the color mapping.
        colorbar_ticks: Ticks for the colorbar.
    """
    # default palette not inversed
    inverse_palette = False

    # get individual images either with state colored based on steps to center or not
    if color_meaning == "steps":
        if steps is None:
            state_values = get_steps_to_center(env, states)
        else:
            state_values = steps
        # get color palette from min to max steps
        min_value = min(state_values)
        max_value = max(state_values)
        color_palette = plt.cm.get_cmap("viridis", max_value - min_value + 1)
        state_colors = [np.array(color_palette(step - min_value))[:3]*255 for step in state_values]
    
    env_img = env._get_env_img()
    
    radius = 3 # size of the state squares drawn into the environment
    if color_meaning == "steps":
        for state, color in zip(states, state_colors):
            env_img = env._draw_state(env_img, state, color=color, scale=30, radius=radius)
    elif color_meaning == "critic":
        state_values = state_colors
        # state_colors are list of critic values, need to be mapped to colors
        # scale critic values to [0,1]
        min_value = min(state_colors)
        max_value = max(state_colors)
        print("Min value:", min_value)
        print("Max value:", max_value)
        scaled_values = [(value - min_value) / (max_value - min_value) for value in state_colors]
        color_palette =  plt.cm.get_cmap("viridis")
        # invert color palette (e.g. so that steps-baseline and value-visualization match in regards to where colors should ideally lie)
        inverse_palette = True
        if inverse_palette:
            color_palette = plt.cm.get_cmap("viridis_r")
        state_colors = [np.array(color_palette(value))[:3]*255 for value in scaled_values]
        for state, color in zip(states, state_colors):
            env_img = env._draw_state(env_img, state, color=color, scale=30, radius=radius)
    elif color_meaning == "ids":
        color_meaning = False
    else:
        if state_colors is None:
            for state in states:
                env_img = env._draw_state(env_img, state, color=[255,0,0], scale=30, radius=radius) # set state color to red
        else:
            for state, color in zip(states, state_colors):
                env_img = env._draw_state(env_img, state, color=color, scale=30, radius=radius)

    plt.imshow(env_img)

    if color_meaning == "steps" or color_meaning == "critic":
        # add colorbar
        colorbar = plt.cm.ScalarMappable(cmap=color_palette).set_array(state_values)
        if not inverse_palette:
            # compute colorbar ticks, we want to have 6 ticks (max as upper bound, min as lower bound, and 4 in between)
            value_range = max_value - min_value
            colorbar_ticks = [min_value + i * value_range / 5 for i in range(6)]
        elif inverse_palette:
            # we inversed the color palette, so we need to invert the colorbar ticks as well
            colorbar_ticks = range(int(max_value), int(min_value), -10)
        # round colorbar ticks to decimal places
        colorbar_ticks = [round(tick, 2) for tick in colorbar_ticks]
        if color_meaning == "steps":
            if steps is None:
                plt.colorbar(colorbar, label="Steps to Center").set_ticklabels(list(map(str, list(colorbar_ticks))))
            else:
                plt.colorbar(colorbar, label="Steps to Goal").set_ticklabels(list(map(str, list(colorbar_ticks))))
        elif color_meaning == "critic":
            plt.colorbar(colorbar, label="Critic Value").set_ticklabels(list(map(str, list(colorbar_ticks))))
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()

    if color_meaning == "steps" or color_meaning == "critic":
        return state_colors, colorbar, colorbar_ticks

def produce_plot(log_paths, labels, x_var, y_var, saving_path, max_steps=None, figsize=(10,6)):
        # define colors, one for each curve
        colors = []
        for i in range(len(log_paths)):
            colors.append(plt.cm.get_cmap("tab10")(i))
        # define lookup table for axis labels
        axis_labels = {
            "actor_steps": "Actor Steps",
            "evaluator_steps": "Evaluator Steps",
            "learner_steps": "Learner Steps",
            "success_1000": "Success Rate",
            "actor_loss": "Actor Loss",
            "critic_loss": "Critic Loss"
        }
        # initialize plot
        plt.figure()
        # set figsize
        plt.figure(figsize=figsize)
        for i,(log_path, label) in enumerate(zip(log_paths, labels)):
            # load logs.csv from path, if path is list of paths, load all logs
            if isinstance(log_path, list):
                logs = []
                for path in log_path:
                    logs.append(pd.read_csv(path))
            else:
                logs = pd.read_csv(log_path)
            
            # extract X-axis, if logs is list of logs: average over all logs
            if isinstance(log_path, list):
                x_values = [log[x_var] for log in logs]
                # trim all logs to the shortest log
                min_length = min([len(x) for x in x_values])
                x_values = [x[:min_length] for x in x_values]
                mean_x = np.mean(x_values, axis=0)
            else:
                x = logs[x_var]
            
            # extract Y-axis, if logs is list of logs: average over all logs
            if isinstance(log_path, list):
                y_values = [log[y_var] for log in logs]
                # trim all logs to the shortest log
                min_length = min([len(y) for y in y_values])
                y_values = [y[:min_length] for y in y_values]
                mean_y = np.mean(y_values, axis=0)
                std_y = np.std(y_values, axis=0)
            else:
                y = logs[y_var]
            
            # if a max_steps is specified, only plot up to this step
            if max_steps is not None:
                if isinstance(log_path, list):
                    # find first index where x value is equal or greater than max_steps
                    idx = np.argmax(mean_x >= max_steps)
                    mean_x = mean_x[:idx]
                    mean_y = mean_y[:idx]
                    std_y = std_y[:idx]
                else:
                    # find first index where x value is equal or greater than max_steps
                    idx = np.argmax(x >= max_steps)
                    x = x[:idx]
                    y = y[:idx]

            # plot the curve for this log
            if isinstance(log_path, list):
                # if we plot an average x value, we also need to plot the standard deviation
                plt.plot(mean_x, mean_y, label=label, color=colors[i])
                plt.fill_between(mean_x, mean_y-std_y, mean_y+std_y, alpha=0.3, color=colors[i])
            else:
                plt.plot(x, y, label=label, color=colors[i])
            
        # set labels, retrieve axis labels from lookup table (if it doesn't exist in there, use the variable name)
        plt.xlabel(axis_labels.get(x_var, x_var), fontsize=12)
        plt.ylabel(axis_labels.get(y_var, y_var), fontsize=12)

        # add legend
        plt.legend()

        # save plot
        plt.savefig(saving_path+f"_{y_var}.png")
        plt.close()
    
def plot_training_curves(checkpoint_paths, curve_labels, saving_path, vars, max_steps=[None,None,None], figsize=(10,6)):
    """
    Produces and saves training curves for the variables specified in the curves list.

    Args:
        checkpoint_path: Path to the checkpoint files containing the logs.
        curve_labels: The labels for the different checkpoints, describing the training setting.
        saving_path: Path to save the training curves plot.
        vars: List of variables to plot.
    
    Returns:
        None
    """
    # define different paths for actor, evaluator, and learner logs
    def get_separate_paths(checkpoint_path):
        actor_log_path = checkpoint_path+"/logs/actor/logs.csv"
        evaluator_log_path = checkpoint_path+"/logs/evaluator/logs.csv"
        learner_log_path = checkpoint_path+"/logs/learner/logs.csv"
        return actor_log_path, evaluator_log_path, learner_log_path

    actor_log_paths = []
    evaluator_log_paths = []
    learner_log_paths = []

    for checkpoint_path in checkpoint_paths:
        # check if checkpoint path is list of paths
        if isinstance(checkpoint_path, list):
            actor_log_list = []
            evaluator_log_list = []
            learner_log_list = []
            for checkpoint in checkpoint_path:
                actor_log_path, evaluator_log_path, learner_log_path = get_separate_paths(checkpoint)
                actor_log_list.append(actor_log_path)
                evaluator_log_list.append(evaluator_log_path)
                learner_log_list.append(learner_log_path)
            actor_log_paths.append(actor_log_list)
            evaluator_log_paths.append(evaluator_log_list)
            learner_log_paths.append(learner_log_list)
        else:
            actor_log_path, evaluator_log_path, learner_log_path = get_separate_paths(checkpoint_path)
            actor_log_paths.append(actor_log_path)
            evaluator_log_paths.append(evaluator_log_path)
            learner_log_paths.append(learner_log_path)


    # define which logs contain which variables
    actor_vars = ["success_1000"]
    evaluator_vars = ["success_1000"]
    learner_vars = ["actor_loss", "critic_loss"]

    # produce plots of each variable for each log that contains it
    for var in vars:
        if var in actor_vars:
            # create figure
            produce_plot(actor_log_paths, curve_labels, "actor_steps", var, saving_path+"_actor", max_steps[0], figsize=figsize)
        if var in evaluator_vars:
            produce_plot(evaluator_log_paths, curve_labels, "evaluator_steps", var, saving_path+"_evaluator", max_steps[1], figsize=figsize)
        if var in learner_vars:
            produce_plot(learner_log_paths, curve_labels, "learner_steps", var, saving_path+"_learner", max_steps[2], figsize=figsize)

def combine_repr(repr1, repr2):
    """
    Function that computes the similarity between SA- and G-representations.
    Similarity is computed as the inner/dot product between the two representations.

    Args:
        repr1: (State-action) representation as a numpy array.
        repr2: (Goal) representation as a numpy array.
    
    Returns:
        numpy.ndarray: Matrix containing the similarity between the two representations.
    """
    # add batch dimension to both representations
    repr1 = repr1[np.newaxis, :]
    repr2 = repr2[np.newaxis, :]
    inner_product = np.einsum('ik,jk->ij', repr1, repr2)
    # remove batch dimension and reduce to scalar
    return inner_product.squeeze()

def critic_dist(array1, array2):
    """
    Calculates the distance between two representations.
    Uses the same distance computation as in the critic function.

    Args:
        array1: First representation as a numpy array.
        array2: Second representation as a numpy array.

    Returns:
        numpy.ndarray: Matrix containing the distance between the two representations.
    """
    return combine_repr(array1, array2)

def get_action_colors(actions):
    """
    Maps actions to rgb colors based on their strength (strength is the vector length of the action).
    Mapping is done from light gray (low strength) to black (high strength).

    Args:
        actions: Array of actions to map to colors, shape (nr_actions, action_dim).
    
    Returns:
        colors: List of rgb colors corresponding to the actions.
        colorbar: Colorbar object for the color mapping.
        colorbar_ticks: Ticks for the colorbar.
    """
    # calculate action strengths
    action_strengths = np.linalg.norm(actions, axis=1)
    # get min and max strength
    #min_strength = np.min(action_strengths)
    #max_strength = np.max(action_strengths)
    min_strength = 0
    max_strength = np.sqrt(2)
    # create color palette from light gray to black
    color_palette = plt.cm.get_cmap("Greys", 256)
    # map strengths to colors
    colors = [np.array(color_palette((strength - min_strength) / (max_strength - min_strength))[:3])*255 for strength in action_strengths]
    # create colorbar
    colorbar = plt.cm.ScalarMappable(cmap=color_palette).set_array(action_strengths)
    colorbar_ticks = [min_strength, max_strength]
    return colors, colorbar, colorbar_ticks

def sample_tasks(env, n_tasks, seed=42):
    """
    Samples n_tasks tasks from the environment.

    Args:
        env: Gym environment.
        n_tasks: Number of tasks to sample.
    
    Returns:
        List of tasks.
    """
    tasks = []
    for _ in range(n_tasks):
        env.reset()
        tasks.append(env._get_obs())
    return tasks


if __name__ == "__main__":
    params_path = "manual_checkpoints/two_encoders/point_Spiral11x11/1_million_steps_original"
    actor_path = params_path + "/policy_params.txt"

    inspect_params(actor_path)

    policy_net = load_policy_net(actor_path)

    inspect_network(policy_net)