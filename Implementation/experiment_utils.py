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
import time

def sigmoid(x):
    """
    Computes the sigmoid function for the given input x.

    Args:
        x: Input to the sigmoid function, can be scalar or numpy array.
    """
    return 1 / (1 + np.exp(-x))

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

def plot_encodings(encodings, path, state_colors=None, colorbar=None, colorbar_ticks=None, figsize=None):
    """
    Plots the encodings using the specified method.

    Args:
        encodings: List of encodings to plot, already mapped to 2d.
        path: Path to save the plot.
    
    Returns:
        None
    """
    # check colorbar_ticks: if its an embedded list, extract first (step colorbar ticks) and second (gray scale/action colorbar ticks) list
    if isinstance(colorbar_ticks[0], list):
        two_colorbars = True
        step_colorbar_ticks = colorbar_ticks[0]
        action_colorbar_ticks = colorbar_ticks[1]
    else:
        two_colorbars = False
        step_colorbar_ticks = colorbar_ticks

    if figsize is None:
        figsize = (8,6)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    if state_colors is None:
        plt.scatter(encodings[:, 0], encodings[:, 1], c="red")
    else:
        plt.scatter(encodings[:, 0], encodings[:, 1], c=np.array(state_colors)/255)
        if colorbar_ticks is not None:
            colorbar = plt.cm.ScalarMappable(cmap="viridis").set_array(state_colors)
            plt.colorbar(colorbar, label="Steps to Center").set_ticklabels(list(map(str, list(step_colorbar_ticks))))

            if two_colorbars:
                colorbar = plt.cm.ScalarMappable(cmap="Greys")
                plt.colorbar(colorbar, label="Sampled Actions' Strength", ax=[ax], location="left").set_ticklabels(list(map(str, list(action_colorbar_ticks)))[::-1])

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

def visualize_states(env, states, color_meaning="steps", save_path=None, state_colors=None, steps=None, radius=3):
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
        for i, state in enumerate(states):
            env_img = env._draw_state(env_img, state, color=state_colors[i], scale=30, radius=radius)

    else:
        if state_colors is None:
            for state in states:
                env_img = env._draw_state(env_img, state, color=[255,0,0], scale=30, radius=radius) # set state color to red
        else:
            for state, color in zip(states, state_colors):
                env_img = env._draw_state(env_img, state, color=color, scale=30, radius=radius)

    plt.imshow(env_img)

    # turn axis ticks, only keep the lines
    plt.xticks([])
    plt.yticks([])

    if color_meaning == "steps" or color_meaning == "critic":
        # add colorbar
        colorbar = plt.cm.ScalarMappable(cmap=color_palette).set_array(state_values)
        if not inverse_palette:
            # compute colorbar ticks, we want to have 6 ticks (max as upper bound, min as lower bound, and 4 in between)
            value_range = max_value - min_value
            colorbar_ticks = [min_value + i * value_range / 5 for i in range(6)]
        elif inverse_palette:
            value_range = max_value - min_value
            colorbar_ticks = [min_value + i * value_range / 5 for i in range(6)]
            print("Colorbar ticks:", colorbar_ticks)
            # we inversed the color palette, so we need to invert the colorbar ticks as well
            colorbar_ticks = colorbar_ticks[::-1]
        # round colorbar ticks to decimal places
        colorbar_ticks = [int(tick) for tick in colorbar_ticks]
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
        plt.legend(loc="lower right")

        # add grid
        plt.grid(alpha=0.5)

        # save plot
        plt.savefig(saving_path+f"_{y_var}.png")
        plt.close()
    
def plot_training_curves(checkpoint_paths, curve_labels, saving_path, vars, max_steps=[None,None,None], figsize=(10,6)):
    """
    Produces and saves training curves for the variables specified in the curves list.

    Args:
        checkpoint_path: List of paths to the checkpoint files containing the logs, one for each seed.
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
    Uses the same distance computation as in the critic function: sigmoid(inner product).

    Args:
        array1: First representation as a numpy array.
        array2: Second representation as a numpy array.

    Returns:
        numpy.ndarray: Matrix containing the distance between the two representations.
    """
    return sigmoid(combine_repr(array1, array2))

def get_action_colors(actions, state_color=None, color=True):
    """
    Maps actions to rgb colors based on their strength (strength is the vector length of the action).
    Mapping can be done in two ways:
        - color: True: Actions are colored based on the state they are paired with. The color fades out the stronger the action.
        - color: False: Actions are grayscale based on their strength (white = low strength, black = high strength).

    Args:
        actions: Array of actions to map to colors, shape (nr_actions_sampled_per_state, action_dim).
        state_color: Color of the state for which the actions were sampled.
        color: Boolean indicating whether to color the actions based on the state they are paired with or whether to grayscale them based on their strengths.
    
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

    # if color is False, actions are grayscale depending on their strength (white = low strength, black = high strength)
    if not color:
        # create color palette from light gray to black
        color_palette = plt.cm.get_cmap("Greys", 256)
        # map strengths to colors
        colors = [np.array(color_palette((strength - min_strength) / (max_strength - min_strength))[:3])*255 for strength in action_strengths]
        # create colorbar
        colorbar = plt.cm.ScalarMappable(cmap=color_palette).set_array(action_strengths)
        colorbar_ticks = [min_strength, max_strength]

    # if color is True, actions share the state color of the state they are paired with
    # however, they get more opaque with increasing strength (low strength = no fade, high strength = strong fade)
    elif color:
        # the action colors are the state color but with increasing opacity the stronger the action
        colors = []
        for strength in action_strengths:
            # calculate opacity based on strength
            opacity = (strength - min_strength) / (max_strength - min_strength)
            # fade out the state color
            color = state_color * (1-opacity) + np.array([255,255,255]) * opacity
            colors.append(color)

        # set colorbar and ticks to None, because we don't need them
        colorbar = None
        colorbar_ticks = None

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

class ContrastiveCritic:
    def __init__(self, env, params_path, obs_dim):
        self.env = env
        self.params_path = params_path
        self.sa_encoder, self.g_encoder = self.load_encoder_nets()
        self.obs_dim = obs_dim
    
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
        state = obs[:self.obs_dim]
        goal = obs[self.obs_dim:]

        # encode state-action pair
        sa_encoding = self.encode_sa(state, action)
        # encode goal
        g_encoding = self.encode_g(goal)

        # calculate critic value, i.e. inner product of the two encodings
        inner_product = np.dot(sa_encoding, g_encoding)

        # apply sigmoid activation
        # inner_product = sigmoid(inner_product)

        return inner_product


class GreedyAgent:
    def __init__(self, env, action_grid, params_path, obs_dim, epsilon=0.05, value_type="contrastive_critic"):
        self.env = env
        self.params_path = params_path
        self.action_grid = action_grid
        self.value_type = value_type
        self.obs_dim = obs_dim
        self.epsilon = epsilon
        self.evaluator = self.load_evaluator()

    def load_evaluator(self):
        if self.value_type == "contrastive_critic":
            return ContrastiveCritic(self.env, self.params_path, self.obs_dim)

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

    def solve_task(self, task, max_episode_length=100):
        """
        Gets a task (initial state and goal) and solves it using the greedy agent.

        Parameters:
            task: 2D numpy array of shape (1, 2*obs_dim)
        
        Returns:
            A tuple containing the trajectory and a boolean indicating whether the goal was reached.
        """
        # set task to be environment's current observation
        self.env._set_obs(task)

        # split task into intial state and goal
        start_state = task[:self.obs_dim]
        goal = task[self.obs_dim:]

        # initialize trajectory
        trajectory = []

        # reached goal flag
        reached_goal = []

        current_state = start_state
        # iterate through the task
        for i in range(max_episode_length):
            # select action
            action = self.select_action(np.concatenate((current_state, goal)))
            # apply action
            next_obs, reward, done, _ = self.env.step(action)
            next_state = next_obs[:self.obs_dim]
            # check if goal is reached
            if reward == 1:
                reached_goal.append(i)
            # update start state
            current_state = next_state
            # append to trajectory
            trajectory.append((current_state, action, reward, done))
            # check if task is done
            if done:
                break
        
        return trajectory, reached_goal


class RandomAgent:
    def __init__(self, action_grid, obs_dim, env):
        self.action_grid = action_grid
        self.obs_dim = obs_dim
        self.env = env

    def select_action(self, obs=None):
        """
        Selects an action randomly from the action grid.

        Returns:
            A tuple representing the random action.
        """
        random_row_index = np.random.randint(self.action_grid.shape[0])
        random_column_index = np.random.randint(self.action_grid.shape[1])
        return self.action_grid[random_row_index, random_column_index]
    
    def solve_task(self, task, max_episode_length=100):
        # set task to be environment's current observation
        self.env._set_obs(task)

        # split task into intial state and goal
        start_state = task[:self.obs_dim]
        goal = task[self.obs_dim:]

        # initialize trajectory
        trajectory = []

        # reached goal flag
        reached_goal = []

        current_state = start_state
        # iterate through the task
        for i in range(max_episode_length):
            # select action
            action = self.select_action()
            # apply action
            next_obs, reward, done, _ = self.env.step(action)
            next_state = next_obs[:self.obs_dim]
            # check if goal is reached
            if reward == 1:
                reached_goal.append(i)
            # update current state
            current_state = next_state
            # append to trajectory
            trajectory.append((current_state, action, reward, done))
            # check if task is done
            if done:
                break

        return trajectory, reached_goal


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
    def __init__(self, env, params_path, obs_dim, eval_mode=False):
        self.env = env
        self.obs_dim = obs_dim
        self.eval_mode = eval_mode
        self.evaluator = ContrastiveCritic(env, params_path, obs_dim)
        self.actor = self.get_actor(params_path)
    
    def get_actor(self, params_path):
        return ContrastiveActor(params_path, self.obs_dim, self.eval_mode)

    def select_action(self, obs):
        return self.actor.select_action(obs)

    def solve_task(self, task, max_episode_length=100):
        # set task to be environment's current observation
        self.env._set_obs(task)

        # split task into intial state and goal
        start_state = task[:self.obs_dim]
        goal = task[self.obs_dim:]

        # initialize trajectory
        trajectory = []

        # reached goal flag
        reached_goal = []

        # iterate through the task
        current_state = start_state
        for i in range(max_episode_length):
            # select action
            action = self.actor.select_action(np.concatenate((current_state, goal)))
            # apply action
            next_obs, reward, done, _ = self.env.step(action)
            next_state = next_obs[:self.obs_dim]
            # check if goal is reached
            if reward == 1:
                reached_goal.append(i)
            # update start state
            current_state = next_state
            # append to trajectory
            trajectory.append((current_state, action, reward, done))
            # check if task is done
            if done:
                break

        return trajectory, reached_goal


def evaluate_performances(trajectories, steps_in_goal, agent_names, nr_tasks = 100, only_final_reward_matters = False):
    """
    Evaluates the performances of the agents on the tasks.
    Prints the success rate of each agent on each task and the average success rate of each agent.
    Trajectories can be:
    - list of lists of tuples; first list contains one entry per agent, second list contains one entry per task
    - list of lists of lists of tuples, first list contains one entry per agent, second list one entry per seed, third one per task

    Parameters:
        trajectories: List of lists of tuples, containing the trajectories of the agents on the tasks.
        steps_in_goal: List of lists of timesteps at which the agent is within the goal distance.
        agent_names: List of strings, containing the names of the agents.
    """
    # TODO: remove the whole steps in goal thing, doesn't really seem to work and not used anyway
    success_rates = []
    first_steps_in_goal = []

    # check that trajectories don't contain trajectories of multiple seeds per agent, recognize this by checking at which depth the tuples lie
    if type(trajectories[0][0][0]) == tuple:
        # print the performances of each agent for each task (store whether it was successful or not)
        for i in range(len(trajectories)):
            print("Agent:", agent_names[i])
            agent_success_rates = []
            agent_goal_steps = []
            for j in range(len(trajectories[i])):
                all_rewards = [traj[2] for traj in trajectories[i][j]]
                any_reward = np.sum(all_rewards) >= 1
                final_reward = trajectories[i][j][-1][2]
                if len(steps_in_goal[i][j]) > 0 and final_reward == 1:
                    first_step_in_goal = steps_in_goal[i][j][0]
                else:
                    first_step_in_goal = np.nan
                if only_final_reward_matters:
                        success = final_reward
                else:
                    success = any_reward
                print("Task", j+1, ":", "Reached goal:", success)
                print("First Step in Goal:", first_step_in_goal)
                agent_success_rates.append(success)
                agent_goal_steps.append(first_step_in_goal)
                print()
            success_rates.append(agent_success_rates)
            first_steps_in_goal.append(agent_goal_steps)

        # print summary, i.e. average success rate overall and of each agent
        print("Summary:")
        for i in range(len(trajectories)):
            print(agent_names[i], "success rate:", np.mean(success_rates[i]))
            try:
                print(agent_names[i], "average steps to reach goal first time:", np.nanmean(first_steps_in_goal[i]))
            except ZeroDivisionError or IndexError:
                print(agent_names[i], "average steps to reach goal first time: N/A, didn't solve any task")
            
    
    # if trajectories contain trajectories of multiple seeds per agent, average the performances of the agents
    elif type(trajectories[0][0][0][0]) == tuple:
        # extract success and first step in goal for each task for each seed of each agent
        success_rates = []
        first_steps_in_goal = []
        for i in range(len(trajectories)):
            agent_success_rates = []
            agent_first_steps_in_goal = []
            for j in range(len(trajectories[i])):
                seed_success_rates = []
                seed_first_steps_in_goal = []
                for k in range(len(trajectories[i][j])):
                    all_rewards = [traj[2] for traj in trajectories[i][j][k]]
                    any_reward = np.sum(all_rewards) >= 1
                    final_reward = trajectories[i][j][k][-1][2]
                    if len(steps_in_goal[i][j][k]) > 0 and final_reward == 1:
                        first_step_in_goal = steps_in_goal[i][j][k][0]
                    else:
                        first_step_in_goal = np.nan
                    if only_final_reward_matters:
                        success = final_reward
                    else:
                        success = any_reward
                    seed_success_rates.append(success)
                    seed_first_steps_in_goal.append(first_step_in_goal)
                agent_success_rates.append(seed_success_rates)
                agent_first_steps_in_goal.append(seed_first_steps_in_goal)
            success_rates.append(agent_success_rates)
            first_steps_in_goal.append(agent_first_steps_in_goal)

        # print summary, i.e. average success rate overall and of each agent
        print("Summary:")
        for i in range(len(trajectories)):
            # compute average success rates per seed ([avg_success_seed1, avg_success_seed2, ...])
            average_success_rates_per_seed = np.mean(success_rates[i], axis=1)
            print(agent_names[i], "is evaluated across", len(success_rates[i]), "seeds.")
            print(agent_names[i], "success rates mean per seed:", average_success_rates_per_seed)
            print(agent_names[i], "success rate mean across seeds:", np.mean(average_success_rates_per_seed))
            print(agent_names[i], "success rate std across seeds:", np.std(average_success_rates_per_seed))
            try:
                print(agent_names[i], "average steps to reach goal first time:", np.nanmean(first_steps_in_goal[i]))
            except ZeroDivisionError or IndexError:
                print(agent_names[i], "average steps to reach goal first time: N/A, didn't solve any task")


def evaluate_checkpoints(checkpoint_paths, labels, nr_tasks=500, environment_name="point_Spiral11x11", epsilon=0.1):
    """
    Performs the experiment 02 evaluation using the given checkpoints.
    Evaluation consists of:
        - loading the checkpoints' encoders (making up the critic) and actors
        - sampling 100 tasks from the environment
        - letting a paramerized, greedy, and random policy solve the tasks
        - plotting and saving the results in a bar plot

    Done for embedded list of checkpoints containing different seeds for different actor types (greedy vs. parameterized).

    Args:
        checkpoint_paths: List of of lists of paths to the checkpoints, each embedded path is for one seed.
        labels: List of labels for the checkpoints.
        nr_tasks: Number of tasks to evaluate on.
        environment_name: Name of environment to evaluate on.
        epsilon: Epsilon value for the epsilon-greedy policy
    
    Returns:
        None
    """
    # load the environment, obs_dim, and action grid
    env, obs_dim = get_env(environment_name, return_obs_dim=True, seed=42, return_raw_gym_env=True)
    action_grid_9 = scale_action_grid(basic_action_grid, 0)
    action_grid_25 = scale_action_grid(basic_action_grid, 1)
    action_grid_81 = scale_action_grid(basic_action_grid, 2)

    action_grids = {
        "09": action_grid_9,
        "25": action_grid_25,
        "81": action_grid_81
    }

    # sample tasks
    tasks = sample_tasks(env, nr_tasks)

    # load the encoders and actors
    sa_encoders = []
    g_encoders = []
    actors = []
    for i,agent_type in enumerate(labels):
        agent_type_sa_encoders = []
        agent_type_g_encoders = []
        agent_type_actors = []
        for j,checkpoint_path in enumerate(checkpoint_paths[i]):
            sa_encoder, g_encoder = load_encoders(env, checkpoint_path+"/critic_params.txt")
            if agent_type.lower() == "contrastive" or agent_type.lower() == "parameterized":
                actor = ContrastiveAgent(env, checkpoint_path, obs_dim)
            elif agent_type.lower() == "greedy" or agent_type.lower().split(" ")[0] == "greedy":
                #print("Grid key:", agent_type.lower().split(" ")[1][1:3])
                actor = GreedyAgent(env, action_grids[agent_type.lower().split(" ")[1][1:3]], checkpoint_path, obs_dim, epsilon=epsilon, value_type="contrastive_critic")
            elif agent_type.lower() == "random":
                actor = RandomAgent(action_grid_25, obs_dim, env)
            else:
                raise ValueError("Invalid agent type, check labels. They must be either ")
            agent_type_sa_encoders.append(sa_encoder)
            agent_type_g_encoders.append(g_encoder)
            agent_type_actors.append(actor)
        sa_encoders.append(agent_type_sa_encoders)
        g_encoders.append(agent_type_g_encoders)
        actors.append(agent_type_actors)

    # let the agents solve the tasks, record the trajectories and steps in goal in embedded lists
    print("Solving tasks...")
    trajectory_collection = []
    steps_in_goal_collection = []
    for i,agent_type in enumerate(labels):
        agent_trajectories = []
        agent_steps_in_goal = []
        for j in range(len(actors[i])):
            seed_trajectory_collection = []
            seed_steps_in_goal_collection = []
            print("Agent:", agent_type, "Seed:", j)
            for task in tasks:
                trajectory, steps_in_goal = actors[i][j].solve_task(task)
                seed_trajectory_collection.append(trajectory)
                seed_steps_in_goal_collection.append(steps_in_goal)
            agent_trajectories.append(seed_trajectory_collection)
            agent_steps_in_goal.append(seed_steps_in_goal_collection)
        trajectory_collection.append(agent_trajectories)
        steps_in_goal_collection.append(agent_steps_in_goal)
    
    # evaluate the performances of the agents
    evaluate_performances(trajectory_collection, steps_in_goal_collection, labels, nr_tasks)

def record_inference_times(checkpoint_paths, labels, nr_tasks=500, environment_name="point_Spiral11x11", epsilon=0.1):
    """
    Uses the checkpoint paths to construct different actors.
    Then, the inference time of each actor is recorded by solving a number of tasks.
    Times are averaged over the tasks and seeds and an overview is printed.

    Args:
        checkpoint_paths: List of paths to the checkpoints.
        labels: List of labels for the checkpoints.
        nr_tasks: Number of tasks to evaluate on.
        environment_name: Name of environment to evaluate on.
        epsilon: Epsilon value for the epsilon-greedy policy
    """
    # load the environment, obs_dim, and action grid
    env, obs_dim = get_env(environment_name, return_obs_dim=True, seed=42, return_raw_gym_env=True)
    action_grid_9 = scale_action_grid(basic_action_grid, 0)
    action_grid_25 = scale_action_grid(basic_action_grid, 1)
    action_grid_81 = scale_action_grid(basic_action_grid, 2)

    action_grids = {
        "09": action_grid_9,
        "25": action_grid_25,
        "81": action_grid_81
    }

    # sample tasks
    tasks = sample_tasks(env, nr_tasks)

    # load the encoders and actors
    sa_encoders = []
    g_encoders = []
    actors = []
    for i,agent_type in enumerate(labels):
        agent_type_sa_encoders = []
        agent_type_g_encoders = []
        agent_type_actors = []
        for j,checkpoint_path in enumerate(checkpoint_paths[i]):
            sa_encoder, g_encoder = load_encoders(env, checkpoint_path+"/critic_params.txt")
            if agent_type.lower() == "contrastive" or agent_type.lower() == "parameterized":
                actor = ContrastiveAgent(env, checkpoint_path, obs_dim)
            elif agent_type.lower() == "greedy" or agent_type.lower().split(" ")[0] == "greedy":
                #print("Grid key:", agent_type.lower().split(" ")[1][1:3])
                actor = GreedyAgent(env, action_grids[agent_type.lower().split(" ")[1][1:3]], checkpoint_path, obs_dim, epsilon=epsilon, value_type="contrastive_critic")
            elif agent_type.lower() == "random":
                actor = RandomAgent(action_grid_25, obs_dim, env)
            else:
                raise ValueError("Invalid agent type, check labels. They must be either ")
            agent_type_sa_encoders.append(sa_encoder)
            agent_type_g_encoders.append(g_encoder)
            agent_type_actors.append(actor)
        sa_encoders.append(agent_type_sa_encoders)
        g_encoders.append(agent_type_g_encoders)
        actors.append(agent_type_actors)

    # sample tasks
    tasks = sample_tasks(env, nr_tasks)

    # record inference times
    print("Recording inference times...")
    inference_times = []
    for i,agent_type in enumerate(labels):
        agent_inference_times = []
        for j in range(len(actors[i])):
            print("Agent:", agent_type, "Seed:", j)
            for task in tasks:
                start_time = time.time()
                actors[i][j].solve_task(task)
                end_time = time.time()
                inference_time = end_time - start_time
                agent_inference_times.append(inference_time)
        inference_times.append(agent_inference_times)
    
    # print the average inference times
    print("Average Inference Times:")
    for i,agent_type in enumerate(labels):
        print("Agent:", agent_type)
        print("Average Inference Time:", np.mean(inference_times[i]))
        print("Inference Time Std:", np.std(inference_times[i]))
        print()



if __name__ == "__main__":
    params_path = "manual_checkpoints/two_encoders/point_Spiral11x11/1_million_steps_original"
    actor_path = params_path + "/policy_params.txt"

    inspect_params(actor_path)

    policy_net = load_policy_net(actor_path)

    inspect_network(policy_net)