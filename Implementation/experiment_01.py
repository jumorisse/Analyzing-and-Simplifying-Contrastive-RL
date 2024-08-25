from experiment_utils import inspect_params, load_encoders, inspect_network, get_env, sample_env_states, visualize_states, get_sa_encodings, get_g_encodings, reduce_dim, plot_encodings, critic_dist, get_action_colors, generate_env_states, get_steps_to_center
from experiment_02 import scale_action_grid, basic_action_grid
import numpy as np
import matplotlib.pyplot as plt

def plot_dist_grid(dist_grid, save_path):
    """
    Gets a grid with cells containing distance measures.
    Plots the grid using color coding to indicate distance.
    Saves the plot to a specified path.

    Args:
    - dist_grid: grid with distance measures
    - save_path: path where the plot is saved

    Returns:
    - None
    """
    # get dimensions of grid
    rows = dist_grid.shape[0]
    cols = dist_grid.shape[1]
    # define color map, simple map from blue to red
    cmap = plt.get_cmap('coolwarm')
    # plot the grid
    plt.figure()
    plt.imshow(dist_grid, cmap=cmap)
    # add color bar with description of color coding
    cbar = plt.colorbar()
    cbar.set_label("Critic Value")
    # adjust x and y ticks to show the actions per x-/y-coordinate
    # action range is from -1 to 1
    plt.xticks(np.arange(0, cols, 1), np.linspace(-1, 1, cols))
    plt.yticks(np.arange(0, rows, 1), np.linspace(-1, 1, rows))
    plt.xlabel("2nd Action Dimension")
    plt.ylabel("1st Action Dimension")

    plt.savefig(save_path)
    plt.close()

def neighborhood_investigation(start_state, neighbor_states, sa_encoders, g_encoders, save_dir, action_grid_scale=2):
    """
    Investigates the relations between goal and state-action representations in a small local neighborhood.
    Considers actions from an action grid. Computes distances between state-action representations and goal representations.
    Result is distance grid which is visualized using color coding to indicate distance to goal.
    Produces several plots:
    - one for goal equalling to the start state
    - one for each of the neighbor states

    Critic values are computed for each seed, i.e. once using the sa-encoder and g-encoder of each seed.

    Args:
    - start_state (array): the state for which the neighborhood is investigated
    - neighbor_states (list of arrays): the states that are considered neighbors of the start state
    - sa_encoders (list of sa-encoders): list of sa-encoders for each seed
    - g_encoders (list of g-encoders): list of g-encoders for each seed

    Returns:
    - None
    """
    # visualize all states (start + neighbors)
    start_color = [255,0,0]
    # get colors for neighborhood states (gradient from [0,255,0] to [0,0,255])
    neighborhood_colors = []
    for i in range(len(neighbor_states)):
        # first neighbor is always green ([0,255,0]), last neighbor is always blue ([0,0,255])
        color = [0, 255 - i*255//(len(neighbor_states)-1), i*255//(len(neighbor_states)-1)]
        neighborhood_colors.append(color)
    print("Nr. of neighbors:", len(neighbor_states))
    print("Neighbor colors:", neighborhood_colors)
    visualize_states(env, [start_state]+neighbor_states, color_meaning="ids", save_path=save_dir+"neighborhood_0_spiral11x11.png", state_colors=[start_color]+neighborhood_colors, radius=8)
    # get action grid
    action_grid = scale_action_grid(basic_action_grid,action_grid_scale)
    action_grid_rows = action_grid.shape[0]
    action_grid_cols = action_grid.shape[1]

    # initialize list to store distance grids, will have shape [[dist_grid1_seed1, dist_grid2_seed1, ...], [dist_grid1_seed2, dist_grid2_seed2, ...], ... , [dist_grid1_seed5, dist_grid2_seed5, ...]]
    dist_grids = []
    for seed in range(len(sa_encoders)):
        sa_encoder = sa_encoders[seed]
        g_encoder = g_encoders[seed]
        seed_dist_grids = []
        # get sa-encodings for all cells in action grid
        encoding_dim = 64
        sa_encoding_grid = np.zeros((action_grid_rows, action_grid_cols, encoding_dim))
        for i in range(action_grid.shape[0]):
            for j in range(action_grid.shape[1]):
                sa_encoding_grid[i,j] = get_sa_encodings(sa_encoder, [start_state], [action_grid[i,j]])[0]
        # producing investigation for start_state=goal
        goal_encoding = get_g_encodings(g_encoder, [start_state])[0]
        dist_grid = np.zeros((action_grid_rows, action_grid_cols, 1))
        for i in range(action_grid.shape[0]):
            for j in range(action_grid.shape[1]):
                dist_grid[i,j] = np.dot(sa_encoding_grid[i,j], goal_encoding)
        seed_dist_grids.append(dist_grid)
        # producing investigation for each neighbor state
        for neighbor_state in neighbor_states:
            goal_encoding = get_g_encodings(g_encoder, [neighbor_state])[0]
            dist_grid = np.zeros((action_grid_rows, action_grid_cols, 1))
            for i in range(action_grid.shape[0]):
                for j in range(action_grid.shape[1]):
                    dist_grid[i,j] = np.dot(sa_encoding_grid[i,j], goal_encoding)
            seed_dist_grids.append(dist_grid)
        dist_grids.append(seed_dist_grids)
    
    print("Shape of dist grids of all seeds:", np.array(dist_grids).shape)
    # average over seeds, results in [avg_dist_grid1, avg_dist_grid2, ...]
    avg_dist_grids = np.mean(dist_grids, axis=0)
    print("Shape of avg dist grids:", avg_dist_grids.shape)
    # plotting distance grids
    for i, dist_grid in enumerate(avg_dist_grids):
        save_path = save_dir+f"neighborhood_{i+1}.png"
        plot_dist_grid(dist_grid, save_path)    

def normalize_encodings(encodings):
    """
    Normalizes encoding vectors to have unit length.

    Args:
    - encodings: list of encoding vectors

    Returns:
    - normalized_encodings: list of normalized encoding vectors
    """
    normalized_encodings = []
    for encoding in encodings:
        normalized_encoding = encoding / np.linalg.norm(encoding)
        normalized_encodings.append(normalized_encoding)
    return normalized_encodings

def dot_product(array1, array2):
    """
    Computes the dot product between two arrays.

    Args:
    - array1 (np.array): first array
    - array2 (np.array): second array

    Returns:
    - (float) dot product between array1 and array2
    """
    return np.dot(array1, array2)

def get_extreme_dots(encodings):
    """
    Gets a list of encodings, computes the dot product between each pair of encodings.
    Returns the minimum and maximum dot product.

    Args:
    - encodings (list of np.arrays): list of encoding vectors

    Returns:
    - min_dot (float): minimum dot product
    - max_dot (float): maximum dot product
    """
    min_dot = np.inf
    max_dot = -np.inf
    for i in range(len(encodings)):
        for j in range(i+1, len(encodings)):
            dot = dot_product(encodings[i], encodings[j])
            if dot < min_dot:
                min_dot = dot
            if dot > max_dot:
                max_dot = dot
    
    return min_dot, max_dot

def get_dot_distance_func(min_dot, max_dot, normalize=False):
    """
    Returns a distance function that computes the dot product distance between two arrays.
    The raw dot product is a similarity measure in range [-inf, inf].
    This function maps the dot product to a distance measure in range [0,inf] or [0,1] (depending on normalize flag).
    """
    def dot_distance(array1, array2):
        """
        Computes the dot product between array1 and array2.
        The dot product is a similarity measure in range [-inf, inf].
        This function maps the dot product to a distance measure in range [0,inf] or [0,1] (depending on normalize flag).
        """
        dot_sim = dot_product(array1, array2)

        if normalize:
            dot_sim = (dot_sim - min_dot) / (max_dot - min_dot)
            dot_dist = 1 - dot_sim
        else:
            dot_dist = -dot_sim + max_dot
        
        return dot_dist
    return dot_distance

def state_vs_goal_investigation(states, sa_encoders, g_encoders, save_dir, goals=["center", "entrance", "middle"]):
    """
    Visualizes critic values between set of state-[0,0] encodings and goal encodings.
    A plot each for each goal listed.
    Plot draws states colored by their respective critic value given the plot's goal.

    Average critic values across seeds, one element in sa_encoders and g_encoders corresponds to one seed.

    Args:
    - states (list of arrays): list of states
    - sa_encoders (list of sa-encoders): list of sa-encoders for each seed
    - g_encoders (list of g-encoders): list of g-encoders for each seed
    - save_dir (str): directory where plots are saved
    - goals (list of str): list of goals, can be "center", "entrance", "middle"
    """
    critic_values = []
    for seed in range(len(sa_encoders)):
        sa_encoder = sa_encoders[seed]
        g_encoder = g_encoders[seed]
        # identify goal encodings
        goal_encodings = []
        steps_to_center = get_steps_to_center(env, states)
        steps_to_goal = []
        for g in goals:
            if g == "center":
                goal = [5.5,5.5]
                goal_encodings.append(get_g_encodings(g_encoder, [goal])[0])
                steps = steps_to_center
                steps_to_goal.append(steps)
            elif g == "entrance":
                goal = [10.5,10.5]
                goal_encodings.append(get_g_encodings(g_encoder, [goal])[0])
                # if entrance is goal, steps are reversed steps to center
                steps = steps_to_center[::-1]
                steps_to_goal.append(steps)
            elif g == "middle":
                goal = [9.5,1.5]
                goal_encodings.append(get_g_encodings(g_encoder, [goal])[0])
                # extract number of steps from middle to center, i.e. the value of the middle point of steps_to_center
                max_steps = steps_to_center[len(steps_to_center)//2]
                steps_to_middle  = [abs(step - max_steps) for step in steps_to_center]
                steps = steps_to_middle
                steps_to_goal.append(steps)
        
        # get sa-encodings for [0,0] action
        sa_encodings = get_sa_encodings(sa_encoder, states, [np.array([0,0])])

        # get critic values for each state-[0,0] encoding and each goal encoding
        seed_critic_values = []
        for goal_encoding in goal_encodings:
            goal_critic_values = []
            for sa_encoding in sa_encodings:
                goal_critic_values.append(np.dot(sa_encoding, goal_encoding))
            seed_critic_values.append(goal_critic_values)
        critic_values.append(seed_critic_values)
    
    print("Shape of seed critic values for all seeds:", np.array(critic_values).shape)
    # average over seeds
    avg_critic_values = np.mean(critic_values, axis=0)
    print("Shape of avg critic values:", avg_critic_values.shape)
    
    # plot critic values for each goal
    for i, goal in enumerate(goals):
        # turn list's 0-dimensional arrays into floats
        state_values = [float(value) for value in avg_critic_values[i]]
        save_path = save_dir+f"critic_values_{goal}_steps_spiral11x11.png"
        visualize_states(env, states, color_meaning="steps", save_path=save_path, steps=steps_to_goal[i])
        save_path = save_dir+f"critic_values_{goal}_spiral11x11.png"
        visualize_states(env, states, color_meaning="critic", save_path=save_path, state_colors=state_values)



if __name__ == "__main__":
    # parameter for the experiment
    sample_states = False # whether to randomly sample states or generate evenly spaced states
    sample_actions_for_all_states = False # whether to sample actions for all states or only for a single state
    if sample_states:
        n_state_samples = 10000
        sample_actions_for_all_states = False # to prevent crashing (too many states)
        perplexity = 100
    else:
        generation_factor = 3 # used to generate evenly spaced states, the higher the more states (e.g. 3 -> 217 states)
        perplexity = 20
        if sample_actions_for_all_states:
            perplexity = 30
    n_actions = 10 # nr of actions sampled per state
    normalize_representations = False # whether to normalize the encodings to have unit length
    inspect_encoders = False

    # parameter used during training (used to construct paths for loading encoder parameter)
    environment_name = "point_Spiral11x11"
    encoder_nr = "two_encoders" # "only_sa" or "two_encoders"
    actor = "original" # "original", "greedy_randominit_25actions"
    if actor == "greedy_randominit_25actions":
        action_grid_scale = 1
    else:
        action_grid_scale = 2
    if sample_actions_for_all_states:
        subdir = "with_sampled_actions"
    else:
        subdir = "no_sampled_actions"

    print("Performing experiment after contrastive rl done with:")
    print("Environment:", environment_name)
    print("Encoder:", encoder_nr)
    print("Actor used during CRL:", actor)

    # loading sa- and g-encoder parameter and inspecting them
    critic_path = f"manual_checkpoints/{encoder_nr}/{environment_name}/{actor}_seed0/critic_params.txt"
    actor_path = f"manual_checkpoints/{encoder_nr}/{environment_name}/{actor}_seed0/policy_params.txt"
    save_dir = f"experiment_results/experiment_01/{actor}/{subdir}/"
    print("Inspecting Critic Parameters")
    inspect_params(critic_path)
    print("Inspecting Actor Parameters")
    inspect_params(actor_path)


    print("\n##########################")
    print("#### Loading Networks ####")
    print("##########################")
    # reconstructing sa- and g-encoder from loaded parameter, potentially also inspecting the final networks
    sa_encoder, g_encoder = load_encoders(environment_name, critic_path)
    if inspect_encoders:
        print("#### Inspecting State-Action Encoder ####")
        inspect_network(sa_encoder)
        print()
        print("#### Inspecting Goal Encoder ####")
        inspect_network(g_encoder)


    print("\n#############################")
    print("#### Loading Environment ####")
    print("#############################")
    env = get_env(environment_name)


    print("\n#############################")
    print("###### Getting States #######")
    print("#############################")

    if sample_states:
        print("Getting", n_state_samples, "states from the environment.")
        states = sample_env_states(env, n_state_samples)
    else:
        states = generate_env_states(env, environment_name, generation_factor)
        print("Generated", len(states), " evenly spaced states from the environment.")
    n_states = len(states)
    # define a state, two of its direct neighbors, the actions leading to the neighbors, and a color for each state
    # in the following states are referred to as neighborhood states
    s_0 = np.array([1.5,5.5])
    s_1 = np.array([1.5,4.5])
    s_2 = np.array([1.5,6.5])

    # render states
    print("Visualizing used states.")
    state_colors, colorbar, colorbar_ticks = visualize_states(env, states, color_meaning="steps", save_path=save_dir+"states_spiral11x11.png")


    print("\n#############################")
    print("########### Encoding ##########")
    print("###############################")
    g_encodings = get_g_encodings(g_encoder, states)
    g_colors = state_colors

    # getting encodings for fixed action (0,0) and (if sampling for all states) sampled actions for all states
    if not sample_actions_for_all_states:
        sa_encodings = get_sa_encodings(sa_encoder, states, [np.array([0,0])])
        sa_encodings_fixed = sa_encodings
        sa_colors = state_colors
    else:
        sampled_actions = np.array([np.random.uniform(-1, 1, (n_actions, 2)) for _ in range(n_states)])
        action_colors = []
        for i in range(n_states):
            state_action_colors, _, _ = get_action_colors(sampled_actions[i], state_color=state_colors[i], color=True)
            action_colors += state_action_colors
        sa_encodings_samples = get_sa_encodings(sa_encoder, states, sampled_actions)
        sa_encodings_fixed = get_sa_encodings(sa_encoder, states, [np.array([0,0])])
        sa_encodings = sa_encodings_fixed + sa_encodings_samples
        sa_colors = state_colors + action_colors
    isolated_sa_colors = state_colors


    print("\n####################################")
    print("##### Mapping Encodings to 2d #####")
    print("####################################")
    all_encodings = sa_encodings + g_encodings

    min_dot, max_dot = get_extreme_dots(all_encodings)
    mapping_metric = get_dot_distance_func(min_dot, max_dot) # "euclidean", "cosine", or get_dot_distance_func(min_dot, max_dot)

    encodings_dict = {
        "sa_encodings_zero": sa_encodings[:n_states],
        "sa_encodings_sampled": sa_encodings[n_states:],
        "g_encodings": g_encodings
    }
    # store encodings dictionary in a file
    np.save(save_dir+"encodings_dict.npy", encodings_dict)

    if normalize_representations:
        print("Normalizing Encodings")
        all_encodings = normalize_encodings(all_encodings)
        print("Checking normalization...")
        print("Max Norm:", np.max([np.linalg.norm(encoding) for encoding in all_encodings]))
    all_encodings_2d = reduce_dim(all_encodings, method="TSNE", target_dim=2, perplexity=perplexity, distance_metric=mapping_metric)
    isolated_sa_encodings_2d = reduce_dim(sa_encodings_fixed, method="TSNE", target_dim=2, perplexity=20, distance_metric=mapping_metric)
    sa_encodings_2d = all_encodings_2d[:len(sa_encodings)]
    g_encodings_2d = all_encodings_2d[len(sa_encodings):len(sa_encodings)+len(g_encodings)]


    print("\n####################################")
    print("####### Plotting Encodings #########")
    print("####################################")
    # if each state has sampled actions, we want to reverse the sa encodings and colors so that the SA(s,[0,0]) encodings are drawn last and in the foreground
    if sample_actions_for_all_states:
        # reverse sa_encodings_2d array
        sa_encodings_2d = sa_encodings_2d[::-1]
        # reverse sa_colors list
        sa_colors.reverse()
        zero_action_sa_encodings_2d = sa_encodings_2d[-n_states:]
        zero_action_colors = sa_colors[-n_states:]
    else:
        zero_action_sa_encodings_2d = sa_encodings_2d
        zero_action_colors = sa_colors
    
    plot_encodings(np.concatenate([zero_action_sa_encodings_2d, g_encodings_2d]), save_dir+"sa_and_g_encodings_spiral11x11.png", np.concatenate([zero_action_colors, g_colors]), colorbar, colorbar_ticks)
    plot_encodings(isolated_sa_encodings_2d, save_dir+"isolated_sa_encodings_spiral11x11.png", np.concatenate([isolated_sa_colors]), colorbar, colorbar_ticks)
    if sample_actions_for_all_states:
        # prepare gray scale colorbar ticks, gray scale shows aciton strength from 0 to sqrt(2)
        # compute colorbar ticks, we want to have 6 ticks (max as upper bound, min as lower bound, and 4 in between)
        value_range = np.sqrt(2)
        gray_ticks = [np.round(0 + i * value_range / 5, 2) for i in range(6)]
        print("gray ticks:", gray_ticks)
        plot_encodings(np.concatenate([sa_encodings_2d, g_encodings_2d]), save_dir+"all_encodings_spiral11x11.png", np.concatenate([sa_colors, g_colors]), colorbar, [colorbar_ticks, gray_ticks], (12,6))
    

    # the critic visualizations are done using averages across seeds, here we define the seeds so that we can load there respective encoders
    #seeds = [0, 21, 42, 97, 1453]
    seeds = [0, 21, 42]
    seed_paths = [f"manual_checkpoints/{encoder_nr}/{environment_name}/{actor}_seed" + str(seed) for seed in seeds]
    sa_encoders = [load_encoders(environment_name, seed_path+"/critic_params.txt")[0] for seed_path in seed_paths]
    g_encoders = [load_encoders(environment_name, seed_path+"/critic_params.txt")[1] for seed_path in seed_paths]

    print("\n####################################")
    print("##### Critic Value Investigation (when varying action for neighbor goals) #######")
    print("####################################")
    neighborhood_investigation(s_0, [s_1, s_2], sa_encoders, g_encoders, save_dir, action_grid_scale=action_grid_scale)

    print("\n####################################")
    print("##### Critic Value Investigation (when varying states-[0,0] for fixed goal) #######")
    print("####################################")
    state_vs_goal_investigation(states, sa_encoders, g_encoders, save_dir)