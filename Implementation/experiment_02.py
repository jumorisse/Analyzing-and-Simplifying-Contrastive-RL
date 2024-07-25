import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import TransformedDistribution, TanhTransform, Normal
from experiment_utils import get_env, sample_tasks, load_encoders, load_policy_net, generate_env_states, evaluate_performances, scale_action_grid, get_action_strengths, ContrastiveAgent, GreedyAgent, RandomAgent, basic_action_grid
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
    n_tasks = 2 # number of tasks to sample, for reproduction should be 100
    max_episode_length = 3 # maximum number of steps in an episode, for reproduction should be 100
    #seeds  = [0, 21, 42, 97, 1453]
    seeds = [0, 21]
    params_paths = ["manual_checkpoints/two_encoders/point_Spiral11x11/original_seed"+str(seed) for seed in seeds]
    numpy_seed = 42 # seed for numpy, for reproduction should be 42
    action_grid_scale = 2 # scaling factor for the action grid from which greedy actor chooses, for reproduction should be 2
    np.random.seed(numpy_seed)
    contrastive_eval_mode = False

    evaluating_performances = False
    computing_action_distances = True
    visualizing_actions = False

    # get the action grid, i.e. the actions the greedy agent can select from
    action_grid = scale_action_grid(basic_action_grid, action_grid_scale)
    print("Action Grid Shape:", action_grid.shape)
    print("Total Number of Actions:", action_grid.size/2)
    strength_grid = get_action_strengths(action_grid)
    # load the environment
    env, obs_dim = get_env(environment_name, return_obs_dim=True, seed=42, return_raw_gym_env=True)

    # sample tasks
    tasks = sample_tasks(env, n_tasks)

    # initialize lists to store agents of different seeds
    greedy_agents_00 = []
    greedy_agents_01 = []
    greedy_agents_05 = []
    greedy_agents_10 = []
    random_agents = []
    contrastive_agents = []

    for params_path in params_paths:
        # create agents
        greedy_agents_00.append(GreedyAgent(env=env, action_grid=action_grid, params_path=params_path, obs_dim=obs_dim, epsilon=0.0, value_type="contrastive_critic"))
        greedy_agents_01.append(GreedyAgent(env=env, action_grid=action_grid, params_path=params_path, obs_dim=obs_dim, epsilon=0.01, value_type="contrastive_critic"))
        greedy_agents_05.append(GreedyAgent(env=env, action_grid=action_grid, params_path=params_path, obs_dim=obs_dim, epsilon=0.05, value_type="contrastive_critic"))
        greedy_agents_10.append(GreedyAgent(env=env, action_grid=action_grid, params_path=params_path, obs_dim=obs_dim, epsilon=0.1, value_type="contrastive_critic"))
        random_agents.append(RandomAgent(action_grid, obs_dim, env))
        contrastive_agents.append(ContrastiveAgent(env=env, params_path=params_path, obs_dim=obs_dim, eval_mode=contrastive_eval_mode))

    if evaluating_performances:
        print("##########################")
        print("Evaluating Performances")
        print("##########################")
        # record agent performances on tasks
        greedy_00_trajectories = []
        greedy_00_steps_in_goal = []
        greedy_01_trajectories = []
        greedy_01_steps_in_goal = []
        greedy_05_trajectories = []
        greedy_05_steps_in_goal = []
        greedy_10_trajectories = []
        greedy_10_steps_in_goal = []
        random_trajectories = []
        random_steps_in_goal = []
        contrastive_trajectories = []
        contrastive_steps_in_goal = []
        for i in range(len(random_agents)):
            print("Seed", seeds[i])
            seed_greedy_00_trajectories = []
            seed_greedy_00_steps_in_goal = []
            seed_greedy_01_trajectories = []
            seed_greedy_01_steps_in_goal = []
            seed_greedy_05_trajectories = []
            seed_greedy_05_steps_in_goal = []
            seed_greedy_10_trajectories = []
            seed_greedy_10_steps_in_goal = []
            seed_random_trajectories = []
            seed_random_steps_in_goal = []
            seed_contrastive_trajectories = []
            seed_contrastive_steps_in_goal = []
            for j,task in enumerate(tasks):
                print("Solving Task, ", j+1)
                seed_greedy_00_results = greedy_agents_00[i].solve_task(task)
                seed_greedy_00_trajectories.append(seed_greedy_00_results[0])
                seed_greedy_00_steps_in_goal.append(seed_greedy_00_results[1])
                seed_greedy_01_results = greedy_agents_01[i].solve_task(task)
                seed_greedy_01_trajectories.append(seed_greedy_01_results[0])
                seed_greedy_01_steps_in_goal.append(seed_greedy_01_results[1])
                seed_greedy_05_results = greedy_agents_05[i].solve_task(task)
                seed_greedy_05_trajectories.append(seed_greedy_05_results[0])
                seed_greedy_05_steps_in_goal.append(seed_greedy_05_results[1])
                seed_greedy_10_results = greedy_agents_10[i].solve_task(task)
                seed_greedy_10_trajectories.append(seed_greedy_10_results[0])
                seed_greedy_10_steps_in_goal.append(seed_greedy_10_results[1])
                seed_random_results = random_agents[i].solve_task(task)
                seed_random_trajectories.append(seed_random_results[0])
                seed_random_steps_in_goal.append(seed_random_results[1])
                seed_contrastive_results = contrastive_agents[i].solve_task(task)
                seed_contrastive_trajectories.append(seed_contrastive_results[0])
                seed_contrastive_steps_in_goal.append(seed_contrastive_results[1])
            # store results
            greedy_00_trajectories.append(seed_greedy_00_trajectories)
            greedy_00_steps_in_goal.append(seed_greedy_00_steps_in_goal)
            greedy_01_trajectories.append(seed_greedy_01_trajectories)
            greedy_01_steps_in_goal.append(seed_greedy_01_steps_in_goal)
            greedy_05_trajectories.append(seed_greedy_05_trajectories)
            greedy_05_steps_in_goal.append(seed_greedy_05_steps_in_goal)
            greedy_10_trajectories.append(seed_greedy_10_trajectories)
            greedy_10_steps_in_goal.append(seed_greedy_10_steps_in_goal)
            random_trajectories.append(seed_random_trajectories)
            random_steps_in_goal.append(seed_random_steps_in_goal)
            contrastive_trajectories.append(seed_contrastive_trajectories)
            contrastive_steps_in_goal.append(seed_contrastive_steps_in_goal)

        # evaluate agent performances
        evaluate_performances([greedy_00_trajectories, greedy_01_trajectories, greedy_05_trajectories, greedy_10_trajectories, random_trajectories, contrastive_trajectories], [greedy_00_steps_in_goal, greedy_01_steps_in_goal, greedy_05_steps_in_goal, greedy_10_steps_in_goal, random_steps_in_goal, contrastive_steps_in_goal], ["Greedy_00", "Greedy_01", "Greedy_05", "Greedy_10", "Random", "Contrastive"])

    if computing_action_distances:
        print("##########################")
        print("Comparing Actions")
        print("##########################")
        euclidean_distances = []
        cosine_similarities = []
        for i,seed in enumerate(seeds):
            print("Seed", seed)
            seed_euclidean_distances = []
            seed_cosine_similarities = []
            action_pairs = [(contrastive_agents[i], greedy_agents_00[i]), (contrastive_agents[i], random_agents[i]), (contrastive_agents[i], contrastive_agents[i])]
            for (agent_1, agent_2) in action_pairs:
                pair_euclidean_distances = []
                pair_cosine_similarities = []
                for i,task in enumerate(tasks):
                    #print("Comparing Actions for Task:", i+1)
                    eucl_dists, cos_sims = compare_actions(env=env, task=task, leading_agent=agent_1, trailing_agent=agent_2, obs_dim=obs_dim)
                    pair_euclidean_distances.append(eucl_dists)
                    pair_cosine_similarities.append(cos_sims)
                seed_euclidean_distances.append(pair_euclidean_distances)
                seed_cosine_similarities.append(pair_cosine_similarities)
            euclidean_distances.append(seed_euclidean_distances)
            cosine_similarities.append(seed_cosine_similarities)
                
        # computing mean, max, and min euclidean distances and cosine similarities for each pair of agents
        # compute across seeds and also compute the std of the mean across seeds
        for i,agent_names in enumerate([["ContrastiveAgent", "GreedyAgent"], ["ContrastiveAgent", "RandomAgent"], ["ContrastiveAgent", "ContrastiveAgent"]]):
            print("Comparing Actions of", agent_names[0], "(leading) and", agent_names[1], "(trailing) on", n_tasks, "tasks with", max_episode_length, "max episode steps.")
            # compute mean, max, and min euclidean distances and cosine similarities for each seed of this agent pair
            mean_euclidean_distances = []
            max_euclidean_distances = []
            min_euclidean_distances = []
            mean_cosine_similarities = []
            max_cosine_similarities = []
            min_cosine_similarities = []

            for j in range(len(seeds)):
                # compute mean, max, and min euclidean distances and cosine similarities for this seed
                mean_euclidean_distances.append(np.mean(euclidean_distances[j][i]))
                max_euclidean_distances.append(np.max(euclidean_distances[j][i]))
                min_euclidean_distances.append(np.min(euclidean_distances[j][i]))
                mean_cosine_similarities.append(np.mean(cosine_similarities[j][i]))
                max_cosine_similarities.append(np.max(cosine_similarities[j][i]))
                min_cosine_similarities.append(np.min(cosine_similarities[j][i]))
            
            print("###################################")
            print("### Comparison Across All Tasks ###")
            print("###################################")
            print("Mean Euclidean Distance between Actions:", np.mean(mean_euclidean_distances))
            print("Std of Mean Euclidean Distance between Actions:", np.std(mean_euclidean_distances))
            print("Max Euclidean Distance between Actions:", np.max(max_euclidean_distances))
            print("Min Euclidean Distance between Actions:", np.min(min_euclidean_distances))
            print()
            print("Mean Cosine Similarity between Actions:", np.mean(mean_cosine_similarities))
            print("Std of Mean Cosine Similarity between Actions:", np.std(mean_cosine_similarities))
            print("Max Cosine Similarity between Actions:", np.max(max_cosine_similarities))
            print("Min Cosine Similarity between Actions:", np.min(min_cosine_similarities))
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
        Scaling Factor: 5 (65x65=4225)
        Seed 42
        Summary:
        Greedy_100 success rate: 0.59
        Greedy_99 success rate: 0.59
        Greedy_95 success rate: 0.64
        Greedy_90 success rate: 0.63
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

    Scaling Factor: 4 (33x33=1089)
        Greedy_00 success rate: 0.57
        Greedy_00 average steps to reach goal first time: 6.508771929824562
        Greedy_01 success rate: 0.6
        Greedy_01 average steps to reach goal first time: 6.310344827586207
        Greedy_05 success rate: 0.6
        Greedy_05 average steps to reach goal first time: 9.017241379310345
        Greedy_10 success rate: 0.62
        Greedy_10 average steps to reach goal first time: 10.245901639344263
        Random success rate: 0.13
        Random average steps to reach goal first time: 23.0
        Contrastive success rate: 0.69
        Contrastive average steps to reach goal first time: 7.144927536231884
        ##########################
        Comparing Actions
        ##########################
        Comparing Actions of <__main__.ContrastiveAgent object at 0x7eff5f160f70> (leading) and <__main__.RandomAgent object at 0x7eff5f160fa0> (trailing) on 100 tasks with 100 max episode steps.
        ###################################
        ### Comparison Across All Tasks ###
        ###################################
        Mean Euclidean Distance between Actions: 1.2237433
        Max Euclidean Distance between Actions: 2.828427
        Min Euclidean Distance between Actions: 8.940697e-07

        Mean Cosine Similarity between Actions: -0.004637177765555542
        Max Cosine Similarity between Actions: 1.0000001192092896
        Min Cosine Similarity between Actions: -1.0000001192092896

        Comparing Actions of <__main__.ContrastiveAgent object at 0x7eff5f160f70> (leading) and <__main__.ContrastiveAgent object at 0x7eff5f160f70> (trailing) on 100 tasks with 100 max episode steps.
        ###################################
        ### Comparison Across All Tasks ###
        ###################################
        Mean Euclidean Distance between Actions: 9.3650704e-07
        Max Euclidean Distance between Actions: 9.596355e-05
        Min Euclidean Distance between Actions: 0.0

        Mean Cosine Similarity between Actions: 1.0
        Max Cosine Similarity between Actions: 1.0000002
        Min Cosine Similarity between Actions: 0.99998575

        ##########################
        Visualizing Actions of Greedy_00 Agent and Contrastive Agent
        ##########################
        Distances shape: (11, 11)
        Number of cells in base environment image: 326700

    Scaling Factor 3 (17x17=289)


    Scaling Factor 2 (9x9=81)
        Greedy_00 success rate: 0.56
        Greedy_00 average steps to reach goal first time: 4.872727272727273
        Greedy_01 success rate: 0.6
        Greedy_01 average steps to reach goal first time: 5.824561403508772
        Greedy_05 success rate: 0.61
        Greedy_05 average steps to reach goal first time: 8.152542372881356
        Greedy_10 success rate: 0.6
        Greedy_10 average steps to reach goal first time: 8.76271186440678
        Random success rate: 0.14
        Random average steps to reach goal first time: 24.833333333333332
        Contrastive success rate: 0.67
        Contrastive average steps to reach goal first time: 6.850746268656716
        ##########################
        Comparing Actions
        ##########################
        Comparing Actions of <__main__.ContrastiveAgent object at 0x7fd660d30f40> (leading) and <__main__.RandomAgent object at 0x7fd660d30f70> (trailing) on 100 tasks with 100 max episode steps.
        ###################################
        ### Comparison Across All Tasks ###
        ###################################
        Mean Euclidean Distance between Actions: 1.2587274
        Max Euclidean Distance between Actions: 2.828427
        Min Euclidean Distance between Actions: 0.0

        Mean Cosine Similarity between Actions: -0.0017395358945420916
        Max Cosine Similarity between Actions: 1.0000001192092896
        Min Cosine Similarity between Actions: -1.0000001192092896

        Comparing Actions of <__main__.ContrastiveAgent object at 0x7fd660d30f40> (leading) and <__main__.ContrastiveAgent object at 0x7fd660d30f40> (trailing) on 100 tasks with 100 max episode steps.
        ###################################
        ### Comparison Across All Tasks ###
        ###################################
        Mean Euclidean Distance between Actions: 9.397061e-07
        Max Euclidean Distance between Actions: 1.0402113e-05
        Min Euclidean Distance between Actions: 0.0

        Mean Cosine Similarity between Actions: 1.0
        Max Cosine Similarity between Actions: 1.0000002
        Min Cosine Similarity between Actions: 0.9999985

        ##########################
        Visualizing Actions of Greedy_00 Agent and Contrastive Agent
        ##########################
        Distances shape: (11, 11)
        Number of cells in base environment image: 326700


    Scaling Factor 1 (5x5=25)
        Greedy_00 success rate: 0.58
        Greedy_00 average steps to reach goal first time: 6.43859649122807
        Greedy_01 success rate: 0.64
        Greedy_01 average steps to reach goal first time: 7.524590163934426
        Greedy_05 success rate: 0.65
        Greedy_05 average steps to reach goal first time: 9.609375
        Greedy_10 success rate: 0.56
        Greedy_10 average steps to reach goal first time: 10.444444444444445
        Random success rate: 0.1
        Random average steps to reach goal first time: 38.142857142857146
        Contrastive success rate: 0.67
        Contrastive average steps to reach goal first time: 5.880597014925373
        ##########################
        Comparing Actions
        ##########################
        Comparing Actions of <__main__.ContrastiveAgent object at 0x7f37f533af10> (leading) and <__main__.RandomAgent object at 0x7f37f533af40> (trailing) on 100 tasks with 100 max episode steps.
        ###################################
        ### Comparison Across All Tasks ###
        ###################################
        Mean Euclidean Distance between Actions: 1.304517
        Max Euclidean Distance between Actions: 2.828427
        Min Euclidean Distance between Actions: 0.0

        Mean Cosine Similarity between Actions: 0.0030229974198491974
        Max Cosine Similarity between Actions: 1.0000001192092896
        Min Cosine Similarity between Actions: -1.0000001192092896

        Comparing Actions of <__main__.ContrastiveAgent object at 0x7f37f533af10> (leading) and <__main__.ContrastiveAgent object at 0x7f37f533af10> (trailing) on 100 tasks with 100 max episode steps.
        ###################################
        ### Comparison Across All Tasks ###
        ###################################
        Mean Euclidean Distance between Actions: 9.173586e-07
        Max Euclidean Distance between Actions: 4.4614077e-05
        Min Euclidean Distance between Actions: 0.0

        Mean Cosine Similarity between Actions: 1.0
        Max Cosine Similarity between Actions: 1.0000002
        Min Cosine Similarity between Actions: 0.99999154

        ##########################
        Visualizing Actions of Greedy_00 Agent and Contrastive Agent
        ##########################
        Distances shape: (11, 11)
        Number of cells in base environment image: 326700


    Scaling Factor 0 (3x3=9)
        Summary:
        Greedy_00 success rate: 0.52
        Greedy_00 average steps to reach goal first time: 4.686274509803922
        Greedy_01 success rate: 0.52
        Greedy_01 average steps to reach goal first time: 4.88
        Greedy_05 success rate: 0.56
        Greedy_05 average steps to reach goal first time: 6.109090909090909
        Greedy_10 success rate: 0.55
        Greedy_10 average steps to reach goal first time: 8.584905660377359
        Random success rate: 0.09
        Random average steps to reach goal first time: 4.875
        Contrastive success rate: 0.67
        Contrastive average steps to reach goal first time: 5.7611940298507465
        ##########################
        Comparing Actions
        ##########################
        Comparing Actions of <__main__.ContrastiveAgent object at 0x7f007142a130> (leading) and <__main__.RandomAgent object at 0x7f007142a160> (trailing) on 100 tasks with 100 max episode steps.
        ###################################
        ### Comparison Across All Tasks ###
        ###################################
        Mean Euclidean Distance between Actions: 1.4064063253038723
        Max Euclidean Distance between Actions: 2.8284271247461903
        Min Euclidean Distance between Actions: 0.0

        Mean Cosine Similarity between Actions: 0.007324839335468681
        Max Cosine Similarity between Actions: 1.00000004263042
        Min Cosine Similarity between Actions: -1.0000000366900599

        Comparing Actions of <__main__.ContrastiveAgent object at 0x7f007142a130> (leading) and <__main__.ContrastiveAgent object at 0x7f007142a130> (trailing) on 100 tasks with 100 max episode steps.
        ###################################
        ### Comparison Across All Tasks ###
        ###################################
        Mean Euclidean Distance between Actions: 8.7007487e-07
        Max Euclidean Distance between Actions: 5.22235e-05
        Min Euclidean Distance between Actions: 0.0

        Mean Cosine Similarity between Actions: 1.0
        Max Cosine Similarity between Actions: 1.0000002
        Min Cosine Similarity between Actions: 0.99999774

        ##########################
        Visualizing Actions of Greedy_00 Agent and Contrastive Agent
        ##########################
        Distances shape: (11, 11)
        Number of cells in base environment image: 326700
    """