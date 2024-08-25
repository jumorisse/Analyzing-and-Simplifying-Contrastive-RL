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
    legend_elements = []
    for i in range(len(agent_names)):
        legend_elements.append(patches.Patch(color=np.array(colors[i])/255, label=agent_names[i]))
    plt.legend(handles=legend_elements, loc="lower left")

    # turn off axis ticks, but keep axis
    plt.axis("on")
    plt.xticks([])
    plt.yticks([])

    plt.savefig(save_path)

if __name__ == "__main__":
    environment_name = "point_Spiral11x11"
    n_tasks = 500 # number of tasks to sample, for reproduction should be 500
    max_episode_length = 100 # maximum number of steps in an episode, for reproduction should be 100
    seeds  = [0, 21, 42, 97, 1453]
    params_paths = ["manual_checkpoints/two_encoders/point_Spiral11x11/original_seed"+str(seed) for seed in seeds]
    numpy_seed = 42 # seed for numpy, for reproduction should be 42
    action_grid_scale = 2 # scaling factor for the action grid from which greedy actor chooses, for reproduction should be 2
    np.random.seed(numpy_seed) # set seed for numpy
    contrastive_eval_mode = False # whether to evaluate the contrastive agent in evaluation mode (using the mean of the distribution) or not

    evaluating_performances = True
    computing_action_distances = True
    visualizing_actions = True

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
            greedy_action = greedy_agents_00[0].select_action(task)
            contrastive_action = contrastive_agents[0].select_action(task)
            # store actions
            greedy_actions.append(greedy_action)
            contrastive_actions.append(contrastive_action)
        # Visualize actions
        visualize_actions([greedy_actions, contrastive_actions], ["Greedy Agent", "Parameterized Agent"], states, goal, env, [[255,0,0], [0,0,255]], save_path)


    """
    Final Results for Scaling factor 2, Seed 42 (any reward, non-sigmoid critic, 500 tasks)
        Greedy_00 is evaluated across 5 seeds.
        Greedy_00 success rates mean per seed: [0.71  0.71  0.698 0.694 0.638]
        Greedy_00 success rate mean across seeds: 0.69
        Greedy_00 success rate std across seeds: 0.0267731208490904
        Greedy_00 average steps to reach goal first time: 7.50346565847511
        Greedy_01 is evaluated across 5 seeds.
        Greedy_01 success rates mean per seed: [0.74  0.744 0.726 0.712 0.654]
        Greedy_01 success rate mean across seeds: 0.7152
        Greedy_01 success rate std across seeds: 0.032609201155502095
        Greedy_01 average steps to reach goal first time: 8.306184935701163
        Greedy_05 is evaluated across 5 seeds.
        Greedy_05 success rates mean per seed: [0.768 0.798 0.77  0.77  0.738]
        Greedy_05 success rate mean across seeds: 0.7688
        Greedy_05 success rate std across seeds: 0.018998947339260684
        Greedy_05 average steps to reach goal first time: 9.986374407582938
        Greedy_10 is evaluated across 5 seeds.
        Greedy_10 success rates mean per seed: [0.796 0.822 0.798 0.792 0.768]
        Greedy_10 success rate mean across seeds: 0.7952
        Greedy_10 success rate std across seeds: 0.01718604084715265
        Greedy_10 average steps to reach goal first time: 10.688563049853373
        Random is evaluated across 5 seeds.
        Random success rates mean per seed: [0.442 0.434 0.46  0.438 0.43 ]
        Random success rate mean across seeds: 0.4408
        Random success rate std across seeds: 0.010400000000000008
        Random average steps to reach goal first time: 27.689243027888445
        Contrastive is evaluated across 5 seeds.
        Contrastive success rates mean per seed: [0.774 0.8   0.834 0.804 0.794]
        Contrastive success rate mean across seeds: 0.8012
        Contrastive success rate std across seeds: 0.019374209661299713
        Contrastive average steps to reach goal first time: 6.173820879703233
        ##########################
        Comparing Actions
        ##########################
        Seed 0
        Seed 21
        Seed 42
        Seed 97
        Seed 1453
        Comparing Actions of ContrastiveAgent (leading) and GreedyAgent (trailing) on 500 tasks with 100 max episode steps.
        ###################################
        ### Comparison Across All Tasks ###
        ###################################
        Mean Euclidean Distance between Actions: 0.83827764
        Std of Mean Euclidean Distance between Actions: 0.07118403
        Max Euclidean Distance between Actions: 2.828427
        Min Euclidean Distance between Actions: 0.0

        Mean Cosine Similarity between Actions: 0.38308399886003125
        Std of Mean Cosine Similarity between Actions: 0.051240259689522595
        Max Cosine Similarity between Actions: 1.000000238418579
        Min Cosine Similarity between Actions: -1.0000001192092896

        Comparing Actions of ContrastiveAgent (leading) and RandomAgent (trailing) on 500 tasks with 100 max episode steps.
        ###################################
        ### Comparison Across All Tasks ###
        ###################################
        Mean Euclidean Distance between Actions: 1.2493564
        Std of Mean Euclidean Distance between Actions: 0.030379262
        Max Euclidean Distance between Actions: 2.828427
        Min Euclidean Distance between Actions: 0.0

        Mean Cosine Similarity between Actions: 0.0014817021151853817
        Std of Mean Cosine Similarity between Actions: 0.0026786156459569234
        Max Cosine Similarity between Actions: 1.0000001192092896
        Min Cosine Similarity between Actions: -1.0000001192092896

        Comparing Actions of ContrastiveAgent (leading) and ContrastiveAgent (trailing) on 500 tasks with 100 max episode steps.
        ###################################
        ### Comparison Across All Tasks ###
        ###################################
        Mean Euclidean Distance between Actions: 1.0331364e-06
        Std of Mean Euclidean Distance between Actions: 8.8689966e-08
        Max Euclidean Distance between Actions: 0.00036140194
        Min Euclidean Distance between Actions: 0.0

        Mean Cosine Similarity between Actions: 1.0
        Std of Mean Cosine Similarity between Actions: 0.0
        Max Cosine Similarity between Actions: 1.0000002
        Min Cosine Similarity between Actions: 0.99996716
    """