from experiment_utils import inspect_params, load_encoders, inspect_network, get_env, sample_env_states, visualize_states, get_sa_encodings, get_g_encodings, reduce_dim, plot_encodings, critic_dist, get_action_colors, generate_env_states, get_steps_to_center
import matplotlib.pyplot as plt
import numpy as np

# generating environment image, results in np.ndarray images
env = get_env("point_Spiral11x11")
center = np.array([5.5,5.5])
env_image_30 = visualize_states(env, states=[center],color_meaning="ids", save_path="experiment_results/appendix/env_images/env_image_30.png", state_colors=[[255,255,255]])

# Compute how often a sampled task is successful without any training
env = get_env("point_Spiral11x11")
samples = 10000
successes = 0
for i in range(samples):
    task = env.reset()
    start_state = env.state
    goal_state = env.goal
    # Calculate the distance between the goal and the current state (copied from env implementation)
    dist = np.linalg.norm(start_state - goal_state)
    # Does it fulfill success criteria (copied from env implementation)
    success = float(dist < 2.0)
    successes += success
print("Success rate without training: ", successes/samples)

if False:
    # sample tasks until sampling a task that already satisfies the success criterion
    # (goal distance of 2)
    env = get_env("point_Spiral11x11")
    save_dir = "experiment_results/appendix/env_images/"
    while True:
        task = env.reset()
        start_state = env.state
        goal_state = env.goal
        # Calculate the distance between the goal and the current state (copied from env implementation)
        dist = np.linalg.norm(start_state - goal_state)
        # Does it fulfill success criteria (copied from env implementation)
        success = float(dist < 2.0)
        if success:
            # visualize the task, start in blue and goal in red
            start_color = [0, 0, 255]
            goal_color = [255, 0, 0]
            visualize_states(env, states=[start_state, goal_state], color_meaning="ids", save_path=save_dir+"successful_task_separated_by_walls.png" , state_colors=[start_color, goal_color], radius=8)
            # load image and show it
            img = plt.imread(save_dir+"successful_task_separated_by_walls.png")
            plt.imshow(img)
            plt.axis('off')
            plt.show()
            # ask if image is satisfying
            print("Is the image showing a successful task where start and goal are separated by a wall? (y/n)")
            answer = input()
            if answer == "y":
                break
