from experiment_utils import inspect_params, load_encoders, inspect_network, get_env, sample_env_states, visualize_states, get_sa_encodings, get_g_encodings, reduce_dim, plot_encodings, critic_dist, get_action_colors, generate_env_states, get_steps_to_center
import matplotlib.pyplot as plt

# generating environment image, results in np.ndarray images
env = get_env("point_Spiral11x11")
env_image_30 = env._get_env_img(color=False, scale=30)
env_image_5 = env._get_env_img(color=False, scale=5)

# save images (images are np.ndarray with grayscale values)
directory_path = "experiment_results/appendix/env_images/"
plt.imsave(directory_path + "env_image_30.png", env_image_30, cmap='gray')
plt.imsave(directory_path + "env_image_5.png", env_image_5, cmap='gray')

# TODO: investigate goal condition: goal distance of 2 -> can goal be reached through walls?
