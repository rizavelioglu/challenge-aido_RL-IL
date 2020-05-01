"""
"""
# We need to append the following path to system path, because 'ornstein_uhlenbeck.py' lies in a different folder
# See these for more info
# https://www.devdungeon.com/content/python-import-syspath-and-pythonpath-tutorial
# https://leemendelowitz.github.io/blog/how-does-python-find-packages.html
import sys
sys.path.append("../")
import numpy as np
import matplotlib.pyplot as plt
from duckietown_rl.ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise


def plot_graph(noise_right, noise_left, noise, timesteps, scatter=False, save_img=True):
    plt.figure(figsize=(20, 7))
    if scatter:
        plt.scatter(range(timesteps), noise_left, s=6, color='blue', label="left")
        plt.scatter(range(timesteps), noise_right, s=6, color='orange', label="right")
    else:
        plt.plot(range(timesteps), noise_left, color='green', linewidth=2, markersize=12, label="left")
        plt.plot(range(timesteps), noise_right, color='red', linewidth=2, markersize=12, label="right")

    plt.title(noise.__repr__())
    plt.xlabel("Steps")
    plt.ylabel("Noise")
    plt.legend()
    plt.tight_layout()
    if save_img:
        plt.savefig(f"images/{noise.__repr__()}-{timesteps} steps.png")
    plt.show()


# Create the noise profile with it's parameters as follows
action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(2), sigma=np.ones(2)*0.2, theta=0.7)
# Initialize a list where all the actions are stored
noise_list = []
# Specify how many time-steps the noise should continue
timesteps = 1000
reset_after = 500

for i in range(timesteps):
    # Append the noise to the list
    noise_list.append(action_noise())
    # Reset the noise profile after some time-steps
    if i == reset_after:
        action_noise.reset()

# Convert the list to np.array for enabling slicing operation
noise = np.array(noise_list)
noise_left = noise[:, 0]
noise_right = noise[:, 1]
# Plot the noise profile
plot_graph(noise_right, noise_left, action_noise, timesteps, scatter=True, save_img=True)
