import sys
sys.path.append("../")
import numpy as np
import matplotlib.pyplot as plt
from duckietown_rl.ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--mu",          required=True, type=float, help="mean of the noise profile")
ap.add_argument("-s", "--sigma",       required=True, type=float, help="standard deviation of the noise profile")
ap.add_argument("-t", "--timesteps",   required=True, type=int, help="how many time-steps the noise should continue")
ap.add_argument("-r", "--reset-after", required=True, type=int, help="after how many time-steps noise should be resetted")
ap.add_argument("-i", "--save-img",    required=False, type=int, help="flag for saving the plot, set to 1 if yes")
ap.add_argument("-l", "--line",        required=False, type=int, help="flag for plotting line chart(scatter by default")

args = vars(ap.parse_args())
# Assign corresponding variables
mu = args["mu"]
sigma = args["sigma"]
timesteps = args["timesteps"]
reset_after = args["reset_after"]
save_img = args["save_img"]
line = args["line"]


# A function for plotting the noise profile
def plot_graph(noise_right, noise_left, noise, timesteps, line=False, save_img=False):
    plt.figure(figsize=(20, 7))
    if line:
        plt.plot(range(timesteps), noise_left, color='green', linewidth=2, markersize=12, label="left")
        plt.plot(range(timesteps), noise_right, color='red', linewidth=2, markersize=12, label="right")
    else:
        plt.scatter(range(timesteps), noise_left, s=6, color='blue', label="left")
        plt.scatter(range(timesteps), noise_right, s=6, color='orange', label="right")

    plt.title(f"{noise.__repr__()} - resets after {reset_after} steps")
    plt.xlabel("Steps")
    plt.ylabel("Noise")
    plt.legend()
    plt.tight_layout()
    if save_img:
        plt.savefig(f"images/{noise.__repr__()}-{timesteps} steps - resets after {reset_after} steps.png")
    plt.show()


# Create the noise profile with it's parameters as follows
action_noise = OrnsteinUhlenbeckActionNoise(mu=np.ones(2)*mu, sigma=np.ones(2)*sigma, theta=0.7)
# Initialize a list where all the actions are stored
noise_list = []

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
plot_graph(noise_right, noise_left, action_noise, timesteps, line=line, save_img=save_img)
