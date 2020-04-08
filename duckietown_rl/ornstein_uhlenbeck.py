import numpy as np
import matplotlib.pyplot as plt


# TODO: use action_noise.reset in --> env.reset()
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta   # (float) the rate of mean reversion
        self.mu = mu         # (float) the mean of the noise
        self.sigma = sigma   # (float) the scale of the noise
        self.dt = dt         # (float) the timestep for the noise
        self.x0 = x0         # ([float]) the initial value for the noise output, (if None: 0)
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


def plot_graph(noise_right, noise_left, noise, timesteps, scatter=False):
    plt.figure(figsize=(20, 7))
    if scatter:
        plt.scatter(range(timesteps), noise_left, s=5)
        plt.scatter(range(timesteps), noise_right, s=5)
    else:
        plt.plot(range(timesteps), noise_left, color='green', linewidth=2, markersize=12, label="left")
        plt.plot(range(timesteps), noise_right, color='red', linewidth=2, markersize=12, label="right")

    plt.title(noise.__repr__())
    plt.xlabel("Steps")
    plt.ylabel("Noise")
    plt.legend()
    plt.tight_layout()
    # plt.savefig(f"{noise.__repr__()}-{timesteps} steps.png")
    plt.show()


# action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(2), sigma=np.ones(2)*0.2, theta=0.7)
# noise_list = []
# timesteps = 500
#
# for i in range(timesteps):
#     noise_list.append(action_noise())
#
# noise = np.array(noise_list)
# noise_left = noise[:, 0]
# noise_right = noise[:, 1]
#
# plot_graph(noise_right, noise_left, action_noise, timesteps, scatter=False)
