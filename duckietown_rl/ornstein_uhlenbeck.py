"""
Based on: https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
"""

import numpy as np
import matplotlib.pyplot as plt


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
