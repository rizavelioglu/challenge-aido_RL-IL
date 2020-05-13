import numpy as np
from ddpg import DDPG
import gym
import torch

env = gym.make('CartPole-v0')

state_dim = env.observation_space.shape[0]
action_dim = 1
max_action = 1

# Initialize policy
policy = DDPG(state_dim, action_dim, max_action)
policy.load("model", directory="models")

with torch.no_grad():
    while True:
        obs = env.reset()
        env.render()
        rewards = []
        total_step = 0
        while True:
            action = policy.predict(np.array(obs))
            action = 1 if action > 0.5 else 0
            obs, rew, done, misc = env.step(action)
            rewards.append(rew)
            env.render(mode='rgb_array')
            total_step += 1

            if done:
                break
        print("\nmean episode reward:", np.mean(rewards),
              "\t | mean episode reward:", np.sum(rewards),
              "\t | total_step", total_step)
