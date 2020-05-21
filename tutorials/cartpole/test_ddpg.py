import numpy as np
from ddpg import DDPG
import gym

env = gym.make('CartPole-v0')

state_dim = env.observation_space.shape[0]
action_dim = 1
max_action = 1

# Initialize policy
policy = DDPG(state_dim, action_dim, max_action)
policy.load("model", directory="models")

EPISODES, STEPS = 1, 300
for e in range(EPISODES):
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

        if done or total_step == STEPS:
            break
    print("\nmean episode reward:", np.mean(rewards),
          "\t | mean episode reward:", np.sum(rewards),
          "\t | total_step", total_step)

env.close()
