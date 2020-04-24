import random
import numpy as np
import torch

def seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# Code based on:
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

# Simple replay buffer
class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size

    # Expects tuples of (state, next_state, action, reward, done)
    def add(self, state, next_state, action, reward, done):
        if len(self.storage) < self.max_size:
            self.storage.append((state, next_state, action, reward, done))

        else:
            # Remove random element in the memory before adding a new one
            self.storage.pop(random.randrange(len(self.storage)))
            self.storage.append((state, next_state, action, reward, done))


    def sample(self, batch_size=100, flat=True):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        states, next_states, actions, rewards, dones = [], [], [], [], []

        for i in ind:
            state, next_state, action, reward, done = self.storage[i]

            if flat:
                # TODO: try,except if error occurs OR remove flatten
                states.append(np.array(state, copy=False).flatten())
                next_states.append(np.array(next_state, copy=False).flatten())

            else:
                states.append(np.array(state, copy=False))
                next_states.append(np.array(next_state, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(np.array(reward, copy=False))
            dones.append(np.array(done, copy=False))

        # state_sample, action_sample, next_state_sample, reward_sample, done_sample
        return {
            "state": np.stack(states),
            "next_state": np.stack(next_states),
            "action": np.stack(actions),
            "reward": np.stack(rewards).reshape(-1,1),
            "done": np.stack(dones).reshape(-1,1)
        }


def evaluate_policy(env, policy, eval_episodes=10, max_timesteps=500):
    avg_reward = 0.
    avg_time = 0.
    for _ in range(eval_episodes):
        env.reset()
        obs = env.get_features()  # @riza
        done = False
        step = 0
        while not done and step < max_timesteps:
            action = policy.predict(obs)
            obs, reward, done, _ = env.step(action)
            obs = env.get_features()  # @riza
            avg_reward += reward
            step += 1

        avg_time += env.get_agent_info()['Simulator']['timestamp']

    avg_reward /= eval_episodes
    avg_time /= eval_episodes

    return avg_reward, avg_time
