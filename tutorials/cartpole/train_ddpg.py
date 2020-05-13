from args import get_ddpg_args_train
from ddpg import DDPG
from utils import seed, evaluate_policy, ReplayBuffer
import pandas as pd
import numpy as np
import torch
import gym
import os

policy_name = "cartpole-DDPG"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = get_ddpg_args_train()

file_name = "{}_{}".format(
    policy_name,
    str(args.seed))

# Create folders within the same folder
if not os.path.exists("./results"):
    os.makedirs("./results")
if args.save_models and not os.path.exists("./pytorch_models"):
    os.makedirs("./pytorch_models")

env = gym.make('CartPole-v0')

# Set seeds
seed(args.seed)

state_dim = env.observation_space.shape[0]
action_dim = 1
max_action = 1

# Initialize policy
policy = DDPG(state_dim, action_dim, max_action)
replay_buffer = ReplayBuffer(args.replay_buffer_max_size)

# Evaluate untrained policy
evaluations = [evaluate_policy(env, policy)]

total_timesteps = 0
timesteps_since_eval = 0
episode_num = 0
done = True
episode_reward = None
env_counter = 0
# We will store out log here
data_eval = []*7
while total_timesteps < args.max_timesteps:

    if done:

        if total_timesteps != 0:
            print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (
                total_timesteps, episode_num, episode_timesteps, episode_reward))
            policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau)

        # Evaluate episode
        if timesteps_since_eval >= args.eval_freq:
            timesteps_since_eval %= args.eval_freq
            evaluations.append(evaluate_policy(env, policy))

            if args.save_models:
                policy.save("{}-episode_reward:{}".format(file_name, episode_reward), directory="./pytorch_models")
            data_eval.append([episode_num, total_timesteps, episode_reward, episode_timesteps])

        # Reset environment
        env_counter += 1
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

    # Select action randomly or according to policy
    if total_timesteps < args.start_timesteps:
        action = env.action_space.sample()
    else:
        action = policy.predict(obs)
        if args.expl_noise != 0:
            action = (action + np.random.normal(0, args.expl_noise, size=1)).clip(0, 1)

    # Perform action
    action = 1 if action > 0.5 else 0
    new_obs, reward, done, _ = env.step(action)

    if episode_timesteps >= args.env_timesteps:
        done = True

    done_bool = 0 if episode_timesteps + 1 == args.env_timesteps else float(done)
    episode_reward += reward

    # Store data in replay buffer
    replay_buffer.add(obs, new_obs, action, reward, done_bool)

    obs = new_obs

    episode_timesteps += 1
    total_timesteps += 1
    timesteps_since_eval += 1

# Final evaluation
evaluations.append(evaluate_policy(env, policy))
data_eval.append([episode_num, total_timesteps, episode_reward, episode_timesteps])

if args.save_models:
    policy.save("{}-episode_reward:{}".format(file_name, episode_reward), directory="./pytorch_models")
df_eval = pd.DataFrame(data_eval, columns=["n_episode", "total_step", "reward", "n_step"])
df_eval.to_csv("./results/df_eval.csv")
