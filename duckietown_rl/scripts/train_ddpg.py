import pandas as pd
import numpy as np
import torch
import os

from args import get_ddpg_args_train
from ddpg import DDPG
from utils import seed, evaluate_policy, ReplayBuffer
from wrappers import NormalizeWrapper, ImgWrapper, \
    DtRewardWrapper, ActionWrapper, ResizeWrapper, SteeringToWheelVelWrapper
from env import launch_env
from ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise

policy_name = "DDPG"
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

# Launch environment
env = launch_env()
# Set seeds
seed(args.seed)

state_dim = env.get_features().shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# Initialize policy
policy = DDPG(state_dim, action_dim, max_action, net_type="dense")
replay_buffer = ReplayBuffer(args.replay_buffer_max_size)

# Evaluate untrained policy
rew_eval, time_eval = evaluate_policy(env, policy)

total_timesteps = 0
timesteps_since_eval = 0
episode_num = 0
done = True
episode_reward = None
env_counter = 0

# Crate our variables for logging
# while evaluating the exploration value = 0
evaluations_test = []
evaluations_eval = []

evaluations_test.append([episode_num, total_timesteps, rew_eval, 0])
# Create OrnsteinUhlenbeckActionNoise instance
action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(2), sigma=np.ones(2)*0.2, theta=0.7)

while total_timesteps < args.max_timesteps:

    if done:

        if total_timesteps != 0:
            print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (
                total_timesteps, episode_num, episode_timesteps, episode_reward))
            policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau)
            evaluations_test.append([episode_num, total_timesteps, episode_reward, episode_timesteps])

        # Evaluate episode
        if timesteps_since_eval >= args.eval_freq:
            timesteps_since_eval %= args.eval_freq
            rew_eval, time_eval = evaluate_policy(env, policy)
            # Append logs
            evaluations_eval.append([total_timesteps, rew_eval, time_eval])

            if args.save_models:
                policy.save("{}-episode:{}-reward:{}".format(file_name, episode_num, rew_eval), directory="./pytorch_models")

        # Reset environment
        env_counter += 1
        env.reset()
        obs = env.get_features()
        done = False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1
        # Reset noise profile
        action_noise.reset()

    # Select action randomly or according to policy
    if total_timesteps < args.start_timesteps:
        action = abs(env.action_space.sample())
    else:
        action = policy.predict(obs)
        # Add OU action noise with its effect decreasing over time
        action = action + action_noise() * (1 - total_timesteps / args.max_timesteps) * [1 if episode_num % 2 == 0 else 0.5][0]

    # Perform action
    _, reward, done, _ = env.step(action)
    new_obs = env.get_features()

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
rew_eval, time_eval = evaluate_policy(env, policy)
evaluations_eval.append([total_timesteps, rew_eval, time_eval])

if args.save_models:
    policy.save("{}-episode_reward:{}".format(file_name, episode_reward), directory="./pytorch_models")

"""
total_step: Global step
avg_reward: Average reward 
"""
df_eval = pd.DataFrame(evaluations_eval, columns=["total_step", "avg_reward", "avg_time"])
df_eval.to_csv("./results/df_eval.csv")

"""
n_episode : Episode Id
total_step: Global step
e_reward    : Episode reward
n_step    : #of steps done per episode
"""
df_test = pd.DataFrame(evaluations_test, columns=["n_episode", "total_step", "e_reward", "n_step"])
df_test.to_csv("./results/df_test.csv")
