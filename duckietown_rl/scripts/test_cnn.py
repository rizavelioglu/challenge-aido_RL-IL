import torch
from duckietown_rl.ddpg import DDPG
from duckietown_rl.args import get_ddpg_args_test
from utils import evaluate_policy
from duckietown_rl.wrappers import NormalizeWrapper, ImgWrapper, \
    DtRewardWrapper, ActionWrapper, ResizeWrapper
from duckietown_rl.env import launch_env
import numpy as np

policy_name = "DDPG"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = get_ddpg_args_test()

file_name = "{}_{}".format(
    policy_name,
    args.seed
)

env = launch_env()

# Wrappers
# env = ResizeWrapper(env)
# env = NormalizeWrapper(env)
# env = ImgWrapper(env) # to make the images from 160x120x3 into 3x160x120
env = ActionWrapper(env)
# env = DtRewardWrapper(env) # not during testing

state_dim = env.get_features().shape[0]    # @riza: state_dim = env.observation_space.shape
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# Initialize policy
policy = DDPG(state_dim, action_dim, max_action, net_type="dense")
policy.load(file_name, directory="./pytorch_models")

with torch.no_grad():
    while True:
        env.reset()        # obs = env.reset()
        obs = env.get_features()  # @riza
        env.render()
        rewards = []
        while True:
            action = policy.predict(np.array(obs))
            obs, rew, done, misc = env.step(action)
            obs = env.get_features()  # @riza
            rewards.append(rew)
            env.render()
            if done:
                break
        print("mean episode reward:", np.mean(rewards))
