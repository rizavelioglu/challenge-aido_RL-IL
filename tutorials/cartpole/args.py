import argparse


def get_ddpg_args_train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e2, type=int)  # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=1e3, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=9e3, type=float)  # Max time steps to run environment for
    parser.add_argument("--save_models", action="store_true", default=True)  # Whether or not models are saved
    parser.add_argument("--expl_noise", default=0.2, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=32, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--env_timesteps", default=200, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--replay_buffer_max_size", default=1000, type=int)  # Maximum number of steps to keep in the replay buffer

    return parser.parse_args()
