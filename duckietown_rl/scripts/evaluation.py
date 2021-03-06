import os
import numpy as np
from ddpg import DDPG
from gym_duckietown.simulator import Simulator
import torch
from statistics import median
import matplotlib.pyplot as plt

env = Simulator(seed=123, map_name="zigzag_dists", max_steps=5000001, domain_rand=True, camera_width=640,
                camera_height=480, accept_start_angle_deg=4, full_transparency=True, distortion=True,
                randomize_maps_on_reset=False, draw_curve=False, draw_bbox=True, frame_skip=4, evaluate=True,
                draw_DDPG_features=True)

# Create folders within the same folder
if not os.path.exists("scripts/evaluation_results"):
    os.makedirs("scripts/evaluation_results")

state_dim = env.get_features().shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# Initialize policy
policy = DDPG(state_dim, action_dim, max_action, net_type="dense")
policy.load("model", directory="./models", for_inference=True)

env.reset()
obs = env.get_features()
env.render()
EPISODES, STEPS = 5, 200  # in simulation: (STEPS*frame_skip) time-steps
log = {}
for i in range(EPISODES):
    log["episode#" + str(i)] = {}

with torch.no_grad():
    for episode in range(0, EPISODES):
        rewards = []
        dists = []
        for steps in range(0, STEPS):
            action = policy.predict(np.array(obs))
            _, rew, done, misc = env.step(action)
            obs = env.get_features()
            rewards.append(rew)
            dists.append(env.delta_time * env.speed * env.frame_skip)
            env.render()

            if done:
                break

        log["episode#" + str(episode)]["rewards"] = rewards
        log["episode#" + str(episode)]["dists"] = dists

        if env.env_count + 1 == 7:
            break

        env.reset()

# TODO: remove abs for reward plot
SOLUTION_BY = "Approach#1"
PATH_TO_SAVE = "scripts/evaluation_results/"
# Calculate median of the episode rewards
median_reward = []
for step in range(STEPS):
    median_list = []
    for ep in range(EPISODES):
        median_list.append(log["episode#" + str(ep)]["rewards"][step])
    median_reward.append(median(median_list))

plt.figure(1, figsize=(35, 20))
plt.subplot(2, 1, 1)
for i in range(EPISODES):
    plt.plot(range(STEPS), log["episode#" + str(i)]["rewards"], "--", label="episode#" + str(i))
plt.plot(range(STEPS), median_reward, '-', label="median")
plt.title(f"[{SOLUTION_BY}]Evaluation on 'zigzag_dists' map for {EPISODES} episodes, {STEPS} timesteps each with {env.frame_rate}FPS & frame_skip={env.frame_skip}", {'fontsize': 15, 'color' : 'red'})
plt.xlabel("Timesteps")
plt.xticks(list(range(STEPS)), np.arange(0, env.frame_skip*STEPS, env.frame_skip), rotation=90)
plt.ylabel("Reward")
plt.legend(loc="best")

plt.subplot(2, 1, 2)
for i in range(EPISODES):
    plt.plot(range(STEPS), np.cumsum(np.abs(log["episode#" + str(i)]["rewards"])), "--", label="episode#" + str(i))
plt.plot(range(STEPS), np.cumsum(np.abs(median_reward)), '-', label="median")
plt.title(f"Cumulative sum of the abs(reward) \nsum(median)={np.sum(median_reward):.2f}", {'fontsize': 15, 'color' : 'red'})
plt.xlabel("Timesteps")
plt.ylabel("Reward")
plt.xticks(list(range(STEPS)), np.arange(0, env.frame_skip*STEPS, env.frame_skip), rotation=90)
plt.legend(loc="best")
plt.savefig(PATH_TO_SAVE + SOLUTION_BY + "_plot_reward.png")
plt.show()


# Calculate median of the episode traveled distances
median_dists = []
for step in range(STEPS):
    dists_list = []
    for ep in range(EPISODES):
        dists_list.append(log["episode#" + str(ep)]["dists"][step])
    median_dists.append(median(dists_list))

plt.figure(2, figsize=(35, 20))
plt.subplot(2, 1, 1)
for i in range(EPISODES):
    plt.plot(range(STEPS), log["episode#" + str(i)]["dists"], "--", label="episode#" + str(i))
plt.plot(range(STEPS), median_dists, '-', label="median")
plt.title(f"[{SOLUTION_BY}]Evaluation on 'zigzag_dists' map for {EPISODES} episodes, {STEPS} timesteps each with {env.frame_rate}FPS & frame_skip={env.frame_skip}", {'fontsize': 15, 'color' : 'red'})
plt.xlabel("Timesteps")
plt.ylabel("Distance in meters")
plt.xticks(list(range(STEPS)), np.arange(0, env.frame_skip*STEPS, env.frame_skip), rotation=90)
plt.legend(loc="best")

plt.subplot(2, 1, 2)
for i in range(EPISODES):
    plt.plot(range(STEPS), np.cumsum(log["episode#" + str(i)]["dists"]), "--", label="episode#" + str(i))
plt.plot(range(STEPS), np.cumsum(median_dists), '-', label="median")
plt.title(f"Cumulative sum of the distance traveled \nsum(median)={np.sum(median_dists):.2f}", {'fontsize': 15, 'color' : 'red'})
plt.xlabel("Timesteps")
plt.ylabel("Distance in meters")
plt.xticks(list(range(STEPS)), np.arange(0, env.frame_skip*STEPS, env.frame_skip), rotation=90)
plt.legend(loc="best")
plt.savefig(PATH_TO_SAVE + SOLUTION_BY + "_plot_dist.png")
plt.show()


plt.figure(2, figsize=(35, 10))
for i in range(EPISODES):
    plt.plot(np.cumsum(log["episode#" + str(i)]["dists"]), range(STEPS), "--", label="episode#" + str(i))
plt.plot(np.cumsum(median_dists), range(STEPS), '-', label="median")
plt.title(f"[{SOLUTION_BY}]Evaluation on 'zigzag_dists' map for {EPISODES} episodes, {STEPS} timesteps each with {env.frame_rate}FPS & frame_skip={env.frame_skip}", {'fontsize': 15, 'color' : 'red'})
plt.xlabel("Distance in meters")
plt.ylabel("Timesteps")
plt.yticks(list(range(STEPS)), np.arange(0, env.frame_skip*STEPS, env.frame_skip))
plt.legend(loc="best")
plt.savefig(PATH_TO_SAVE + SOLUTION_BY + "_plot_DistvsTime.png")
plt.show()
