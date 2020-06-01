import sys
sys.path.append("../../")
from duckietown_rl.gym_duckietown.simulator import Simulator
from keras.models import load_model
import cv2

env = Simulator(seed=123, map_name="zigzag_dists", max_steps=5000001, domain_rand=True, camera_width=640,
                camera_height=480, accept_start_angle_deg=4, full_transparency=True, distortion=True,
                randomize_maps_on_reset=False, draw_curve=False, draw_bbox=False, frame_skip=1, draw_DDPG_features=False)

model = load_model("trained_models/01_NVIDIA.h5")

observation = env.reset()
env.render()
cumulative_reward = 0.0
EPISODES = 10
STEPS = 1000

for episode in range(0, EPISODES):
    for steps in range(0, STEPS):
        # Cut the horizon: obs.shape = (480,640,3) --> (300,640,3)
        observation = observation[150:450, :]
        # we can resize the image here
        observation = cv2.resize(observation, (120, 60))
        # NOTICE: OpenCV changes the order of the channels !!!
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)

        action = model.predict(observation.reshape(1, 60, 120, 3))[0]
        observation, reward, done, info = env.step(action)
        cumulative_reward += reward
        if done:
            env.reset()

        print(f"Reward: {reward:.2f}",
              f"\t| Action: [{action[0]:.3f}, {action[1]:.3f}]",
              f"\t| Speed: {env.speed:.2f}")

        cv2.imshow("obs", observation)
        if cv2.waitKey() & 0xFF == ord('q'):
            break

        env.render()
    env.reset()

print('total reward: {}, mean reward: {}'.format(cumulative_reward, cumulative_reward // EPISODES))

env.close()
model.close()
