from model import TensorflowModel
from env import launch_env

# configuration zone
# yes, remember the simulator give us an outrageously large image
# we preprocessed in the logs, but here we rely on the preprocessing step in the model
OBSERVATIONS_SHAPE = (None, 480, 640, 3)
ACTIONS_SHAPE = (None, 2)
SEED = 1234
STORAGE_LOCATION = "trained_models/behavioral_cloning"
EPISODES = 10
STEPS = 65

env = launch_env()

model = TensorflowModel(
    observation_shape=OBSERVATIONS_SHAPE,  # from the logs we've got
    action_shape=ACTIONS_SHAPE,  # same
    graph_location=STORAGE_LOCATION,  # where do we want to store our trained models
    seed=SEED  # to seed all random operations in the model (e.g., dropout)
)

observation = env.reset()

# we can use the gym reward to get an idea of the performance of our model
cumulative_reward = 0.0

for episode in range(0, EPISODES):
    for steps in range(0, STEPS):
        action = model.predict(observation)
        observation, reward, done, info = env.step(action)
        cumulative_reward += reward
        if done:
            env.reset()
        # env.render()
    # we reset after each episode, or not, this really depends on you
    env.reset()

print('total reward: {}, mean reward: {}'.format(cumulative_reward, cumulative_reward // EPISODES))
# didn't look good, ah? Well, that's where you come into the stage... good luck!

env.close()
model.close()