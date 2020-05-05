import numpy as np
from tqdm import tqdm
from _loggers import Reader
from model_tf import TensorflowModel
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

# configuration zone
BATCH_SIZE = 32                           # define the batch size
EPOCHS = 10                               # how many times we iterate through our data
OBSERVATIONS_SHAPE = (None, 60, 80, 3)    # here we assume the observations have been resized to 60x80
ACTIONS_SHAPE = (None, 2)                 # actions have a shape of 2: [leftWheelVelocity, rightWheelVelocity]
SEED = 1234
STORAGE_LOCATION = "trained_models/"      # where we store our trained models
reader = Reader('train.log')              # where our data lies

observations, actions = reader.read()     # read the observations from data
actions = np.array(actions)
observations = np.array(observations)

model = TensorflowModel(
    observation_shape=OBSERVATIONS_SHAPE,  # from the logs we've got
    action_shape=ACTIONS_SHAPE,            # same
    graph_location=STORAGE_LOCATION,       # where do we want to store our trained models
    seed=SEED)                             # to seed all random operations in the model (e.g., dropout)

prev_loss = 10
# we trained for EPOCHS epochs
epochs_bar = tqdm(range(EPOCHS))
for i in epochs_bar:
    loss = None
    for batch in range(0, len(observations), BATCH_SIZE):
        loss = model.train(
                            observations=observations[batch:batch + BATCH_SIZE],
                            actions=actions[batch:batch + BATCH_SIZE])

    epochs_bar.set_postfix({'loss': loss})
    # Save the model if the loss is decreased
    if loss < prev_loss:
        model.commit()
        epochs_bar.set_description('Model saved...')
        prev_loss = loss

    else:
        epochs_bar.set_description('Not saved...')

model.close()
reader.close()
