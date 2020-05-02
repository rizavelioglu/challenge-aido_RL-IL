import numpy as np
from tqdm import tqdm

from _loggers import Reader
from model import TensorflowModel

# configuration zone
BATCH_SIZE = 32
EPOCHS = 10
# here we assume the observations have been resized to 60x80
OBSERVATIONS_SHAPE = (None, 60, 80, 3)
ACTIONS_SHAPE = (None, 2)
SEED = 1234
STORAGE_LOCATION = "trained_models/"

reader = Reader('train.log')

observations, actions = reader.read()
actions = np.array(actions)
observations = np.array(observations)

model = TensorflowModel(
    observation_shape=OBSERVATIONS_SHAPE,  # from the logs we've got
    action_shape=ACTIONS_SHAPE,  # same
    graph_location=STORAGE_LOCATION,  # where do we want to store our trained models
    seed=SEED  # to seed all random operations in the model (e.g., dropout)
)

# we trained for EPOCHS epochs
epochs_bar = tqdm(range(EPOCHS))
for i in epochs_bar:
    loss = None
    for batch in range(0, len(observations), BATCH_SIZE):
        loss = model.train(
            observations=observations[batch:batch + BATCH_SIZE],
            actions=actions[batch:batch + BATCH_SIZE]
        )

    epochs_bar.set_postfix({'loss': loss})

    if i % 10 == 0:
        model.commit()
        epochs_bar.set_description('Model saved...')
    else:
        epochs_bar.set_description('')

# the loss at this point should be on the order of 2e-2, which is far for great, right?

# we release the resources...
model.close()
reader.close()

