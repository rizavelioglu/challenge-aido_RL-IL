import sys
sys.path.append("../")
import numpy as np
from tqdm import tqdm
from _loggers import Reader
from model_keras import build_model
from keras.optimizers import SGD
from keras.losses import mean_squared_error as MSE
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse


# Function to plot model's validation loss and validation accuracy
def plot_model_history(model_history, path_to_save):
    fig, axs = plt.subplots(1, 2, figsize=(25, 8))
    # summarize history for accuracy
    axs[0].plot(range(1, len(model_history.history['accuracy']) + 1), model_history.history['accuracy'])
    axs[0].plot(range(1, len(model_history.history['val_accuracy']) + 1), model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(model_history.history['accuracy']) + 1),
                      len(model_history.history['accuracy']) / 10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1), len(model_history.history['loss']) / 10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.savefig(path_to_save + '/01_VGG16_model_history.png')

    plt.show()


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data", required=True, type=str, help="name of the data to learn from (without .log)")
args = vars(ap.parse_args())
DATA = args["data"]

# configuration zone
BATCH_SIZE = 32                        # define the batch size
EPOCHS = 50                            # how many times we iterate through our data
STORAGE_LOCATION = "trained_models/"   # where we store our trained models
reader = Reader(f'../{DATA}.log')      # where our data lies

observations, actions = reader.read()  # read the observations from data
actions = np.array(actions)
observations = np.array(observations)

# Split the data
x_train, x_validate, y_train, y_validate = train_test_split(observations, actions, test_size=0.1, random_state=2)

test_size = 10000
x_test,  y_test  = x_train[:test_size], y_train[:test_size]
x_train, y_train = x_train[test_size:], y_train[test_size:]

train_datagen = ImageDataGenerator()
train_datagen.fit(x_train)

validation_datagen = ImageDataGenerator()
validation_datagen.fit(x_validate)

# Build the model
model = build_model()
# Define the optimizer
optimizer = SGD(lr=0.01, momentum=0.001, nesterov=False)
# Compile the model
model.compile(optimizer=optimizer,
              loss=MSE,
              metrics=["accuracy"])

es = EarlyStopping(monitor='val_loss', verbose=1, patience=30)
mc = ModelCheckpoint(STORAGE_LOCATION + '01_VGG16.h5', monitor='val_loss', save_best_only=True)

history = model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                              validation_data=validation_datagen.flow(x_validate, y_validate, batch_size=BATCH_SIZE),
                              epochs=EPOCHS,
                              verbose=2,  # for hiding print statements
                              steps_per_epoch=observations.shape[0] // BATCH_SIZE,
                              callbacks=[es, mc],
                              shuffle=True)

# Plot & save the plots
plot_model_history(history, path_to_save=STORAGE_LOCATION)
# Evaluate the model on the test set
test_result = model.evaluate(x_test, y_test)
print(f"Test loss: {test_result[0]:.3f}\t | Test accuracy: %{test_result[1]:.2f}")
