import sys
sys.path.append("../")
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from _loggers import Reader
from model_keras import VGG16_model, NVIDIA_model
from keras.optimizers import SGD, Adam
from keras.losses import mean_squared_error as MSE
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


# Function to plot model's validation loss and validation accuracy
def plot_model_history(model_history, path_to_save, model_name):
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
    plt.savefig(path_to_save + '/' + model_name + '_model_history.png')

    plt.show()


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data", required=True, type=str, help="name of the data to learn from (without .log)")
ap.add_argument("-e", "--epoch", required=True, type=int, help="number of epochs")
ap.add_argument("-b", "--batch-size", required=True, type=int, help="batch size")
args = vars(ap.parse_args())
DATA = args["data"]

# configuration zone
BATCH_SIZE = args["batch_size"]        # define the batch size
EPOCHS     = args["epoch"]             # how many times we iterate through our data
STORAGE_LOCATION = "trained_models/"   # where we store our trained models
reader = Reader(f'../{DATA}.log')      # where our data lies
MODEL_NAME = "01_NVIDIA"

observations, actions = reader.read()  # read the observations from data
actions = np.array(actions)
observations = np.array(observations)

# Split the data
x_train, x_validate, y_train, y_validate = train_test_split(observations, actions, test_size=0.2, random_state=2)

test_size = int(len(x_train) * 0.1)
x_test,  y_test  = x_train[:test_size], y_train[:test_size]
x_train, y_train = x_train[test_size:], y_train[test_size:]

train_datagen = ImageDataGenerator()
train_datagen.fit(x_train)

validation_datagen = ImageDataGenerator()
validation_datagen.fit(x_validate)

# Build the model
# model = VGG16_model()
model = NVIDIA_model()
# Define the optimizer
# optimizer = SGD(lr=0.01, momentum=0.001, nesterov=False)
optimizer = Adam(lr=1e-3, decay=1e-3/EPOCHS)
# Compile the model
model.compile(optimizer=optimizer,
              loss=MSE,
              metrics=["accuracy"])

es = EarlyStopping(monitor='val_loss', verbose=1, patience=30)
mc = ModelCheckpoint(STORAGE_LOCATION + MODEL_NAME + '.h5', monitor='val_loss', save_best_only=True)

history = model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                              validation_data=validation_datagen.flow(x_validate, y_validate, batch_size=BATCH_SIZE),
                              epochs=EPOCHS,
                              verbose=2,  # for hiding print statements
                              steps_per_epoch=observations.shape[0] // BATCH_SIZE,
                              callbacks=[es, mc],
                              shuffle=True)

# Plot & save the plots
plot_model_history(history, path_to_save=STORAGE_LOCATION, model_name=MODEL_NAME)
# Evaluate the model on the test set
test_result = model.evaluate(x_test, y_test)
print(f"Test loss: {test_result[0]:.3f}\t | Test accuracy: %{test_result[1]:.2f}")
