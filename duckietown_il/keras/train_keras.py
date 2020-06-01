import sys
sys.path.append("../")
import numpy as np
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt
import argparse
from _loggers import Reader
from model_keras import VGG16_model, NVIDIA_model
from keras.optimizers import SGD, Adam
from keras.losses import mean_squared_error as MSE
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
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

# Split the data: Train and Test
x_train, x_test, y_train, y_test = train_test_split(observations, actions, test_size=0.2, random_state=2)
# Split Train data once more for Validation data
val_size = int(len(x_train) * 0.1)
x_validate, y_validate = x_train[:val_size], y_train[:val_size]
x_train, y_train       = x_train[val_size:], y_train[val_size:]

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(rescale=1./255,              # rescaling factor
                                   width_shift_range=0.2,       # float: fraction of total width, if < 1
                                   height_shift_range=0.2,      # float: fraction of total height, if < 1
                                   brightness_range=None,       # Range for picking a brightness shift value from
                                   zoom_range=0.0,              # Float or [lower, upper]. Range for random zoom
                                   )
train_datagen.fit(x_train)
# this is the augmentation configuration we will use for validating: only rescaling
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen.fit(x_validate)
# this is the augmentation configuration we will use for testing: only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)
test_datagen.fit(x_test)

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

# Create Keras callbacks
es = EarlyStopping(monitor='val_loss', verbose=1, patience=30)
mc = ModelCheckpoint(STORAGE_LOCATION + MODEL_NAME + '.h5', monitor='val_loss', save_best_only=True)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tb = TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                              validation_data=validation_datagen.flow(x_validate, y_validate, batch_size=BATCH_SIZE),
                              epochs=EPOCHS,
                              verbose=2,
                              steps_per_epoch=observations.shape[0] // BATCH_SIZE,
                              callbacks=[es, mc, tb],
                              shuffle=True)

# Plot & save the plots
plot_model_history(history, path_to_save=STORAGE_LOCATION, model_name=MODEL_NAME)
# Test the model on the test set
test_result = model.evaluate(test_datagen.flow(x_test, y_test, batch_size=BATCH_SIZE))
print(f"Test loss: {test_result[0]:.3f}\t | Test accuracy: %{test_result[1]:.2f}")
