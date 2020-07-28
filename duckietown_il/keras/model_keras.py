"""
This script implements various types of neural network models: e.g. VGG16.
"""
from keras.applications.vgg16 import VGG16
from keras.layers import GlobalAveragePooling2D, Dense, Flatten, Lambda, MaxPooling2D, Conv2D, Lambda, Dropout
from keras.models import Model, load_model, Sequential


def VGG16_model():
    base_model = VGG16(classes=2, input_shape=(60, 80, 3), weights=None, include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(2, activation="sigmoid")(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    return model


def NVIDIA_model():
    # Source:  https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    model = Sequential()
    # Conv_1
    model.add(Conv2D(24, (5, 5), activation="relu", padding="same", strides=(2, 2), input_shape=(60, 120, 3)))
    # Conv_2
    model.add(Conv2D(36, (5, 5), activation="relu", padding="same", strides=(2, 2)))
    # Conv_3
    model.add(Conv2D(48, (5, 5), activation="relu", padding="same", strides=(2, 2)))
    # Conv_4
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same", strides=(1, 1)))
    # Conv_5
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same", strides=(1, 1)))
    # Pool_1
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Flatten())
    # Next, five fully connected layers
    model.add(Dense(1164, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(2, activation="sigmoid"))

    return model
