from keras.applications.vgg16 import VGG16
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model, load_model


def build_model():
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
