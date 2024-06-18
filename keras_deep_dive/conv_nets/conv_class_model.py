from tensorflow import keras
from keras import layers

"""
simple convolutional model for image classification. 
With this comment my branch is ahead of origin by 2 commits

With augmentation I fight against overfitting, remixing the data 
to feed the model to. Like dropout layers, this augmentation layers are
inactive during model inference (prediction and evaluation).

The filters parameter defines the number of output filters (or feature maps) in the convolution.
This means it controls the number of different features that the convolutional layer will learn to detect.
"""

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2)
    ]
)

inputs = keras.Input(shape=(180, 180, 3))
x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x)
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Flatten()(x)
x = layers.Dropout()(x) # between conv and pooling layers this needs to be implemente differently

outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs=inputs, outputs=outputs)