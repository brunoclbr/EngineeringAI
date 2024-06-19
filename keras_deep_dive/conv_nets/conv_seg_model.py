from tensorflow import keras 
from keras import layers

"""
To increment the number of features as you go deeper into your model means to catch
more specific charachteristics of the images (features of features)

Feature Map Learning in Downsampling:

    During downsampling, the network learns to extract and condense important features from the input image. 
    Each layer learns different levels of abstraction. Early layers might capture edges and textures, while deeper layers
    capture more complex structures and context.These learned feature maps, although reduced in spatial resolution,
      contain rich, high-level information about the image content.

Utilizing Downsampled Features in Upsampling:

    In the upsampling path, the network aims to use the high-level features learned during downsampling to accurately reconstruct 
    the detailed structure of the input image. To help with this, architectures like U-Net employ skip connections that directly 
    connect corresponding layers of the encoder and decoder. These skip connections concatenate or add feature maps from the downsampling 
    path to the upsampling path, allowing the network to combine detailed spatial information from earlier layers with the high-level features learned later.

    stride: distance between two successive windows. Without striding the patches a lot of tiles
    repeat themselves when going from patch to patch

    padding: output feature map with the same spatial dimensions as the input (add tiles to keep dimension)

    architecture best practices: 
        residual connections - avoids vanishing gradient problem by using residual connections -
        batch normalization - normalizing data even if mean and variance change over time "no one really knows why batch normalization helps" -
                            - dont do batch normalization while fine tuning!
        depthwise separable convolutions - layer that does a spatial convolution on each channel of its input
                                            independently, before mixing output channels via pointwise convolution

        The number of filters should increase with the depth of the layers, as the size of the feature maps decreases

        Xception-like model on page 260 using best practices

        Also filter/features visualization techniques
""" 
  
def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))
    x = layers.Rescaling(1./255)(inputs)
    """
    for segmentation task we do downsampling by using the stride argument instead of maxpooling
    because we care a lot about the spatial location of information in the image, since we need to
    produce per-pixel target masks as output of the model
    """
    x = layers.Conv2D(64, 3, strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(128, 3, strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(256, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(256, 3, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(256, 3, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(
        256, 3, activation="relu", padding="same", strides=2)(x)
    """
    the purpose of this first half is to encode the images into smaller feature maps,
    where each spatial location (or pixel) contains infromation about a large spatial chunk 
    of the original image. The shape of x after the input layer is (200,200,3)
    and after this layer is (25,25,256)
    """  
    x = layers.Conv2DTranspose(128, 3, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(
        128, 3, activation="relu", padding="same", strides=2)(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(
        64, 3, activation="relu", padding="same", strides=2)(x)
  
    outputs = layers.Conv2D(num_classes, 3, activation="softmax",
     padding="same")(x)
 
    model = keras.Model(inputs, outputs)
    return model
  
