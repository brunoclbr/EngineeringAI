"""
VAEs are great for learning latent spaces that are well structured. Its trained by using as 
target data the *same input images*. The autoencoder learns to reconstruct the original inputs
(self supervised learning). 

Whenever ypu depart from classic supervised learning, its common to subclass the Model class 
and implement a custom train_step() to specify the new training logic. !!!

Classicaly autoencoders constrain the code to a low-dimensional and sparse (mostly zeros) code 
--> no so succesful, no continious information. VAEs augment autoencoders with small random noise tensor (eps), that forces 
them to learn CONTINIOUS, HIGHLY STRUCTURED latent spaces. 

1) input_img --> encoder --> z_mean & z_log_var
2) sample random point z --> z = z_mean + exp(z_log_var)*eps
3) z --> decoder --> reconstructed image

training via two loss functions: reconstruction loss for the decoded samples (minimizing the 
                                                distance between input and output image)
                                 regularization loss (first chapters this was e.g. weight penalty) 
                                                that helps to learn well-rounded latent distributions 
                                                and reduces overfitting. [maybe regularize the size/dimensions
                                                of the distribution of latent space, e.g. from 0 to 1]
                                                --> Kullback-Leibler nudges encoder distribution toward a
                                                well rounded normal distibution centered around 0.
                                
"""
import tensorflow as tf
from tensorflow import keras 
from keras import layers
import numpy as np

  
# Encoder layer

latent_dim = 2
  
encoder_inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(
    32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
# we use strides for downsampling feature maps instead of max pooling since 
# we care about information location within each image
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
# 64 features/filters, conv window kernel size = 3x3
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
# input image gets ENCODED into these two parameters
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var], name="encoder")

# random sampling of a latent space point z
  
class Sampler(layers.Layer):
    def call(self, z_mean, z_log_var):
        batch_size = tf.shape(z_mean)[0]
        z_size = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch_size, z_size))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
# decoder layer

latent_inputs = keras.Input(shape=(latent_dim,))
# produce the same number of coefficients that we had at the level of the flatten layer in the encoder
# stride = 2 reduced the input size by 2 when passing every layer
x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)
# output ends up with shape (28,28,1)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

# subclass the Model class  and implement a custom train_step() to specify the new training logic
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.sampler = Sampler()
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
  
    @property
    def metrics(self):
        """
        by listing the metrics in the metrics property we enable the model to reset them 
        after each epoch
        """
        return [self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker]
  
    def train_step(self, data):

        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(data)
            z = self.sampler(z_mean, z_log_var)
            reconstruction = decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(                      
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            total_loss = reconstruction_loss + tf.reduce_mean(kl_loss)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
# Instantiate the model
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
mnist_digits = np.concatenate([x_train, x_test], axis=0)
mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255 
  
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam(), run_eagerly=True) # no loss
vae.fit(mnist_digits, epochs=30, batch_size=128) # we dont pass targets since train_step is not expecting any

import matplotlib.pyplot as plt
  
n = 30
digit_size = 28 
figure = np.zeros((digit_size * n, digit_size * n))
# we use random points to create new images --> linspace function
# i cant predict a specific picture, but i can predict an array of pictures and choose the direction
# of the vector i'd like to use  
grid_x = np.linspace(-1, 1, n)
grid_y = np.linspace(-1, 1, n)[::-1]
  
for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        z_sample = np.array([[xi, yi]])
        x_decoded = vae.decoder.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[
            i * digit_size : (i + 1) * digit_size,
            j * digit_size : (j + 1) * digit_size,
        ] = digit
  
plt.figure(figsize=(15, 15))
start_range = digit_size // 2 
end_range = n * digit_size + start_range
pixel_range = np.arange(start_range, end_range, digit_size)
sample_range_x = np.round(grid_x, 1)
sample_range_y = np.round(grid_y, 1)
plt.xticks(pixel_range, sample_range_x)
plt.yticks(pixel_range, sample_range_y)
plt.xlabel("z[0]")
plt.ylabel("z[1]")
plt.axis("off")
plt.imshow(figure, cmap="Greys_r")