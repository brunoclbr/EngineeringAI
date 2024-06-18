"""
# Google Colab
!mkdir celeba_gan
!gdown --id 1O7m1010EJjLE5QxLZiM9Fpjs7Oj6e684 -O celeba_gan/data.zip
!unzip -qq celeba_gan/data.zip -d celeba_gan
"""
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt 

"""
The synthetic images are statisticllaly almost indisinguishable from the real ones [probably a lot of noise].
The latent space is not continuous. Here when applying gradient descent, every step taken down the hill changes the 
landscape itself a little.

gan(x) = discriminator(generator(x)) --> to train gen(x), we compute the loss of GAN w.r.t to the weights of GEN. 
This means, we set the loss as Y_real - gen(x) --> Y_real is a real image, not a fake, so it tunes the weights to get
more realistic images. del(gan)/del(w_gen) shows in direction of a minimum for the loss function, i.e. Y-gen should be 
as close as possible. The generator will be trained to predict "real" always, because we want the generator to create 
the best fake images. The discriminator will be trained with mixd images (real and fake).

Bag of tricks on page 404. A lot of heuristics for GAN.
"""

dataset = keras.utils.image_dataset_from_directory(
    "celeba_gan",
    label_mode=None, # i dont need labels for semi-supervised learning
    image_size=(64, 64),
    batch_size=32,
    smart_resize=True)

dataset = dataset.map(lambda x: x / 255.)

for x in dataset:
    plt.axis("off")
    plt.imshow((x.numpy() * 255).astype("int32")[0])
    break

# the discriminator
  
discriminator = keras.Sequential(
    [
        keras.Input(shape=(64, 64, 3)),
        layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Flatten(),
        layers.Dropout(0.2), # this dropout layer is an important trick so that gen does not get stuck with generated images that look like noise
        layers.Dense(1, activation="sigmoid"),
    ],
    name="discriminator",
)

# the generator 
latent_dim = 128
  
generator = keras.Sequential(
    [
        keras.Input(shape=(latent_dim,)),
        layers.Dense(8 * 8 * 128), # same number of coefficients we had at the level of the flatten layer in the encoder
        layers.Reshape((8, 8, 128)),
        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2), # for gans use leakyrelu to diminish sparsity!
        layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(3, kernel_size=5, padding="same", activation="sigmoid"),
    ],
    name="generator",
)

# the GAN model
# we'll use two optimizers, one for the gen and one for dis, so we also need to override the compile() method

class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()
        self.discriminator = discriminator #convnet
        self.generator = generator #convnet
        self.latent_dim = latent_dim
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")
 
    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile() #passing the subclassed model itself to super?
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
  
    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]
    
    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.latent_dim)) #sample random points in latent space normal distribution
        generated_images = self.generator(random_latent_vectors) # decoding into fake images (batch_size, 128) 
        combined_images = tf.concat([generated_images, real_images], axis=0)
        #assembling labels for discrimination between real and fake images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))],
            axis=0
        )
        labels += 0.05 * tf.random.uniform(tf.shape(labels)) #adding random noise to the LABELS, from bag of tricks
        
        # DISCRIMINATOR TRAINING
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions) # discriminator learns based on real labels
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )
  
        # GENERATOR TRAINING. Its important to first train the discriminator so that the predictions tend first to go in the right direction?
        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.latent_dim)) # sampls random points in the latent space
  
        misleading_labels = tf.zeros((batch_size, 1)) # labels saying "these are all real images!" (its a lie)

        with tf.GradientTape() as tape:
            predictions = self.discriminator(
                self.generator(random_latent_vectors)) # the predictions should all be fake
            g_loss = self.loss_fn(misleading_labels, predictions) 
        grads = tape.gradient(g_loss, self.generator.trainable_weights) # generator learns based on fake labels, forcing its weights to create better "fake" copies
        self.g_optimizer.apply_gradients(
            zip(grads, self.generator.trainable_weights))
  
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {"d_loss": self.d_loss_metric.result(),
                "g_loss": self.g_loss_metric.result()}
    
# monitor results
class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=3, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim
  
    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(
            shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255 
        generated_images.numpy()
        for i in range(self.num_img):
            img = keras.utils.array_to_img(generated_images[i])
            img.save(f"generated_img_{epoch:03d}_{i}.png")

# compiling and training the GAN Model
epochs = 100
  
gan = GAN(discriminator=discriminator, generator=generator,
          latent_dim=latent_dim)
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss_fn=keras.losses.BinaryCrossentropy(),
)
  
gan.fit(
    dataset, epochs=epochs,
    callbacks=[GANMonitor(num_img=10, latent_dim=latent_dim)]
)

# if the adversarial loss begins to increase considerably(generator is not producing good quality fakes),
# while the discriminator loss goes to zero, try reducing the discriminator learning rate, and increase 
# the dropout rate of the discriminator
