"""
commands for downloading data on colab:
    !wget http:/ /www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
    !wget http:/ /www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
    !tar -xf images.tar.gz
    !tar -xf annotations.tar.gz
"""

import os 
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.utils import load_img, img_to_array
import numpy as np 
import random
import conv_seg_model as seg_model

input_dir = "images/"
target_dir = "annotations/trimaps"

input_img_paths = sorted(
    [os.path.join(input_dir, fname)
     for fname in os.listdir(input_dir) if fname.endswith(".jpg")]
)

target_paths = sorted(
    [os.path.join(target_dir, fname)
     for fname in os.listdir(target_dir) if fname.endswith(".jpg")]
)

plt.axis("off")
plt.imshow(load_img(input_img_paths[9])) #input nr 9

def display_target(target_array):
    normalized_array = (target_array.astype("uint8") -1)*127 # -1 so that labels go from 0-2 istead of 1-3 and *127 so that labels become 0, 127, 2554 (b,g,w)
    plt.axis("off")
    plt.imshow(normalized_array[:,:,0])

img = img_to_array(load_img(target_paths[9], color_mode="grayscale"))   
display_target(img)

img_size = (200, 200)
num_imgs = len(input_img_paths)
 
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_paths)
 
def path_to_input_image(path):
    return img_to_array(load_img(path, target_size=img_size))
  
def path_to_target(path):
    img = img_to_array(
        load_img(path, target_size=img_size, color_mode="grayscale"))
    img = img.astype("uint8") - 1
    return img
  
input_imgs = np.zeros((num_imgs,) + img_size + (3,), dtype="float32")
targets = np.zeros((num_imgs,) + img_size + (1,), dtype="uint8")
for i in range(num_imgs):
    input_imgs[i] = path_to_input_image(input_img_paths[i])
    targets[i] = path_to_target(target_paths[i])
 
num_val_samples = 1000
train_input_imgs = input_imgs[:-num_val_samples]
train_targets = targets[:-num_val_samples]
val_input_imgs = input_imgs[-num_val_samples:]
val_targets = targets[-num_val_samples:]

model = seg_model.get_model(img_size=img_size, num_classes=3) #functiono get_model() from script conv_seg_model
model.summary()

model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")
  
callbacks = [
    keras.callbacks.ModelCheckpoint("oxford_segmentation.keras",
                                    save_best_only=True)
]
  
history = model.fit(train_input_imgs, train_targets,
                    epochs=50,
                    callbacks=callbacks,
                    batch_size=64,
                    validation_data=(val_input_imgs, val_targets))

epochs = range(1, len(history.history["loss"]) + 1)
loss = history.history["loss"]
val_loss = history.history["val_loss"]
plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()