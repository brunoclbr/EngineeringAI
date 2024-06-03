"""
Convnets are basically stacked convolutional and pooling layers that use as input 3D-Tensors with
size (img_height, img_width, img_channels). The first two dimensions tend to shrink as you go deeper into the model
- the channels are e.g 1 (black and grey) of 3 (RGB)
- later size (patch_dim1, patch_dim2, filter_number/output_depth) and a filter could be "*presence* of a face"

The conv operations learns patterns locally with small 2D windows, unlike dense layers, that use all
the pixels from the image at once. Convnets can localize pattern that was found in the lower right corner
of the picture at any location. Convnets learn patterns that are translation invariant and their spatial hierarchies.
So the patterns they learn (e.g edges) can be localized everywhere in the picture.

padding: valid and same to get (or not get) all the tiles of an image after covolution
    stride: distance between two succesive windows. 

max pooling: aka downsampling. Idea is to reduce number of trainable parameters and to induce spatial-filter
hierarchis by making succesive conv layers to look at increasingly large windows. Basically without there is too much
information to learn from

Work with pretrained models: feature extraction and fine-tuning
    -feature extraction: using representations learned by previous nets. For convnets, we take the conv base
    of the network and add a new classifier layer(s) at the end. Presence maps of convnets are generic maps of generic fetures
    of images. 

    -fine tuning: Either freeze the old model you're using and the add a dense layer(s) on top of it and train, or, 
    freeze the first layers of the old model and train the remaining layers with the new classifier together.
"""

import os, shutil, pathlib
import conv_model1
from tensorflow import keras
from keras.utils import image_dataset_from_directory # set up data pipeline
import matplotlib.pyplot as plt

original_dir = pathlib.Path("train")
new_base_dir = pathlib.Path("cats_vs_dogs_small")

def make_subset(subset_name, start_index, end_index):
    for category in ("cat", "dog"):
        dir = new_base_dir / subset_name / category #this will throw an error
        os.makedirs(dir)
        fnames = [f"{category}.{i}.jpg" 
                  for i in range(start_index, end_index)]
        for fname in fnames:
            shutil.copyfile(src=original_dir / fname, dst=dir / fname)

make_subset("train", start_index=0, end_index=1000)
make_subset("validation", start_index=1000, end_index=1500)
make_subset("test", start_index=1500, end_index=2500)

model = conv_model1

# transform images into preprocessed floating-point tensors with a generator
# 1) read files 2) JPEG --> RGB grid of pixels 3) convert into floating points 4) resize 5) pack into batches

train_dataset = image_dataset_from_directory(
    new_base_dir / "train",
    image_size= (180, 180),
    batch_size=32
)

validation_dataset = image_dataset_from_directory(
    new_base_dir / "validation",
    image_size= (180, 180),
    batch_size=32
)

test_dataset = image_dataset_from_directory(
    new_base_dir / "test",
    image_size= (180, 180),
    batch_size=32
)

# quick look into one Dataset object
for data_batch, labels_batch in train_dataset:
    print("data batch shape:", data_batch.shape)
    print("labels batch shape:", labels_batch.shape)
    break

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="convnet_from_scratch.keras",
        save_best_only=True,
        monitor="val_loss"
    )
]

history = model.fit(
    train_dataset,
    epochs=100,
    validation_data=validation_dataset,
    callbacks=callbacks
)

accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, "bo", label="Training accuracy")
plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")

test_model = keras.models.load_model("convnet_from_scratch.keras")
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.2f}")