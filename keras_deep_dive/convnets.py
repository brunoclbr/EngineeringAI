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

Create new branch to practice Git???

And practice coding with the book and collab?
"""

import os, shutil, pathlib
import conv_model1
from tensorflow import keras
from keras.utils import image_dataset_from_directory # set up data pipeline

original_dir = pathlib.Path("train")
new_base_dir = pathlib.Path("cats_vs_dogs_small")

def make_subset(subset_name, start_index, end_index):
    for category in ("cat", "dog"):
        dir = new_base_dir / subset_name / category 
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
    epochs=30,
    validation_data=validation_dataset,
    callbacks=callbacks
)