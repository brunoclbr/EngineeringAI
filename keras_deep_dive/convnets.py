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

import numpy as np