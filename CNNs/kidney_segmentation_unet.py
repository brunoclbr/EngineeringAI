import os
from sklearn.model_selection import train_test_split
import cv2
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import tifffile as tiff
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam
from keras.utils import normalize

#train data
directory_path_images = '/kaggle/input/blood-vessel-segmentation/train/kidney_1_dense/images/'
directory_path_labels = '/kaggle/input/blood-vessel-segmentation/train/kidney_1_dense/labels/'
tif_files_img = [f for f in os.listdir(directory_path_images) if f.endswith('.tif')]
tif_files_lab = [f for f in os.listdir(directory_path_labels) if f.endswith('.tif')]
# Read all TIFF images as a stack
large_image_stacks = tiff.imread([os.path.join(directory_path_images, file) for file in tif_files_img])
large_mask_stacks = tiff.imread([os.path.join(directory_path_labels, file) for file in tif_files_lab])
# train on a few slices
large_image_stack = large_image_stacks[:150,:,:]
large_mask_stack = large_mask_stacks[:150,:,:]
print(np.max(large_image_stacks))# (12, 1303, 902)

def pp_imgs(large_images_stack, test_case = False):  
    '''
    Image pre-processing 
    '''
    large_images_stack = np.array(large_images_stack)
    all_img_patches = []
    
    if not test_case:
        max_norm = np.max(large_images_stack)
        min_norm = np.min(large_images_stack)
        print(f'save this normalization numbers: max_norm is {max_norm} & min_norm is {min_norm}')
    
        for img in range(large_images_stack.shape[0]):
            
            large_image = large_images_stack[img,:,:]
            large_image = (large_image - min_norm)/(max_norm - min_norm)
            res = cv2.resize(large_image, dsize=(256,256), interpolation=cv2.INTER_CUBIC)
            if img == 1:
                print(res.shape)
            all_img_patches.append(res)
            
        all_img_patches = np.array(all_img_patches) 
        images = np.expand_dims(all_img_patches, -1)    
        return images
    else:
        max_norm = 65445
        min_norm = 0
        for img in range(large_images_stack.shape[0]):
            
            large_image = large_images_stack[img,:,:]
            large_image = (large_image - min_norm)/(max_norm - min_norm)
            res = cv2.resize(large_image, dsize=(256,256), interpolation=cv2.INTER_CUBIC)
            if img == 1:
                print(res.shape)
            all_img_patches.append(res)
            
        all_img_patches = np.array(all_img_patches) 
        images = np.expand_dims(all_img_patches, -1)    
        return images
    
images = pp_imgs(large_image_stack) 
masks = pp_imgs(large_mask_stack) 

X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size = 0.10, random_state = 0)

#Sanity check, view few mages
image_number = random.randint(0, len(X_train))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(X_train[image_number], (256,256)), cmap='gray')
plt.subplot(122)
plt.imshow(np.reshape(y_train[image_number], (256,256)), cmap='gray')
plt.show()

# create architecture
import u_net_model 

IMG_HEIGHT = images.shape[1]
IMG_WIDTH  = images.shape[2]
IMG_CHANNELS = images.shape[3]

input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
print(input_shape)
model = u_net_model.build_unet(input_shape)
model.compile(optimizer=Adam(lr = 1e-3), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

from keras.preprocessing.image import ImageDataGenerator
# data augmentation. when you apply data augmentation to images using parameters 
# like rotation_range, width_shift_range, height_shift_range, etc., you create modified copies of the original images
img_data_gen_args = dict(rotation_range=90,
                     width_shift_range=0.3,
                     #height_shift_range=0.3,
                     #shear_range=0.5,
                     zoom_range=0.3,
                     horizontal_flip=True,
                     #vertical_flip=True,
                     #fill_mode='reflect')
                        )

mask_data_gen_args = dict(rotation_range=90,
                     width_shift_range=0.3,
                     #height_shift_range=0.3,
                     #shear_range=0.5,
                     zoom_range=0.3,
                     horizontal_flip=True,
                     #vertical_flip=True,
                     #fill_mode='reflect',
                     preprocessing_function = lambda x: np.where(x>0, 1, 0).astype(x.dtype)) # Binarize the output again. 

# why am I giving batch_size here and then steps_per_epoch?
batch_size = 8
seed = 24
# Image Data
image_data_generator = ImageDataGenerator(**img_data_gen_args)
image_generator = image_data_generator.flow(X_train, seed=seed, batch_size=batch_size) # Takes data & label arrays, generates batches of augmented data.
valid_img_generator = image_data_generator.flow(X_test, seed=seed, batch_size=batch_size) 
# Mask Data
mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
mask_generator = mask_data_generator.flow(y_train, seed=seed, batch_size=batch_size)
valid_mask_generator = mask_data_generator.flow(y_test, seed=seed, batch_size=batch_size) 

def my_image_mask_generator(image_generator, mask_generator):
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        yield (img, mask)

my_generator = my_image_mask_generator(image_generator, mask_generator)

validation_datagen = my_image_mask_generator(valid_img_generator, valid_mask_generator)

x = image_generator.next()
y = mask_generator.next()
for i in range(0,1):
    image = x[i]
    mask = y[i]
    plt.subplot(1,2,1)
    plt.imshow(image[:,:,0], cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(mask[:,:,0])
    plt.show()

print(f'this is the length of my X_train: {len(X_train)}')
steps_per_epoch = 3*(len(X_train))//batch_size

#UserWarning: `Model.fit_generator` is deprecated and will be removed in a future version. Please use `Model.fit`, which supports generators.
# with this configuration 7 epochs were enough. Apply early stopping etc.
history = model.fit(my_generator, validation_data=validation_datagen, 
                    steps_per_epoch=steps_per_epoch, 
                    validation_steps=steps_per_epoch, epochs=8)


#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
#acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
#val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#IOU
y_pred = model.predict(X_test)
y_pred_thresholded = y_pred > 0.5

intersection = np.logical_and(y_test, y_pred_thresholded)
union = np.logical_or(y_test, y_pred_thresholded)
iou_score = np.sum(intersection) / np.sum(union)
print("IoU socre is: ", iou_score)

#Predict on a few images
#model = get_model()
#model.load_weights('mitochondria_50_plus_100_epochs.hdf5') #Trained for 50 epochs and then additional 100
#model.load_weights('mitochondria_gpu_tf1.4.hdf5')  #Trained for 50 epochs

# (240, 228, 228, 1) --> #, H, W, C
test_img_number = random.randint(0, len(X_test))
test_img = X_test[test_img_number]# slice of kidney
ground_truth=y_test[test_img_number]# these are the test mask I selected from train folder
test_img_norm=test_img[:,:,0][:,:,None]
test_img_input=np.expand_dims(test_img_norm, 0)
prediction = (model.predict(test_img_input)[0,:,:,0] > 0.2).astype(np.uint8)

plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='gray')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(prediction, cmap='gray')

plt.show()

# Specify the folder name
folder_name = 'predictive_models'

# Create a directory if it doesn't exist
output_dir = '/your_path/' + folder_name
os.makedirs(output_dir, exist_ok=True)

# Save your model
model.save(output_dir + '/model_150sp_8epch.keras')

