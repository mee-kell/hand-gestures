# PREPROCESSING DATA

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io
import os
import pickle
import sys

# Location of data
DATA_PATH = sys.argv[1]

# One-Hot encoded representations of hand positions
shape_to_label = {
    'thumbsup':np.array([1.,0.,0.]),
    'circle':np.array([0.,1.,0.]),
    'victory':np.array([0.,0.,1.])
}
arr_to_shape = {np.argmax(shape_to_label[x]):x for x in shape_to_label.keys()}

imgData = list()
labels = list()

for dr in os.listdir(DATA_PATH):

    # Folder name serves as label, which is converted to One-Hot representation
    if dr not in ['thumbsup', 'circle', 'victory']:
        continue
    print(dr)

    lb = shape_to_label[dr]
    i = 0
    
    # Iterate over images in each folder and return vector representation
    for pic in os.listdir(os.path.join(DATA_PATH, dr)):
        path = os.path.join(DATA_PATH, dr + '/' + pic)
        img = cv2.imread(path)

        # Data augmentation: create new images to increase dataset size
        # Horizontally flip and zoom in
        imgData.append([img, lb])
        imgData.append([cv2.flip(img, 1), lb])
        imgData.append([cv2.resize(img[50:250, 50:250], (300, 300)), lb])

        i += 3

    print(i)

# Images and labels are stored in separate arrays
np.random.shuffle(imgData)
imgData, labels = zip(*imgData)
imgData = np.array(imgData)
labels = np.array(labels)

# MODEL

from keras.models import Sequential, load_model
from keras.layers import Dense, MaxPool2D, Dropout, Flatten, Conv2D, GlobalAveragePooling2D, Activation
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications.densenet import DenseNet121

# Define inputs for the DenseNet model: 300 x 300 px, 3 RGB layers
densenet = DenseNet121(
    include_top = False, 
    weights = 'imagenet', 
    classes = 3,
    input_shape = (300, 300, 3)
)
densenet.trainable=True

# Use DenseNet layer as base, followed by our own Dense Neural Network
def genericModel(base):
    model = Sequential()
    model.add(base)
    model.add(MaxPool2D())
    model.add(Flatten())
    # Final layer: 3 neurons for each class
    model.add(Dense(3, activation='sigmoid'))
    model.compile(
        optimizer = "adam",
        loss = 'categorical_crossentropy', 
        metrics = ['acc']
    )
    # Returns probability of image belonging to a particular class
    return model

dnet = genericModel(densenet)

# Retrain the weights of the DenseNet for best results
checkpoint = ModelCheckpoint(
    'model.h5', 
    monitor = 'val_acc', 
    verbose = 1, 
    save_best_only = True, 
    save_weights_only = True,
    mode = 'auto'
)
es = EarlyStopping(patience = 3)
history = dnet.fit(
    x = imgData,
    y = labels,
    batch_size = 16,
    epochs = 8,
    callbacks = [checkpoint, es],
    validation_split = 0.2
)

# Store model architecture in a json file, store weights in .h5 file
dnet.save_weights('model.h5')
with open("model.json", "w") as json_file:
    json_file.write(dnet.to_json())

