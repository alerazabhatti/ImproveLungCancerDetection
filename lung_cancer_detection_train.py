
from __future__ import division, print_function, absolute_import
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt



import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

much_data = np.load('muchdata-50-50-20.npy')
# If you are working with the basic sample data, use maybe 2 instead of 100 here... you don't have enough data to really do this
train_data = much_data[:-100]
validation_data = much_data[-100:]

#training data load
# X = much_data[0:2][0]
X = []
Y = []
for data in much_data:
    X.append(data[0])
    Y.append(data[1])
print(np.array(X).shape)
print(np.array(Y).shape)

#Convolutional network building
## To Do.
    ## Define your network here
network = input_data(shape=[None, 20, 50, 50])
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 256, activation='relu')
network = dropout(network, 0.75)
network = fully_connected(network, 256, activation='relu')
network = dropout(network, 0.75)
network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)


# Train using classifier
## To Do
    ## Define model and assign network
    ## Call the fit function for training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=5, shuffle=True, validation_set=0.1, show_metric=True, batch_size=2, run_id='cancer_cnn')
# Manually save model
## To Do
## Save model
model.save('file.tfl')


