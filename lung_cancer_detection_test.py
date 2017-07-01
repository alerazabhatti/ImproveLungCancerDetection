
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

much_data = np.load('muchdata_testdata-50-50-20.npy')


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
model = tflearn.DNN(network, tensorboard_verbose=0)

from tflearn.data_utils import image_preloader
import numpy as np

# import numpy as np
# np.save('file.tfl', a)
# a = np.load('file.tfl')


model.load('file.tfl')


# # Load path/class_id image file:
# dataset_file = 'images_database.txt'


# # X_test, Y_test = image_preloader(dataset_file, image_shape=(32, 32), mode='file', categorical_labels=True, normalize=True, files_extension=['.jpg', '.png'], filter_channel=True)
# X_test = np.array(X_test)
# Y_test = np.array(Y_test)


# predict test images label
y_pred = model.predict(X)

# Compute accuracy of trained model on test images
print ("Accuracy: ",np.sum(np.argmax(y_pred, axis=1) == np.argmax(Y, axis=1))*100/np.array(Y).shape[0],"%")


