# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 18:34:05 2017

@author: Thomas PESNEAU
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import pandas as pd
import data_augmenting_tools as dtools
#################################### DATA AUGMENTING SCRIPT ####################################

### Open datacs
IMAGE_SIZE = 32
CHANEL_SIZE = IMAGE_SIZE * IMAGE_SIZE
#
#X = pd.read_csv('../../data/Xtr.csv', header=None)
#X = X.as_matrix()
#X = X[:, 0:-1]
#
#Y = pd.read_csv('../../data/Ytr.csv')
#Y = Y.as_matrix()

#%% Apply stochastic transformation
def data_augmentation(X, Y, k):
    """
    X: dataset to augmented
    Y: labels
    k: number of loops
    """
    augmented_X = np.zeros((k * X.shape[0], 24*24*3))
    augmented_Y = np.zeros((k * X.shape[0], 2))
    for i in range(k):
        for index in range(X.shape[0]):
            image = X[index,:]
            image = image.reshape((3, CHANEL_SIZE))
            image = image.reshape((3, IMAGE_SIZE, IMAGE_SIZE))
            image = image.swapaxes(0,1)
            image = image.swapaxes(1,2)
            
            if(i != 1):
                image = dtools.random_rotate(image)
                image = dtools.random_flip(image)
            else:
                image = image[5:29,5:29,:]
            image = image.flatten()

            augmented_X[index + i*X.shape[0],:] = image
            augmented_Y[index + i*X.shape[0],0] = index + (i)*X.shape[0] + 1
            augmented_Y[index + i*X.shape[0],1] = Y[index,1]
    return augmented_X, augmented_Y
