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
import random
#################################### DATA AUGMENTING SCRIPT ####################################

## Open data
IMAGE_SIZE = 32
CHANEL_SIZE = IMAGE_SIZE * IMAGE_SIZE

X = pd.read_csv('../data/Xtr.csv', header=None)
X = X.as_matrix()
X = X[:, 0:-1]

Y = pd.read_csv('../data/Ytr.csv')
Y = Y.as_matrix()

#%% Apply stochastic transformation
def data_augmentation(X, Y, k):
    """
    X: dataset to augmented
    Y: labels
    k: number of loops
    """
    augmented_X = np.zeros((k * X.shape[0], IMAGE_SIZE*IMAGE_SIZE*3))
    augmented_Y = np.zeros((k * X.shape[0], 2))
    for i in range(k):
        for index in range(X.shape[0]):
            r = random.uniform(0,5)
            image = X[index,:]
            image = image.reshape((3, CHANEL_SIZE))
            image = image.reshape((3, IMAGE_SIZE, IMAGE_SIZE))
            image = image.swapaxes(0,1)
            image = image.swapaxes(1,2)
            if(r<1):
                image_tf = dtools.rotate(image,1)
            elif(r<2 and r>=1):
                image_tf = dtools.rotate(image,2)
            elif(r<3 and r>=2):
                image_tf = dtools.rotate(image,3)
            elif(r<4 and r>=3):
                image_tf = dtools.blur(image, sigma=0.5)
            elif(r<5 and r>=4):
                image_tf = dtools.noise(image,sigma=0.01)
            image = image.reshape((1, IMAGE_SIZE*IMAGE_SIZE*3))

            augmented_X[index + i*X.shape[0],:] = image
            augmented_Y[index + i*X.shape[0],0] = index + (i+1)*X.shape[0]
            augmented_Y[index + i*X.shape[0],1] = Y[index,1]
